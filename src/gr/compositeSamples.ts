import { hamiltonian } from './geodesic';
import { type KerrSchildParams, type Vec3, type Vec4 } from './kerrSchild';
import { type DiskRadianceModel } from './radiance';
import { statusToReadback } from './readback';
import { COMPOSITE_INPUT_FLOATS_PER_RAY, COMPOSITE_OUTPUT_FLOATS_PER_RAY } from './compositeReadback';
import { launchPhotonFromTetrad, type GrTetrad } from './tetrad';
import { type ProbeGrid } from './referenceProbe';

export const COMPOSITE_CAMERA_UNIFORM_FLOATS = 64;

export interface CompositeTraceOptions {
  stepSize: number;
  maxSteps: number;
  escapeRadius: number;
  singularityRadius: number;
}

export interface CompositeCameraSampleOptions {
  width: number;
  height: number;
  position: Vec3;
  tetrad: GrTetrad;
  observerVelocity: Vec4;
  params: KerrSchildParams;
  verticalFovRadians: number;
  traceOptions: CompositeTraceOptions;
  disk: {
    innerRadius: number;
    outerRadius: number;
  };
  radianceModel: DiskRadianceModel;
}

export interface CompositeRaySample {
  position: Vec3;
  momentum: Vec4;
  observerVelocity: Vec4;
  params: KerrSchildParams;
  traceOptions: CompositeTraceOptions;
  disk: {
    innerRadius: number;
    outerRadius: number;
  };
  radianceModel: DiskRadianceModel;
}

export function createCompositeCameraUniforms(options: CompositeCameraSampleOptions): Float32Array {
  const uniforms = new Float32Array(COMPOSITE_CAMERA_UNIFORM_FLOATS);
  packVec4(uniforms, 0, { t: 0, x: options.position.x, y: options.position.y, z: options.position.z });
  packVec4(uniforms, 4, options.observerVelocity);
  packVec4(uniforms, 8, options.tetrad.eTime);
  packVec4(uniforms, 12, options.tetrad.eRight);
  packVec4(uniforms, 16, options.tetrad.eUp);
  packVec4(uniforms, 20, options.tetrad.eForward);
  uniforms[24] = options.params.spin;
  uniforms[25] = options.traceOptions.stepSize;
  uniforms[26] = options.traceOptions.escapeRadius;
  uniforms[27] = options.traceOptions.singularityRadius;
  uniforms[28] = options.traceOptions.maxSteps;
  uniforms[29] = options.params.mass;
  uniforms[30] = options.width;
  uniforms[31] = options.height;
  uniforms[32] = options.disk.innerRadius;
  uniforms[33] = options.disk.outerRadius;
  uniforms[34] = options.radianceModel.innerTemperature;
  uniforms[35] = options.radianceModel.emissivityScale;
  uniforms[36] = options.radianceModel.boostPower;
  uniforms[37] = options.radianceModel.spinDirection ?? 1;
  uniforms[38] = Math.tan(options.verticalFovRadians * 0.5);
  uniforms[39] = options.width / options.height;
  uniforms[40] = options.radianceModel.emissionPhase ?? 0;
  return uniforms;
}

export function createCompositeCameraSamples(options: CompositeCameraSampleOptions): Float32Array {
  const samples = new Float32Array(options.width * options.height * COMPOSITE_INPUT_FLOATS_PER_RAY);
  const aspect = options.width / options.height;
  const tanHalfFov = Math.tan(options.verticalFovRadians * 0.5);

  for (let y = 0; y < options.height; y++) {
    for (let x = 0; x < options.width; x++) {
      const index = y * options.width + x;
      const ndcX = (2 * (x + 0.5) / options.width - 1) * aspect;
      const ndcY = 1 - 2 * (y + 0.5) / options.height;
      const localDirection = normalize3({ x: ndcX * tanHalfFov, y: ndcY * tanHalfFov, z: 1 });
      const momentum = launchPhotonFromTetrad(options.position, options.params, options.tetrad, localDirection);
      packCompositeRaySample(samples, index, {
        position: options.position,
        momentum,
        observerVelocity: options.observerVelocity,
        params: options.params,
        traceOptions: options.traceOptions,
        disk: options.disk,
        radianceModel: options.radianceModel,
      });
    }
  }

  return samples;
}

export function createCompositeSamplesFromProbeGrid(
  grid: ProbeGrid,
  options: Omit<CompositeCameraSampleOptions, 'width' | 'height' | 'verticalFovRadians'>,
): Float32Array {
  const samples = new Float32Array(grid.rays.length * COMPOSITE_INPUT_FLOATS_PER_RAY);
  grid.rays.forEach((ray, index) => {
    const momentum = launchPhotonFromTetrad(options.position, options.params, options.tetrad, ray.localDirection);
    packCompositeRaySample(samples, index, {
      position: options.position,
      momentum,
      observerVelocity: options.observerVelocity,
      params: options.params,
      traceOptions: options.traceOptions,
      disk: options.disk,
      radianceModel: options.radianceModel,
    });
  });
  return samples;
}

export function createCompositeExpectedFromProbeGrid(grid: ProbeGrid): Float32Array {
  const expected = new Float32Array(grid.rays.length * COMPOSITE_OUTPUT_FLOATS_PER_RAY);
  grid.rays.forEach((ray, index) => {
    const base = index * COMPOSITE_OUTPUT_FLOATS_PER_RAY;
    expected[base] = statusToReadback(ray.status);
    expected[base + 1] = ray.steps;
    expected[base + 2] = ray.finalRadius;
    expected[base + 3] = ray.diskHit?.radius ?? -1;
    expected[base + 4] = ray.color[0];
    expected[base + 5] = ray.color[1];
    expected[base + 6] = ray.color[2];
    expected[base + 7] = ray.maxHamiltonianDrift;
  });
  return expected;
}

export function packCompositeRaySample(
  samples: Float32Array<ArrayBufferLike>,
  index: number,
  sample: CompositeRaySample,
): void {
  const base = index * COMPOSITE_INPUT_FLOATS_PER_RAY;
  samples[base] = 0;
  samples[base + 1] = sample.position.x;
  samples[base + 2] = sample.position.y;
  samples[base + 3] = sample.position.z;
  samples[base + 4] = sample.momentum.t;
  samples[base + 5] = sample.momentum.x;
  samples[base + 6] = sample.momentum.y;
  samples[base + 7] = sample.momentum.z;
  samples[base + 8] = sample.observerVelocity.t;
  samples[base + 9] = sample.observerVelocity.x;
  samples[base + 10] = sample.observerVelocity.y;
  samples[base + 11] = sample.observerVelocity.z;
  samples[base + 12] = sample.params.spin;
  samples[base + 13] = sample.traceOptions.stepSize;
  samples[base + 14] = sample.traceOptions.escapeRadius;
  samples[base + 15] = sample.traceOptions.singularityRadius;
  samples[base + 16] = sample.traceOptions.maxSteps;
  samples[base + 17] = sample.params.mass;
  samples[base + 18] = 0;
  samples[base + 19] = 0;
  samples[base + 20] = sample.disk.innerRadius;
  samples[base + 21] = sample.disk.outerRadius;
  samples[base + 22] = sample.radianceModel.innerTemperature;
  samples[base + 23] = sample.radianceModel.emissivityScale;
  samples[base + 24] = sample.radianceModel.boostPower;
  samples[base + 25] = sample.radianceModel.spinDirection ?? 1;
  samples[base + 26] = sample.radianceModel.emissionPhase ?? 0;
  samples[base + 27] = 0;
}

export function compositeSampleHamiltonian(
  samples: Float32Array<ArrayBufferLike>,
  index: number,
  params: KerrSchildParams,
): number {
  const base = index * COMPOSITE_INPUT_FLOATS_PER_RAY;
  return hamiltonian(
    {
      position: {
        t: samples[base],
        x: samples[base + 1],
        y: samples[base + 2],
        z: samples[base + 3],
      },
      momentum: {
        t: samples[base + 4],
        x: samples[base + 5],
        y: samples[base + 6],
        z: samples[base + 7],
      },
    },
    params,
  );
}

function normalize3(v: Vec3): Vec3 {
  const length = Math.hypot(v.x, v.y, v.z) || 1;
  return { x: v.x / length, y: v.y / length, z: v.z / length };
}

function packVec4(target: Float32Array, offset: number, value: Vec4): void {
  target[offset] = value.t;
  target[offset + 1] = value.x;
  target[offset + 2] = value.y;
  target[offset + 3] = value.z;
}
