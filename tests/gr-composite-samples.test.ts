import { describe, expect, it } from 'vitest';
import {
  COMPOSITE_INPUT_FLOATS_PER_RAY,
  COMPOSITE_OUTPUT_FLOATS_PER_RAY,
} from '../src/gr/compositeReadback';
import {
  COMPOSITE_CAMERA_UNIFORM_FLOATS,
  compositeSampleHamiltonian,
  createCompositeCameraUniforms,
  createCompositeCameraSamples,
  createCompositeExpectedFromProbeGrid,
  createCompositeSamplesFromProbeGrid,
} from '../src/gr/compositeSamples';
import { kerrSchildParams } from '../src/gr/kerrSchild';
import { ReadbackStatus } from '../src/gr/readback';
import { renderProbeGrid } from '../src/gr/referenceProbe';
import { buildObserverTetrad, staticObserverFourVelocity } from '../src/gr/tetrad';

describe('composite ray sample packing', () => {
  it('packs camera uniforms with tetrad and render controls', () => {
    const params = kerrSchildParams(0.55, 1);
    const position = { x: 10, y: 0, z: 3 };
    const observerVelocity = staticObserverFourVelocity(position, params);
    const tetrad = buildObserverTetrad(position, params, observerVelocity);
    const uniforms = createCompositeCameraUniforms({
      width: 64,
      height: 36,
      position,
      tetrad,
      observerVelocity,
      params,
      verticalFovRadians: 0.82,
      traceOptions: {
        stepSize: 0.04,
        maxSteps: 520,
        escapeRadius: 32,
        singularityRadius: 0.2,
      },
      disk: { innerRadius: 3, outerRadius: 18 },
      radianceModel: {
        innerRadius: 3,
        outerRadius: 18,
        innerTemperature: 7200,
        emissivityScale: 1,
        boostPower: 4,
      },
    });

    expect(uniforms.length).toBe(COMPOSITE_CAMERA_UNIFORM_FLOATS);
    expect([...uniforms.slice(0, 4)]).toEqual([0, position.x, position.y, position.z]);
    expect(uniforms[24]).toBeCloseTo(params.spin);
    expect(uniforms[25]).toBeCloseTo(0.04);
    expect(uniforms[26]).toBe(32);
    expect(uniforms[27]).toBeCloseTo(0.2);
    expect([...uniforms.slice(28, 32)]).toEqual([520, params.mass, 64, 36]);
    expect([...uniforms.slice(32, 38)]).toEqual([3, 18, 7200, 1, 4, 1]);
    expect(uniforms[38]).toBeCloseTo(Math.tan(0.82 * 0.5));
    expect(uniforms[39]).toBeCloseTo(64 / 36);
  });

  it('packs camera-generated tetrad rays as null covectors', () => {
    const params = kerrSchildParams(0.55, 1);
    const position = { x: 10, y: 0, z: 3 };
    const observerVelocity = staticObserverFourVelocity(position, params);
    const tetrad = buildObserverTetrad(position, params, observerVelocity);
    const samples = createCompositeCameraSamples({
      width: 4,
      height: 3,
      position,
      tetrad,
      observerVelocity,
      params,
      verticalFovRadians: 0.82,
      traceOptions: {
        stepSize: 0.04,
        maxSteps: 520,
        escapeRadius: 32,
        singularityRadius: 0.2,
      },
      disk: { innerRadius: 3, outerRadius: 18 },
      radianceModel: {
        innerRadius: 3,
        outerRadius: 18,
        innerTemperature: 7200,
        emissivityScale: 1,
        boostPower: 4,
      },
    });

    expect(samples.length).toBe(4 * 3 * COMPOSITE_INPUT_FLOATS_PER_RAY);
    for (let i = 0; i < 12; i++) {
      expect(Math.abs(compositeSampleHamiltonian(samples, i, params))).toBeLessThan(5e-8);
    }
  });

  it('packs probe-grid samples and expected rows with matching ray counts', () => {
    const params = kerrSchildParams(0.55, 1);
    const position = { x: 10, y: 0, z: 3 };
    const observerVelocity = staticObserverFourVelocity(position, params);
    const tetrad = buildObserverTetrad(position, params, observerVelocity);
    const traceOptions = {
      stepSize: 0.04,
      maxSteps: 520,
      escapeRadius: 32,
      singularityRadius: 0.2,
    };
    const disk = { innerRadius: 3, outerRadius: 18 };
    const radianceModel = {
      innerRadius: 3,
      outerRadius: 18,
      innerTemperature: 7200,
      emissivityScale: 1,
      boostPower: 4,
    };
    const grid = renderProbeGrid(
      params,
      { position, tetrad, verticalFovRadians: 0.82 },
      5,
      3,
      traceOptions,
      disk,
      radianceModel,
    );

    const samples = createCompositeSamplesFromProbeGrid(grid, {
      position,
      tetrad,
      observerVelocity,
      params,
      traceOptions,
      disk,
      radianceModel,
    });
    const expected = createCompositeExpectedFromProbeGrid(grid);

    expect(samples.length).toBe(grid.rays.length * COMPOSITE_INPUT_FLOATS_PER_RAY);
    expect(expected.length).toBe(grid.rays.length * COMPOSITE_OUTPUT_FLOATS_PER_RAY);
    expect([...expected].filter((value, index) => index % COMPOSITE_OUTPUT_FLOATS_PER_RAY === 0 && value === ReadbackStatus.Disk).length).toBeGreaterThan(0);
  });
});
