import { type GeodesicState } from './geodesic';
import {
  dot3,
  kerrSchildRadius,
  metricDot,
  type KerrSchildParams,
  type Vec3,
  type Vec4,
} from './kerrSchild';

export interface DiskRadianceModel {
  innerRadius: number;
  outerRadius: number;
  innerTemperature: number;
  emissivityScale: number;
  boostPower: number;
  spinDirection?: 1 | -1;
}

export interface DiskRadianceSample {
  radius: number;
  temperature: number;
  redshift: number;
  emitterVelocity: Vec4;
  emittedRgb: [number, number, number];
  observedRgb: [number, number, number];
  bolometricIntensity: number;
}

export function sampleDiskRadiance(
  rayState: GeodesicState,
  params: KerrSchildParams,
  model: DiskRadianceModel,
  observerVelocity?: Vec4,
): DiskRadianceSample | null {
  const position = spatialPosition(rayState.position);
  const radius = kerrSchildRadius(position, params);
  if (radius < model.innerRadius || radius > model.outerRadius) return null;

  const emitterVelocity = diskEmitterFourVelocity(position, params, model.spinDirection ?? 1);
  const redshift = redshiftFactor(rayState.momentum, emitterVelocity, observerVelocity);
  const temperature = model.innerTemperature * Math.pow(radius / model.innerRadius, -0.75);
  const emittedRgb = blackbodyRgb(temperature);
  const radialFalloff = Math.pow(radius / model.innerRadius, -2.4);
  const bolometricIntensity = model.emissivityScale * radialFalloff * Math.pow(Math.max(redshift, 0), model.boostPower);
  const observedRgb = scaleRgb(emittedRgb, bolometricIntensity);

  return {
    radius,
    temperature,
    redshift,
    emitterVelocity,
    emittedRgb,
    observedRgb,
    bolometricIntensity,
  };
}

export function redshiftFactor(photonCovector: Vec4, emitterVelocity: Vec4, observerVelocity?: Vec4): number {
  const emitterFrequency = -covectorContraction(photonCovector, emitterVelocity);
  const observerFrequency = observerVelocity
    ? -covectorContraction(photonCovector, observerVelocity)
    : -photonCovector.t;
  if (emitterFrequency <= 0 || observerFrequency <= 0) return 0;
  return observerFrequency / emitterFrequency;
}

export function diskEmitterFourVelocity(position: Vec3, params: KerrSchildParams, spinDirection: 1 | -1 = 1): Vec4 {
  const radius = Math.max(kerrSchildRadius(position, params), 1e-6);
  const radial = normalize3({ x: position.x, y: position.y, z: 0 });
  const tangent = { x: -spinDirection * radial.y, y: spinDirection * radial.x, z: 0 };
  let betaMagnitude = Math.min(Math.sqrt(params.mass / Math.max(radius - 2 * params.mass, 1.5)), 0.75);
  for (let i = 0; i < 12; i++) {
    const beta = scale3(tangent, betaMagnitude);
    const gamma = 1 / Math.sqrt(Math.max(1 - dot3(beta, beta), 1e-6));
    const flatGuess = { t: gamma, x: gamma * beta.x, y: gamma * beta.y, z: gamma * beta.z };
    if (metricDot(position, params, flatGuess, flatGuess) < -1e-8) {
      return normalizeTimelike(position, params, flatGuess);
    }
    betaMagnitude *= 0.75;
  }
  return normalizeTimelike(position, params, { t: 1, x: 0, y: 0, z: 0 });
}

export function blackbodyRgb(temperature: number): [number, number, number] {
  const t = Math.max(1000, Math.min(40000, temperature)) / 100;
  const red = t <= 66 ? 1 : clamp01(1.292936186 * Math.pow(t - 60, -0.1332047592));
  const green = t <= 66
    ? clamp01(0.3900815788 * Math.log(t) - 0.6318414438)
    : clamp01(1.129890861 * Math.pow(t - 60, -0.0755148492));
  const blue = t >= 66 ? 1 : t <= 19 ? 0 : clamp01(0.5432067891 * Math.log(t - 10) - 1.1962540891);
  return [red, green, blue];
}

function normalizeTimelike(position: Vec3, params: KerrSchildParams, vector: Vec4): Vec4 {
  const norm = metricDot(position, params, vector, vector);
  if (norm >= 0) throw new Error('Disk emitter velocity must be timelike');
  const normalized = scaleVec4(vector, 1 / Math.sqrt(-norm));
  return normalized.t < 0 ? scaleVec4(normalized, -1) : normalized;
}

function covectorContraction(covector: Vec4, vector: Vec4): number {
  return covector.t * vector.t + covector.x * vector.x + covector.y * vector.y + covector.z * vector.z;
}

function spatialPosition(position: Vec4): Vec3 {
  return { x: position.x, y: position.y, z: position.z };
}

function normalize3(v: Vec3): Vec3 {
  const length = Math.hypot(v.x, v.y, v.z) || 1;
  return { x: v.x / length, y: v.y / length, z: v.z / length };
}

function scale3(v: Vec3, scale: number): Vec3 {
  return { x: v.x * scale, y: v.y * scale, z: v.z * scale };
}

function scaleVec4(v: Vec4, scale: number): Vec4 {
  return { t: v.t * scale, x: v.x * scale, y: v.y * scale, z: v.z * scale };
}

function scaleRgb(rgb: [number, number, number], scale: number): [number, number, number] {
  return [rgb[0] * scale, rgb[1] * scale, rgb[2] * scale];
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}
