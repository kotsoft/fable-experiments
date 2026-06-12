import { type KerrSchildParams } from './kerrSchild';

export interface ShadowPoint {
  alpha: number;
  beta: number;
  sphericalOrbitRadius: number;
}

export interface ShadowBounds {
  minAlpha: number;
  maxAlpha: number;
  minBeta: number;
  maxBeta: number;
  centerAlpha: number;
  horizontalDiameter: number;
  verticalDiameter: number;
}

export function schwarzschildCriticalImpactParameter(params: KerrSchildParams): number {
  return 3 * Math.sqrt(3) * params.mass;
}

export function equatorialPhotonOrbitRadii(params: KerrSchildParams): { prograde: number; retrograde: number } {
  if (params.mass <= 0) return { prograde: 0, retrograde: 0 };
  const normalizedSpin = Math.min(Math.abs(params.spin) / params.mass, 1);
  return {
    prograde: 2 * params.mass * (1 + Math.cos((2 / 3) * Math.acos(-normalizedSpin))),
    retrograde: 2 * params.mass * (1 + Math.cos((2 / 3) * Math.acos(normalizedSpin))),
  };
}

export function kerrShadowBoundary(
  params: KerrSchildParams,
  observerInclinationRadians: number,
  samples = 128,
): ShadowPoint[] {
  if (params.mass <= 0) return [];
  const spinMagnitude = Math.abs(params.spin);
  if (spinMagnitude < 1e-10) {
    return circularShadowBoundary(schwarzschildCriticalImpactParameter(params), samples);
  }

  const inclination = clamp(observerInclinationRadians, 1e-4, Math.PI - 1e-4);
  const sinInclination = Math.sin(inclination);
  const cotInclination = Math.cos(inclination) / sinInclination;
  const { prograde, retrograde } = equatorialPhotonOrbitRadii(params);
  const low = Math.min(prograde, retrograde);
  const high = Math.max(prograde, retrograde);
  const points: ShadowPoint[] = [];

  for (let i = 0; i < samples; i++) {
    const radius = low + (high - low) * (i / Math.max(samples - 1, 1));
    const constants = sphericalPhotonOrbitConstants(radius, params.mass, params.spin);
    const betaSquared =
      constants.eta +
      params.spin * params.spin * Math.cos(inclination) ** 2 -
      constants.xi * constants.xi * cotInclination * cotInclination;
    if (betaSquared < -1e-9) continue;

    const alpha = -constants.xi / sinInclination;
    const beta = Math.sqrt(Math.max(betaSquared, 0));
    points.push({ alpha, beta, sphericalOrbitRadius: radius });
    if (beta > 1e-9) points.push({ alpha, beta: -beta, sphericalOrbitRadius: radius });
  }

  return points;
}

export function shadowBounds(points: ShadowPoint[]): ShadowBounds {
  if (points.length === 0) {
    return {
      minAlpha: 0,
      maxAlpha: 0,
      minBeta: 0,
      maxBeta: 0,
      centerAlpha: 0,
      horizontalDiameter: 0,
      verticalDiameter: 0,
    };
  }

  const minAlpha = Math.min(...points.map((point) => point.alpha));
  const maxAlpha = Math.max(...points.map((point) => point.alpha));
  const minBeta = Math.min(...points.map((point) => point.beta));
  const maxBeta = Math.max(...points.map((point) => point.beta));

  return {
    minAlpha,
    maxAlpha,
    minBeta,
    maxBeta,
    centerAlpha: 0.5 * (minAlpha + maxAlpha),
    horizontalDiameter: maxAlpha - minAlpha,
    verticalDiameter: maxBeta - minBeta,
  };
}

function sphericalPhotonOrbitConstants(
  radius: number,
  mass: number,
  spin: number,
): { xi: number; eta: number } {
  const denominator = spin * (mass - radius);
  const xi = (radius * radius * (radius - 3 * mass) + spin * spin * (radius + mass)) / denominator;
  const eta =
    (radius ** 3 * (4 * spin * spin * mass - radius * (radius - 3 * mass) ** 2)) /
    (spin * spin * (mass - radius) ** 2);
  return { xi, eta };
}

function circularShadowBoundary(radius: number, samples: number): ShadowPoint[] {
  return Array.from({ length: samples }, (_, index) => {
    const angle = 2 * Math.PI * (index / samples);
    return {
      alpha: radius * Math.cos(angle),
      beta: radius * Math.sin(angle),
      sphericalOrbitRadius: 3 * radius / Math.sqrt(27),
    };
  });
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
