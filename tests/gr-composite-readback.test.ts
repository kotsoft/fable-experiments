import { describe, expect, it } from 'vitest';
import {
  COMPOSITE_FLOAT_BYTES,
  COMPOSITE_INPUT_FLOATS_PER_RAY,
  COMPOSITE_OUTPUT_FLOATS_PER_RAY,
  compareCompositeReadback,
  compositeOutputByteLength,
  compositeOutputRows,
  compositeProbeDetail,
  compositeRayCount,
} from '../src/gr/compositeReadback';
import { ReadbackStatus } from '../src/gr/readback';

describe('composite WebGPU readback schema', () => {
  it('derives ray count and output byte length from the shared input stride', () => {
    const samples = new Float32Array(COMPOSITE_INPUT_FLOATS_PER_RAY * 3);

    expect(compositeRayCount(samples)).toBe(3);
    expect(compositeOutputByteLength(samples)).toBe(3 * COMPOSITE_OUTPUT_FLOATS_PER_RAY * COMPOSITE_FLOAT_BYTES);
  });

  it('unpacks composite output rows by semantic field', () => {
    const output = new Float32Array([
      ReadbackStatus.Disk,
      42,
      5.5,
      5.5,
      1.25,
      0.5,
      0.25,
      1e-4,
      0,
      0,
      1,
      1,
      1.25,
      0.5,
      0.25,
      0,
    ]);

    const [row] = compositeOutputRows(output);

    expect(row.status).toBe(ReadbackStatus.Disk);
    expect(row.steps).toBe(42);
    expect(row.radius).toBe(5.5);
    expect(row.diskRadius).toBe(5.5);
    expect(row.color).toEqual([1.25, 0.5, 0.25]);
    expect(row.drift).toBeCloseTo(1e-4);
    expect(row.skyDirection).toEqual([0, 0, 1]);
    expect(row.skyMix).toBe(1);
    expect(row.diskColor).toEqual([1.25, 0.5, 0.25]);
  });

  it('compares numeric GPU output against CPU reference diagnostics', () => {
    const expected = new Float32Array([
      ReadbackStatus.Disk,
      10,
      6,
      6,
      1,
      0.5,
      0.25,
      1e-4,
      0,
      0,
      1,
      1,
      1,
      0.5,
      0.25,
      0,
      ReadbackStatus.Horizon,
      20,
      1.4,
      -1,
      0,
      0,
      0,
      2e-4,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
    ]);
    const output = new Float32Array(expected);
    output[4] += 0.01;
    output[16] = ReadbackStatus.Escaped;
    output[19] = 12;

    const comparison = compareCompositeReadback(expected, output);

    expect(comparison.maxAbsDiff).toBe(13);
    expect(comparison.statusMismatches).toBe(1);
    expect(comparison.diskMismatches).toBe(1);
    expect(compositeProbeDetail(expected, output)).toContain('row 1');
  });
});
