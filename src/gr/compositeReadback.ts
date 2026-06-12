export const COMPOSITE_INPUT_FLOATS_PER_RAY = 28;
export const COMPOSITE_OUTPUT_FLOATS_PER_RAY = 16;
export const COMPOSITE_FLOAT_BYTES = Float32Array.BYTES_PER_ELEMENT;
export const COMPOSITE_DETAIL_DIFF_THRESHOLD = 5e-2;

export interface CompositeOutputRow {
  status: number;
  steps: number;
  radius: number;
  diskRadius: number;
  color: [number, number, number];
  drift: number;
  skyDirection: [number, number, number];
  skyMix: number;
  diskColor: [number, number, number];
}

export interface CompositeComparison {
  maxAbsDiff: number;
  statusMismatches: number;
  diskMismatches: number;
}

export function compositeRayCount(samples: Float32Array<ArrayBufferLike>): number {
  return samples.length / COMPOSITE_INPUT_FLOATS_PER_RAY;
}

export function compositeOutputByteLength(samples: Float32Array<ArrayBufferLike>): number {
  return compositeRayCount(samples) * COMPOSITE_OUTPUT_FLOATS_PER_RAY * COMPOSITE_FLOAT_BYTES;
}

export function compareCompositeReadback(
  expected: Float32Array<ArrayBufferLike>,
  output: Float32Array<ArrayBufferLike>,
): CompositeComparison {
  return {
    maxAbsDiff: maxAbsDiff(expected, output),
    statusMismatches: mismatchCount(expected, output, COMPOSITE_OUTPUT_FLOATS_PER_RAY, 0),
    diskMismatches: mismatchCount(expected, output, COMPOSITE_OUTPUT_FLOATS_PER_RAY, 3),
  };
}

export function compositeOutputRows(output: Float32Array<ArrayBufferLike>): CompositeOutputRow[] {
  const rows: CompositeOutputRow[] = [];
  for (let i = 0; i < output.length; i += COMPOSITE_OUTPUT_FLOATS_PER_RAY) {
    rows.push({
      status: output[i],
      steps: output[i + 1],
      radius: output[i + 2],
      diskRadius: output[i + 3],
      color: [output[i + 4], output[i + 5], output[i + 6]],
      drift: output[i + 7],
      skyDirection: [output[i + 8], output[i + 9], output[i + 10]],
      skyMix: output[i + 11],
      diskColor: [output[i + 12], output[i + 13], output[i + 14]],
    });
  }
  return rows;
}

export function compositeProbeDetail(
  expected: Float32Array<ArrayBufferLike>,
  output: Float32Array<ArrayBufferLike>,
): string {
  for (let i = 0; i < expected.length; i += COMPOSITE_OUTPUT_FLOATS_PER_RAY) {
    if (Math.round(expected[i]) !== Math.round(output[i])) {
      return (
        `, row ${i / COMPOSITE_OUTPUT_FLOATS_PER_RAY} ` +
        `expected [${formatVecN(expected, i, COMPOSITE_OUTPUT_FLOATS_PER_RAY)}] ` +
        `got [${formatVecN(output, i, COMPOSITE_OUTPUT_FLOATS_PER_RAY)}]`
      );
    }
  }

  let max = 0;
  let offset = 0;
  for (let i = 0; i < expected.length; i++) {
    const diff = Math.abs(expected[i] - output[i]);
    if (diff > max) {
      max = diff;
      offset = i - i % COMPOSITE_OUTPUT_FLOATS_PER_RAY;
    }
  }

  return max > COMPOSITE_DETAIL_DIFF_THRESHOLD
    ? `, max row ${offset / COMPOSITE_OUTPUT_FLOATS_PER_RAY} ` +
        `expected [${formatVecN(expected, offset, COMPOSITE_OUTPUT_FLOATS_PER_RAY)}] ` +
        `got [${formatVecN(output, offset, COMPOSITE_OUTPUT_FLOATS_PER_RAY)}]`
    : '';
}

function maxAbsDiff(a: Float32Array<ArrayBufferLike>, b: Float32Array<ArrayBufferLike>): number {
  let max = 0;
  const count = Math.min(a.length, b.length);
  for (let i = 0; i < count; i++) {
    max = Math.max(max, Math.abs(a[i] - b[i]));
  }
  return max;
}

function mismatchCount(
  a: Float32Array<ArrayBufferLike>,
  b: Float32Array<ArrayBufferLike>,
  stride: number,
  offset: number,
): number {
  let count = 0;
  const countableLength = Math.min(a.length, b.length);
  for (let i = offset; i < countableLength; i += stride) {
    if (Math.round(a[i]) !== Math.round(b[i])) count += 1;
  }
  return count;
}

function formatVecN(values: Float32Array<ArrayBufferLike>, offset: number, length: number): string {
  return Array.from({ length }, (_, i) => values[offset + i].toPrecision(4)).join(', ');
}
