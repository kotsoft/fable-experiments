import { describe, expect, it } from 'vitest';
import { kerrSchildParams } from '../src/gr/kerrSchild';
import { probeGridToReadback, READBACK_FLOATS_PER_RAY, ReadbackStatus } from '../src/gr/readback';
import { renderProbeGrid } from '../src/gr/referenceProbe';
import { buildObserverTetrad, staticObserverFourVelocity } from '../src/gr/tetrad';

describe('GPU readback fixture schema', () => {
  it('packs one structured diagnostic row per CPU probe ray', () => {
    const params = kerrSchildParams(0.4, 1);
    const position = { x: 10, y: 0, z: 3 };
    const tetrad = buildObserverTetrad(position, params, staticObserverFourVelocity(position, params));
    const grid = renderProbeGrid(
      params,
      { position, tetrad, verticalFovRadians: 0.75 },
      4,
      3,
      {
        stepSize: 0.04,
        maxSteps: 1600,
        escapeRadius: 30,
        singularityRadius: 0.2,
      },
      { innerRadius: 3, outerRadius: 18 },
      {
        innerRadius: 3,
        outerRadius: 18,
        innerTemperature: 7200,
        emissivityScale: 1,
        boostPower: 4,
      },
    );

    const packed = probeGridToReadback(grid);
    const diskRows = rows(packed).filter((row) => row[0] === ReadbackStatus.Disk);

    expect(packed).toBeInstanceOf(Float32Array);
    expect(packed.length).toBe(grid.rays.length * READBACK_FLOATS_PER_RAY);
    expect(diskRows.length).toBeGreaterThan(0);
    expect(diskRows.every((row) => row[4] >= 3 && row[4] <= 18)).toBe(true);
    expect(diskRows.every((row) => row[5] > 0 && row[6] > 0)).toBe(true);
  });
});

function rows(data: Float32Array): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < data.length; i += READBACK_FLOATS_PER_RAY) {
    out.push([...data.slice(i, i + READBACK_FLOATS_PER_RAY)]);
  }
  return out;
}
