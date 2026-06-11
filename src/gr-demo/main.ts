import { horizonRadius, kerrSchildNullSpatial, kerrSchildParams, kerrSchildRadius, kerrSchildScalar } from '../gr/kerrSchild';
import { probeGridToReadback, READBACK_FLOATS_PER_RAY, ReadbackStatus } from '../gr/readback';
import { renderProbeGrid, type ProbeGrid } from '../gr/referenceProbe';
import { buildObserverTetrad, staticObserverFourVelocity } from '../gr/tetrad';
import { runWebGpuMetricProbe } from './webgpuMetricProbe';
import { runWebGpuEchoReadback } from './webgpuReadback';

type GpuNavigator = Navigator & {
  gpu?: {
    requestAdapter(): Promise<{
      info?: { description?: string; vendor?: string; architecture?: string; device?: string };
      requestDevice(): Promise<unknown>;
    } | null>;
  };
};

const root = document.createElement('main');
root.style.cssText =
  'min-height:100vh;background:#08090d;color:#d7dbe5;font:14px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;' +
  'display:grid;grid-template-columns:minmax(280px,380px) 1fr;gap:18px;padding:18px;box-sizing:border-box;';
document.body.style.margin = '0';
document.body.appendChild(root);

const panel = section();
const output = section();
output.style.overflow = 'auto';
root.append(panel, output);

const title = document.createElement('h1');
title.textContent = 'Kerr-Schild GRRT diagnostics';
title.style.cssText = 'font:600 18px system-ui,sans-serif;margin:0 0 10px;color:#fff;';
panel.appendChild(title);

const status = document.createElement('div');
status.style.cssText = 'white-space:pre-wrap;color:#aeb5c5;margin-bottom:12px;';
panel.appendChild(status);

const runButton = document.createElement('button');
runButton.textContent = 'run CPU reference probe';
runButton.style.cssText =
  'background:#e8b873;color:#101114;border:0;border-radius:6px;padding:8px 10px;cursor:pointer;font:600 13px system-ui,sans-serif;';
panel.appendChild(runButton);

const gpuButton = document.createElement('button');
gpuButton.textContent = 'run WebGPU readback echo';
gpuButton.style.cssText =
  'background:#2f3442;color:#e6eaf4;border:1px solid #4a5060;border-radius:6px;padding:8px 10px;' +
  'cursor:pointer;font:600 13px system-ui,sans-serif;margin-left:8px;';
panel.appendChild(gpuButton);

const metricButton = document.createElement('button');
metricButton.textContent = 'run WebGPU metric probe';
metricButton.style.cssText =
  'background:#2f3442;color:#e6eaf4;border:1px solid #4a5060;border-radius:6px;padding:8px 10px;' +
  'cursor:pointer;font:600 13px system-ui,sans-serif;margin:8px 0 0 0;';
panel.appendChild(metricButton);

const summary = document.createElement('pre');
summary.style.cssText = 'white-space:pre-wrap;margin:14px 0 0;color:#d7dbe5;';
panel.appendChild(summary);

const table = document.createElement('pre');
table.style.cssText = 'white-space:pre;tab-size:2;margin:0;color:#cbd1df;';
output.appendChild(table);

let latestReadback: Float32Array<ArrayBufferLike> = new Float32Array();

void detectWebGpu();
renderCpuProbe();
runButton.addEventListener('click', renderCpuProbe);
gpuButton.addEventListener('click', () => {
  void runGpuEcho();
});
metricButton.addEventListener('click', () => {
  void runGpuMetricProbe();
});

async function detectWebGpu() {
  const gpu = (navigator as GpuNavigator).gpu;
  if (!gpu) {
    status.textContent = 'WebGPU: unavailable in this browser\nCPU reference: ready';
    return;
  }
  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    status.textContent = 'WebGPU: no adapter\nCPU reference: ready';
    return;
  }
  status.textContent =
    'WebGPU: available\n' +
    `adapter: ${adapter.info?.description || adapter.info?.vendor || 'unknown'}\n` +
    'CPU reference: ready';
}

function renderCpuProbe() {
  const grid = createReferenceGrid();
  const readback = probeGridToReadback(grid);
  latestReadback = readback;
  const rows = readbackRows(readback);
  const diskHits = rows.filter((row) => row.status === ReadbackStatus.Disk);
  const maxDrift = Math.max(...rows.map((row) => row.maxHamiltonianDrift));

  summary.textContent =
    `grid: ${grid.width} x ${grid.height}\n` +
    `rays: ${grid.rays.length}\n` +
    `disk hits: ${diskHits.length}\n` +
    `max |H drift|: ${maxDrift.toExponential(3)}\n` +
    `floats/ray: ${READBACK_FLOATS_PER_RAY}\n` +
    'gpu echo: not run\n' +
    'gpu metric: not run';

  table.textContent = [
    'px py status steps radius drift diskR redshift intensity rgb',
    ...rows.map((row) =>
      [
        row.pixelX,
        row.pixelY,
        statusName(row.status),
        row.steps,
        row.finalRadius.toFixed(3),
        row.maxHamiltonianDrift.toExponential(1),
        row.diskRadius.toFixed(3),
        row.redshift.toFixed(3),
        row.bolometricIntensity.toExponential(2),
        `${row.color[0].toFixed(2)},${row.color[1].toFixed(2)},${row.color[2].toFixed(2)}`,
      ].join('\t'),
    ),
  ].join('\n');
}

async function runGpuEcho() {
  if (latestReadback.length === 0) renderCpuProbe();
  gpuButton.textContent = 'running WebGPU echo...';
  gpuButton.setAttribute('disabled', 'true');
  try {
    const result = await runWebGpuEchoReadback(latestReadback);
    const gpuLine = result.supported
      ? `max diff ${(result.maxAbsDiff ?? Number.NaN).toExponential(3)}`
      : result.message;
    setGpuEchoLine(gpuLine);
  } catch (error) {
    setGpuEchoLine(`failed (${errorMessage(error)})`);
  } finally {
    gpuButton.textContent = 'run WebGPU readback echo';
    gpuButton.removeAttribute('disabled');
  }
}

async function runGpuMetricProbe() {
  const { samples, expected } = createMetricProbeBuffers();
  metricButton.textContent = 'running WebGPU metric probe...';
  metricButton.setAttribute('disabled', 'true');
  try {
    const result = await runWebGpuMetricProbe(samples, expected);
    const metricLine = result.supported
      ? `max diff ${(result.maxAbsDiff ?? Number.NaN).toExponential(3)}`
      : result.message;
    setSummaryLine('gpu metric', metricLine);
  } catch (error) {
    setSummaryLine('gpu metric', `failed (${errorMessage(error)})`);
  } finally {
    metricButton.textContent = 'run WebGPU metric probe';
    metricButton.removeAttribute('disabled');
  }
}

function createReferenceGrid(): ProbeGrid {
  const params = kerrSchildParams(0.55, 1);
  const position = { x: 10, y: 0, z: 3 };
  const tetrad = buildObserverTetrad(position, params, staticObserverFourVelocity(position, params));
  return renderProbeGrid(
    params,
    { position, tetrad, verticalFovRadians: 0.82 },
    8,
    5,
    {
      stepSize: 0.04,
      maxSteps: 1800,
      escapeRadius: 32,
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
}

function createMetricProbeBuffers(): { samples: Float32Array; expected: Float32Array } {
  const rows = [
    { position: { x: 8, y: 0, z: 1 }, spin: 0 },
    { position: { x: 7, y: 2, z: -1.5 }, spin: 0.45 },
    { position: { x: -4, y: 6, z: 2.25 }, spin: 0.75 },
    { position: { x: 3.5, y: -2.5, z: 4 }, spin: 0.9 },
  ];
  const samples = new Float32Array(rows.length * 4);
  const expected = new Float32Array(rows.length * 4);
  rows.forEach((row, index) => {
    const params = kerrSchildParams(row.spin, 1);
    const l = kerrSchildNullSpatial(row.position, params);
    const base = index * 4;
    samples[base] = row.position.x;
    samples[base + 1] = row.position.y;
    samples[base + 2] = row.position.z;
    samples[base + 3] = row.spin;
    expected[base] = kerrSchildRadius(row.position, params);
    expected[base + 1] = kerrSchildScalar(row.position, params);
    expected[base + 2] = l.x * l.x + l.y * l.y + l.z * l.z;
    expected[base + 3] = horizonRadius(params);
  });
  return { samples, expected };
}

function readbackRows(readback: Float32Array) {
  const rows = [];
  for (let i = 0; i < readback.length; i += READBACK_FLOATS_PER_RAY) {
    rows.push({
      status: readback[i],
      steps: readback[i + 1],
      finalRadius: readback[i + 2],
      maxHamiltonianDrift: readback[i + 3],
      diskRadius: readback[i + 4],
      redshift: readback[i + 5],
      bolometricIntensity: readback[i + 6],
      color: [readback[i + 7], readback[i + 8], readback[i + 9]],
      pixelX: readback[i + 13],
      pixelY: readback[i + 14],
    });
  }
  return rows;
}

function statusName(statusValue: number): string {
  if (statusValue === ReadbackStatus.Escaped) return 'escape';
  if (statusValue === ReadbackStatus.Horizon) return 'horizon';
  if (statusValue === ReadbackStatus.Singularity) return 'sing';
  if (statusValue === ReadbackStatus.MaxSteps) return 'max';
  if (statusValue === ReadbackStatus.Disk) return 'disk';
  return 'unknown';
}

function section(): HTMLElement {
  const el = document.createElement('section');
  el.style.cssText = 'background:#101218;border:1px solid #252936;border-radius:8px;padding:14px;';
  return el;
}

function setGpuEchoLine(text: string) {
  setSummaryLine('gpu echo', text);
}

function setSummaryLine(label: string, text: string) {
  const current = summary.textContent ?? '';
  const line = `${label}: ${text}`;
  const pattern = new RegExp(`\\n${label}: .*`);
  summary.textContent = pattern.test(current) ? current.replace(pattern, `\n${line}`) : `${current}\n${line}`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
