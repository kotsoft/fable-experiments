import {
  hamiltonian,
  nullCovectorFromDirection,
  stepNullGeodesic,
  traceNullGeodesic,
  type GeodesicState,
  type TraceResult,
} from '../gr/geodesic';
import { refineDiskCrossing, type ThinDisk } from '../gr/disk';
import { horizonRadius, kerrSchildNullSpatial, kerrSchildParams, kerrSchildRadius, kerrSchildScalar, type Vec4 } from '../gr/kerrSchild';
import { sampleDiskRadiance, type DiskRadianceModel } from '../gr/radiance';
import { probeGridToReadback, READBACK_FLOATS_PER_RAY, ReadbackStatus, statusToReadback } from '../gr/readback';
import { renderProbeGrid, type ProbeGrid } from '../gr/referenceProbe';
import { buildObserverTetrad, launchPhotonFromTetrad, staticObserverFourVelocity } from '../gr/tetrad';
import {
  COMPOSITE_OUTPUT_FLOATS_PER_RAY,
  compositeOutputRows,
  compositeProbeDetail,
} from '../gr/compositeReadback';
import { runWebGpuComposite, runWebGpuCompositeProbe } from './webgpuCompositeProbe';
import { runWebGpuHamiltonianProbe } from './webgpuHamiltonianProbe';
import { runWebGpuDiskProbe } from './webgpuDiskProbe';
import { runWebGpuMetricProbe } from './webgpuMetricProbe';
import { runWebGpuRadianceProbe } from './webgpuRadianceProbe';
import { runWebGpuEchoReadback } from './webgpuReadback';
import { runWebGpuStepProbe } from './webgpuStepProbe';
import { runWebGpuTraceProbe } from './webgpuTraceProbe';

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

const hamiltonianButton = document.createElement('button');
hamiltonianButton.textContent = 'run WebGPU Hamiltonian probe';
hamiltonianButton.style.cssText =
  'background:#2f3442;color:#e6eaf4;border:1px solid #4a5060;border-radius:6px;padding:8px 10px;' +
  'cursor:pointer;font:600 13px system-ui,sans-serif;margin:8px 0 0 8px;';
panel.appendChild(hamiltonianButton);

const stepButton = document.createElement('button');
stepButton.textContent = 'run WebGPU step probe';
stepButton.style.cssText =
  'background:#2f3442;color:#e6eaf4;border:1px solid #4a5060;border-radius:6px;padding:8px 10px;' +
  'cursor:pointer;font:600 13px system-ui,sans-serif;margin:8px 0 0 0;';
panel.appendChild(stepButton);

const traceButton = document.createElement('button');
traceButton.textContent = 'run WebGPU trace probe';
traceButton.style.cssText =
  'background:#2f3442;color:#e6eaf4;border:1px solid #4a5060;border-radius:6px;padding:8px 10px;' +
  'cursor:pointer;font:600 13px system-ui,sans-serif;margin:8px 0 0 8px;';
panel.appendChild(traceButton);

const diskButton = document.createElement('button');
diskButton.textContent = 'run WebGPU disk probe';
diskButton.style.cssText =
  'background:#2f3442;color:#e6eaf4;border:1px solid #4a5060;border-radius:6px;padding:8px 10px;' +
  'cursor:pointer;font:600 13px system-ui,sans-serif;margin:8px 0 0 0;';
panel.appendChild(diskButton);

const radianceButton = document.createElement('button');
radianceButton.textContent = 'run WebGPU radiance probe';
radianceButton.style.cssText =
  'background:#2f3442;color:#e6eaf4;border:1px solid #4a5060;border-radius:6px;padding:8px 10px;' +
  'cursor:pointer;font:600 13px system-ui,sans-serif;margin:8px 0 0 8px;';
panel.appendChild(radianceButton);

const compositeButton = document.createElement('button');
compositeButton.textContent = 'run WebGPU composite probe';
compositeButton.style.cssText =
  'background:#2f3442;color:#e6eaf4;border:1px solid #4a5060;border-radius:6px;padding:8px 10px;' +
  'cursor:pointer;font:600 13px system-ui,sans-serif;margin:8px 0 0 0;';
panel.appendChild(compositeButton);

const previewButton = document.createElement('button');
previewButton.textContent = 'render WebGPU preview';
previewButton.style.cssText =
  'background:#e8b873;color:#101114;border:0;border-radius:6px;padding:8px 10px;' +
  'cursor:pointer;font:600 13px system-ui,sans-serif;margin:8px 0 0 8px;';
panel.appendChild(previewButton);

const summary = document.createElement('pre');
summary.style.cssText = 'white-space:pre-wrap;margin:14px 0 0;color:#d7dbe5;';
panel.appendChild(summary);

const previewCanvas = document.createElement('canvas');
previewCanvas.width = 64;
previewCanvas.height = 36;
previewCanvas.style.cssText =
  'display:block;width:100%;max-width:960px;aspect-ratio:16/9;background:#000;margin:0 0 14px;' +
  'image-rendering:pixelated;border:1px solid #252936;border-radius:6px;';
output.appendChild(previewCanvas);

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
hamiltonianButton.addEventListener('click', () => {
  void runGpuHamiltonianProbe();
});
stepButton.addEventListener('click', () => {
  void runGpuStepProbe();
});
traceButton.addEventListener('click', () => {
  void runGpuTraceProbe();
});
diskButton.addEventListener('click', () => {
  void runGpuDiskProbe();
});
radianceButton.addEventListener('click', () => {
  void runGpuRadianceProbe();
});
compositeButton.addEventListener('click', () => {
  void runGpuCompositeProbe();
});
previewButton.addEventListener('click', () => {
  void renderGpuPreview();
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
    'gpu metric: not run\n' +
    'gpu Hamiltonian: not run\n' +
    'gpu step: not run\n' +
    'gpu trace: not run\n' +
    'gpu disk: not run\n' +
    'gpu radiance: not run\n' +
    'gpu composite: not run\n' +
    'gpu preview: not run';

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

async function runGpuHamiltonianProbe() {
  const { samples, expected } = createHamiltonianProbeBuffers();
  hamiltonianButton.textContent = 'running WebGPU Hamiltonian probe...';
  hamiltonianButton.setAttribute('disabled', 'true');
  try {
    const result = await runWebGpuHamiltonianProbe(samples, expected);
    const hLine = result.supported
      ? `max diff ${(result.maxAbsDiff ?? Number.NaN).toExponential(3)}`
      : result.message;
    setSummaryLine('gpu Hamiltonian', hLine);
  } catch (error) {
    setSummaryLine('gpu Hamiltonian', `failed (${errorMessage(error)})`);
  } finally {
    hamiltonianButton.textContent = 'run WebGPU Hamiltonian probe';
    hamiltonianButton.removeAttribute('disabled');
  }
}

async function runGpuStepProbe() {
  const { samples, expected } = createStepProbeBuffers();
  stepButton.textContent = 'running WebGPU step probe...';
  stepButton.setAttribute('disabled', 'true');
  try {
    const result = await runWebGpuStepProbe(samples, expected);
    const stepLine = result.supported
      ? `max diff ${(result.maxAbsDiff ?? Number.NaN).toExponential(3)}`
      : result.message;
    setSummaryLine('gpu step', stepLine);
  } catch (error) {
    setSummaryLine('gpu step', `failed (${errorMessage(error)})`);
  } finally {
    stepButton.textContent = 'run WebGPU step probe';
    stepButton.removeAttribute('disabled');
  }
}

async function runGpuTraceProbe() {
  const { samples, expected } = createTraceProbeBuffers();
  traceButton.textContent = 'running WebGPU trace probe...';
  traceButton.setAttribute('disabled', 'true');
  try {
    const result = await runWebGpuTraceProbe(samples, expected);
    const traceLine = result.supported
      ? `max diff ${(result.maxAbsDiff ?? Number.NaN).toExponential(3)}, ` +
        `status mismatches ${result.statusMismatches ?? 0}, step mismatches ${result.stepMismatches ?? 0}`
      : result.message;
    setSummaryLine('gpu trace', traceLine);
  } catch (error) {
    setSummaryLine('gpu trace', `failed (${errorMessage(error)})`);
  } finally {
    traceButton.textContent = 'run WebGPU trace probe';
    traceButton.removeAttribute('disabled');
  }
}

async function runGpuDiskProbe() {
  const { samples, expected } = createDiskProbeBuffers();
  diskButton.textContent = 'running WebGPU disk probe...';
  diskButton.setAttribute('disabled', 'true');
  try {
    const result = await runWebGpuDiskProbe(samples, expected);
    const detail = result.output ? diskProbeDetail(expected, result.output) : '';
    const diskLine = result.supported
      ? `max diff ${(result.maxAbsDiff ?? Number.NaN).toExponential(3)}, ` +
        `hit mismatches ${result.hitMismatches ?? 0}${detail}`
      : result.message;
    setSummaryLine('gpu disk', diskLine);
  } catch (error) {
    setSummaryLine('gpu disk', `failed (${errorMessage(error)})`);
  } finally {
    diskButton.textContent = 'run WebGPU disk probe';
    diskButton.removeAttribute('disabled');
  }
}

async function runGpuRadianceProbe() {
  const { samples, expected } = createRadianceProbeBuffers();
  radianceButton.textContent = 'running WebGPU radiance probe...';
  radianceButton.setAttribute('disabled', 'true');
  try {
    const result = await runWebGpuRadianceProbe(samples, expected);
    const detail = result.output ? radianceProbeDetail(expected, result.output) : '';
    const radianceLine = result.supported
      ? `max diff ${(result.maxAbsDiff ?? Number.NaN).toExponential(3)}, ` +
        `valid mismatches ${result.validMismatches ?? 0}${detail}`
      : result.message;
    setSummaryLine('gpu radiance', radianceLine);
  } catch (error) {
    setSummaryLine('gpu radiance', `failed (${errorMessage(error)})`);
  } finally {
    radianceButton.textContent = 'run WebGPU radiance probe';
    radianceButton.removeAttribute('disabled');
  }
}

async function runGpuCompositeProbe() {
  const { samples, expected } = createCompositeProbeBuffers();
  compositeButton.textContent = 'running WebGPU composite probe...';
  compositeButton.setAttribute('disabled', 'true');
  try {
    const result = await runWebGpuCompositeProbe(samples, expected);
    const detail = result.output ? compositeProbeDetail(expected, result.output) : '';
    const compositeLine = result.supported
      ? `max diff ${(result.maxAbsDiff ?? Number.NaN).toExponential(3)}, ` +
        `status mismatches ${result.statusMismatches ?? 0}, disk mismatches ${result.diskMismatches ?? 0}${detail}`
      : result.message;
    setSummaryLine('gpu composite', compositeLine);
  } catch (error) {
    setSummaryLine('gpu composite', `failed (${errorMessage(error)})`);
  } finally {
    compositeButton.textContent = 'run WebGPU composite probe';
    compositeButton.removeAttribute('disabled');
  }
}

async function renderGpuPreview() {
  const width = 64;
  const height = 36;
  const samples = createCompositeRenderSamples(width, height);
  previewButton.textContent = 'rendering WebGPU preview...';
  previewButton.setAttribute('disabled', 'true');
  try {
    const result = await runWebGpuComposite(samples);
    if (!result.supported || !result.output) {
      setSummaryLine('gpu preview', result.message);
      return;
    }
    drawCompositePreview(previewCanvas, width, height, result.output);
    const rows = compositeOutputRows(result.output);
    const diskHits = rows.filter((row) => row.status === ReadbackStatus.Disk).length;
    const horizons = rows.filter((row) => row.status === ReadbackStatus.Horizon).length;
    const maxDrift = Math.max(...rows.map((row) => row.drift));
    setSummaryLine(
      'gpu preview',
      `${width} x ${height}, disk hits ${diskHits}, horizons ${horizons}, max drift ${maxDrift.toExponential(3)}`,
    );
  } catch (error) {
    setSummaryLine('gpu preview', `failed (${errorMessage(error)})`);
  } finally {
    previewButton.textContent = 'render WebGPU preview';
    previewButton.removeAttribute('disabled');
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

function createHamiltonianProbeBuffers(): { samples: Float32Array; expected: Float32Array } {
  const rows = [
    { position: { x: 8, y: 0, z: 1 }, spin: 0, direction: { x: -0.4, y: 0.1, z: 1 } },
    { position: { x: 7, y: 2, z: -1.5 }, spin: 0.45, direction: { x: 0.2, y: -0.1, z: 1 } },
    { position: { x: -4, y: 6, z: 2.25 }, spin: 0.75, direction: { x: -0.2, y: 0.3, z: 1 } },
    { position: { x: 3.5, y: -2.5, z: 4 }, spin: 0.9, direction: { x: 0.35, y: 0.05, z: 1 } },
  ];
  const samples = new Float32Array(rows.length * 8);
  const expected = new Float32Array(rows.length * 4);
  rows.forEach((row, index) => {
    const params = kerrSchildParams(row.spin, 1);
    const position = { t: 0, ...row.position };
    const momentum = nullCovectorFromDirection(position, row.direction, params);
    const l = kerrSchildNullSpatial(row.position, params);
    const radius = kerrSchildRadius(row.position, params);
    const scalar = kerrSchildScalar(row.position, params);
    const sampleBase = index * 8;
    const expectedBase = index * 4;
    samples[sampleBase] = row.position.x;
    samples[sampleBase + 1] = row.position.y;
    samples[sampleBase + 2] = row.position.z;
    samples[sampleBase + 3] = row.spin;
    samples[sampleBase + 4] = momentum.t;
    samples[sampleBase + 5] = momentum.x;
    samples[sampleBase + 6] = momentum.y;
    samples[sampleBase + 7] = momentum.z;
    expected[expectedBase] = hamiltonian({ position, momentum }, params);
    expected[expectedBase + 1] = radius;
    expected[expectedBase + 2] = scalar;
    expected[expectedBase + 3] = l.x * l.x + l.y * l.y + l.z * l.z;
  });
  return { samples, expected };
}

function createStepProbeBuffers(): { samples: Float32Array; expected: Float32Array } {
  const rows = [
    { position: { x: 8, y: 0, z: 1 }, spin: 0, direction: { x: -0.25, y: 0.05, z: 1 }, step: 0.015 },
    { position: { x: 7, y: 2, z: -1.5 }, spin: 0.45, direction: { x: 0.18, y: -0.1, z: 1 }, step: 0.012 },
    { position: { x: -4, y: 6, z: 2.25 }, spin: 0.75, direction: { x: -0.15, y: 0.22, z: 1 }, step: 0.01 },
  ];
  const samples = new Float32Array(rows.length * 12);
  const expected = new Float32Array(rows.length * 8);
  rows.forEach((row, index) => {
    const params = kerrSchildParams(row.spin, 1);
    const state: GeodesicState = {
      position: { t: 0, ...row.position },
      momentum: nullCovectorFromDirection({ t: 0, ...row.position }, row.direction, params),
    };
    const next = stepNullGeodesic(state, params, row.step);
    const sampleBase = index * 12;
    const expectedBase = index * 8;
    samples[sampleBase] = state.position.t;
    samples[sampleBase + 1] = state.position.x;
    samples[sampleBase + 2] = state.position.y;
    samples[sampleBase + 3] = state.position.z;
    samples[sampleBase + 4] = state.momentum.t;
    samples[sampleBase + 5] = state.momentum.x;
    samples[sampleBase + 6] = state.momentum.y;
    samples[sampleBase + 7] = state.momentum.z;
    samples[sampleBase + 8] = row.spin;
    samples[sampleBase + 9] = row.step;
    samples[sampleBase + 10] = 0;
    samples[sampleBase + 11] = 0;
    expected[expectedBase] = next.position.t;
    expected[expectedBase + 1] = next.position.x;
    expected[expectedBase + 2] = next.position.y;
    expected[expectedBase + 3] = next.position.z;
    expected[expectedBase + 4] = next.momentum.t;
    expected[expectedBase + 5] = next.momentum.x;
    expected[expectedBase + 6] = next.momentum.y;
    expected[expectedBase + 7] = next.momentum.z;
  });
  return { samples, expected };
}

function createTraceProbeBuffers(): { samples: Float32Array; expected: Float32Array } {
  const rows = [
    {
      position: { x: 3.4, y: 0, z: 0.4 },
      spin: 0,
      direction: { x: -1, y: 0.02, z: -0.05 },
      options: { stepSize: 0.025, maxSteps: 96, escapeRadius: 18, singularityRadius: 0.2 },
    },
    {
      position: { x: 8, y: 0.5, z: 1.2 },
      spin: 0.45,
      direction: { x: -0.15, y: 0.16, z: 1 },
      options: { stepSize: 0.018, maxSteps: 48, escapeRadius: 18, singularityRadius: 0.2 },
    },
    {
      position: { x: 5.5, y: -2.2, z: 0.8 },
      spin: 0.7,
      direction: { x: 0.8, y: -0.25, z: 0.15 },
      options: { stepSize: 0.02, maxSteps: 120, escapeRadius: 6, singularityRadius: 0.2 },
    },
  ];
  const samples = new Float32Array(rows.length * 16);
  const expected = new Float32Array(rows.length * 12);
  rows.forEach((row, index) => {
    const params = kerrSchildParams(row.spin, 1);
    const state: GeodesicState = {
      position: { t: 0, ...row.position },
      momentum: nullCovectorFromDirection({ t: 0, ...row.position }, row.direction, params),
    };
    const result = traceNullGeodesic(state, params, row.options);
    const finalRadius = kerrSchildRadius(position3(result.state.position), params);
    const sampleBase = index * 16;
    const expectedBase = index * 12;
    samples[sampleBase] = state.position.t;
    samples[sampleBase + 1] = state.position.x;
    samples[sampleBase + 2] = state.position.y;
    samples[sampleBase + 3] = state.position.z;
    samples[sampleBase + 4] = state.momentum.t;
    samples[sampleBase + 5] = state.momentum.x;
    samples[sampleBase + 6] = state.momentum.y;
    samples[sampleBase + 7] = state.momentum.z;
    samples[sampleBase + 8] = row.spin;
    samples[sampleBase + 9] = row.options.stepSize;
    samples[sampleBase + 10] = row.options.escapeRadius;
    samples[sampleBase + 11] = row.options.singularityRadius;
    samples[sampleBase + 12] = row.options.maxSteps;
    samples[sampleBase + 13] = params.mass;
    samples[sampleBase + 14] = 0;
    samples[sampleBase + 15] = 0;
    expected[expectedBase] = traceStatusToReadback(result.status);
    expected[expectedBase + 1] = result.steps;
    expected[expectedBase + 2] = finalRadius;
    expected[expectedBase + 3] = result.maxHamiltonianDrift;
    expected[expectedBase + 4] = result.state.position.t;
    expected[expectedBase + 5] = result.state.position.x;
    expected[expectedBase + 6] = result.state.position.y;
    expected[expectedBase + 7] = result.state.position.z;
    expected[expectedBase + 8] = result.state.momentum.t;
    expected[expectedBase + 9] = result.state.momentum.x;
    expected[expectedBase + 10] = result.state.momentum.y;
    expected[expectedBase + 11] = result.state.momentum.z;
  });
  return { samples, expected };
}

function createDiskProbeBuffers(): { samples: Float32Array; expected: Float32Array } {
  const rows: Array<{
    position: { x: number; y: number; z: number };
    mass: number;
    spin: number;
    direction: { x: number; y: number; z: number };
    stepSize: number;
    disk: ThinDisk;
  }> = [
    {
      position: { x: 4, y: 0, z: 1 },
      mass: 0,
      spin: 0,
      direction: { x: 0.5, y: 0, z: -1 },
      stepSize: 4,
      disk: { innerRadius: 3, outerRadius: 8 },
    },
    {
      position: { x: 1, y: 0, z: 1 },
      mass: 0,
      spin: 0,
      direction: { x: 0, y: 0, z: -1 },
      stepSize: 3,
      disk: { innerRadius: 3, outerRadius: 8 },
    },
    {
      position: { x: 6, y: 0, z: 0.7 },
      mass: 1,
      spin: 0.55,
      direction: { x: 0, y: 0.03, z: -1 },
      stepSize: 1.5,
      disk: { innerRadius: 3, outerRadius: 12 },
    },
  ];
  const samples = new Float32Array(rows.length * 16);
  const expected = new Float32Array(rows.length * 4);
  rows.forEach((row, index) => {
    const params = kerrSchildParams(row.spin, row.mass);
    const state: GeodesicState = {
      position: { t: 0, ...row.position },
      momentum: nullCovectorFromDirection({ t: 0, ...row.position }, row.direction, params),
    };
    const crossing = refineDiskCrossing(state, params, row.stepSize, row.disk, 24);
    const sampleBase = index * 16;
    const expectedBase = index * 4;
    samples[sampleBase] = state.position.t;
    samples[sampleBase + 1] = state.position.x;
    samples[sampleBase + 2] = state.position.y;
    samples[sampleBase + 3] = state.position.z;
    samples[sampleBase + 4] = state.momentum.t;
    samples[sampleBase + 5] = state.momentum.x;
    samples[sampleBase + 6] = state.momentum.y;
    samples[sampleBase + 7] = state.momentum.z;
    samples[sampleBase + 8] = row.spin;
    samples[sampleBase + 9] = row.stepSize;
    samples[sampleBase + 10] = row.disk.innerRadius;
    samples[sampleBase + 11] = row.disk.outerRadius;
    samples[sampleBase + 12] = params.mass;
    samples[sampleBase + 13] = 0;
    samples[sampleBase + 14] = 0;
    samples[sampleBase + 15] = 0;
    expected[expectedBase] = crossing ? 1 : 0;
    expected[expectedBase + 1] = crossing?.affineParameter ?? -1;
    expected[expectedBase + 2] = crossing?.radius ?? -1;
    expected[expectedBase + 3] = crossing?.height ?? 0;
  });
  return { samples, expected };
}

function createRadianceProbeBuffers(): { samples: Float32Array; expected: Float32Array } {
  const model: DiskRadianceModel = {
    innerRadius: 3,
    outerRadius: 18,
    innerTemperature: 7200,
    emissivityScale: 1.25,
    boostPower: 4,
  };
  const rows: Array<{
    position: { x: number; y: number; z: number };
    spin: number;
    direction: { x: number; y: number; z: number };
    observerVelocity?: Vec4;
    model: DiskRadianceModel;
  }> = [
    {
      position: { x: 4, y: 0, z: 0 },
      spin: 0.5,
      direction: { x: 0, y: -1, z: 0 },
      model,
    },
    {
      position: { x: 12, y: 0, z: 0 },
      spin: 0.5,
      direction: { x: 0, y: -1, z: 0 },
      model: { ...model, boostPower: 2 },
    },
    {
      position: { x: 2, y: 0, z: 0 },
      spin: 0,
      direction: { x: 0, y: -1, z: 0 },
      model,
    },
    {
      position: { x: 8, y: 0, z: 0 },
      spin: 0.55,
      direction: { x: 0, y: -1, z: 0 },
      observerVelocity: staticObserverFourVelocity({ x: 10, y: 0, z: 3 }, kerrSchildParams(0.55, 1)),
      model: { ...model, spinDirection: -1 },
    },
  ];
  const samples = new Float32Array(rows.length * 20);
  const expected = new Float32Array(rows.length * 8);
  rows.forEach((row, index) => {
    const params = kerrSchildParams(row.spin, 1);
    const position = { t: 0, ...row.position };
    const state: GeodesicState = {
      position,
      momentum: nullCovectorFromDirection(position, row.direction, params),
    };
    const radiance = sampleDiskRadiance(state, params, row.model, row.observerVelocity);
    const sampleBase = index * 20;
    const expectedBase = index * 8;
    samples[sampleBase] = state.position.t;
    samples[sampleBase + 1] = state.position.x;
    samples[sampleBase + 2] = state.position.y;
    samples[sampleBase + 3] = state.position.z;
    samples[sampleBase + 4] = state.momentum.t;
    samples[sampleBase + 5] = state.momentum.x;
    samples[sampleBase + 6] = state.momentum.y;
    samples[sampleBase + 7] = state.momentum.z;
    samples[sampleBase + 8] = row.observerVelocity?.t ?? 0;
    samples[sampleBase + 9] = row.observerVelocity?.x ?? 0;
    samples[sampleBase + 10] = row.observerVelocity?.y ?? 0;
    samples[sampleBase + 11] = row.observerVelocity?.z ?? 0;
    samples[sampleBase + 12] = params.spin;
    samples[sampleBase + 13] = params.mass;
    samples[sampleBase + 14] = row.model.innerRadius;
    samples[sampleBase + 15] = row.model.outerRadius;
    samples[sampleBase + 16] = row.model.innerTemperature;
    samples[sampleBase + 17] = row.model.emissivityScale;
    samples[sampleBase + 18] = row.model.boostPower;
    samples[sampleBase + 19] = row.model.spinDirection ?? 1;
    expected[expectedBase] = radiance ? 1 : 0;
    expected[expectedBase + 1] = radiance?.radius ?? kerrSchildRadius(row.position, params);
    expected[expectedBase + 2] = radiance?.temperature ?? 0;
    expected[expectedBase + 3] = radiance?.redshift ?? 0;
    expected[expectedBase + 4] = radiance?.bolometricIntensity ?? 0;
    expected[expectedBase + 5] = radiance?.observedRgb[0] ?? 0;
    expected[expectedBase + 6] = radiance?.observedRgb[1] ?? 0;
    expected[expectedBase + 7] = radiance?.observedRgb[2] ?? 0;
  });
  return { samples, expected };
}

function createCompositeProbeBuffers(): { samples: Float32Array; expected: Float32Array } {
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
  const samples = new Float32Array(grid.rays.length * 28);
  const expected = new Float32Array(grid.rays.length * 8);
  grid.rays.forEach((ray, index) => {
    const momentum = launchPhotonFromTetrad(position, params, tetrad, ray.localDirection);
    const sampleBase = index * 28;
    const expectedBase = index * 8;
    samples[sampleBase] = 0;
    samples[sampleBase + 1] = position.x;
    samples[sampleBase + 2] = position.y;
    samples[sampleBase + 3] = position.z;
    samples[sampleBase + 4] = momentum.t;
    samples[sampleBase + 5] = momentum.x;
    samples[sampleBase + 6] = momentum.y;
    samples[sampleBase + 7] = momentum.z;
    samples[sampleBase + 8] = observerVelocity.t;
    samples[sampleBase + 9] = observerVelocity.x;
    samples[sampleBase + 10] = observerVelocity.y;
    samples[sampleBase + 11] = observerVelocity.z;
    samples[sampleBase + 12] = params.spin;
    samples[sampleBase + 13] = traceOptions.stepSize;
    samples[sampleBase + 14] = traceOptions.escapeRadius;
    samples[sampleBase + 15] = traceOptions.singularityRadius;
    samples[sampleBase + 16] = traceOptions.maxSteps;
    samples[sampleBase + 17] = params.mass;
    samples[sampleBase + 18] = 0;
    samples[sampleBase + 19] = 0;
    samples[sampleBase + 20] = disk.innerRadius;
    samples[sampleBase + 21] = disk.outerRadius;
    samples[sampleBase + 22] = radianceModel.innerTemperature;
    samples[sampleBase + 23] = radianceModel.emissivityScale;
    samples[sampleBase + 24] = radianceModel.boostPower;
    samples[sampleBase + 25] = 1;
    samples[sampleBase + 26] = 0;
    samples[sampleBase + 27] = 0;
    expected[expectedBase] = statusToReadback(ray.status);
    expected[expectedBase + 1] = ray.steps;
    expected[expectedBase + 2] = ray.finalRadius;
    expected[expectedBase + 3] = ray.diskHit?.radius ?? -1;
    expected[expectedBase + 4] = ray.color[0];
    expected[expectedBase + 5] = ray.color[1];
    expected[expectedBase + 6] = ray.color[2];
    expected[expectedBase + 7] = ray.maxHamiltonianDrift;
  });
  return { samples, expected };
}

function createCompositeRenderSamples(width: number, height: number): Float32Array {
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
  const aspect = width / height;
  const tanHalfFov = Math.tan(0.82 * 0.5);
  const samples = new Float32Array(width * height * 28);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      const ndcX = (2 * (x + 0.5) / width - 1) * aspect;
      const ndcY = 1 - 2 * (y + 0.5) / height;
      const localDirection = normalize3({ x: ndcX * tanHalfFov, y: ndcY * tanHalfFov, z: 1 });
      const momentum = launchPhotonFromTetrad(position, params, tetrad, localDirection);
      const base = index * 28;
      samples[base] = 0;
      samples[base + 1] = position.x;
      samples[base + 2] = position.y;
      samples[base + 3] = position.z;
      samples[base + 4] = momentum.t;
      samples[base + 5] = momentum.x;
      samples[base + 6] = momentum.y;
      samples[base + 7] = momentum.z;
      samples[base + 8] = observerVelocity.t;
      samples[base + 9] = observerVelocity.x;
      samples[base + 10] = observerVelocity.y;
      samples[base + 11] = observerVelocity.z;
      samples[base + 12] = params.spin;
      samples[base + 13] = traceOptions.stepSize;
      samples[base + 14] = traceOptions.escapeRadius;
      samples[base + 15] = traceOptions.singularityRadius;
      samples[base + 16] = traceOptions.maxSteps;
      samples[base + 17] = params.mass;
      samples[base + 18] = 0;
      samples[base + 19] = 0;
      samples[base + 20] = disk.innerRadius;
      samples[base + 21] = disk.outerRadius;
      samples[base + 22] = radianceModel.innerTemperature;
      samples[base + 23] = radianceModel.emissivityScale;
      samples[base + 24] = radianceModel.boostPower;
      samples[base + 25] = 1;
      samples[base + 26] = 0;
      samples[base + 27] = 0;
    }
  }
  return samples;
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

function traceStatusToReadback(statusValue: TraceResult['status']): number {
  if (statusValue === 'escaped') return ReadbackStatus.Escaped;
  if (statusValue === 'horizon') return ReadbackStatus.Horizon;
  if (statusValue === 'singularity') return ReadbackStatus.Singularity;
  return ReadbackStatus.MaxSteps;
}

function position3(v: GeodesicState['position']) {
  return { x: v.x, y: v.y, z: v.z };
}

function diskProbeDetail(expected: Float32Array<ArrayBufferLike>, output: Float32Array<ArrayBufferLike>): string {
  for (let i = 0; i < expected.length; i += 4) {
    if (Math.round(expected[i]) !== Math.round(output[i])) {
      return `, row ${i / 4} expected [${formatVec4(expected, i)}] got [${formatVec4(output, i)}]`;
    }
  }
  let max = 0;
  let offset = 0;
  for (let i = 0; i < expected.length; i++) {
    const diff = Math.abs(expected[i] - output[i]);
    if (diff > max) {
      max = diff;
      offset = i - i % 4;
    }
  }
  return max > 1e-3 ? `, max row ${offset / 4} expected [${formatVec4(expected, offset)}] got [${formatVec4(output, offset)}]` : '';
}

function formatVec4(values: Float32Array<ArrayBufferLike>, offset: number): string {
  return [
    values[offset].toFixed(3),
    values[offset + 1].toFixed(3),
    values[offset + 2].toFixed(3),
    values[offset + 3].toExponential(1),
  ].join(', ');
}

function radianceProbeDetail(expected: Float32Array<ArrayBufferLike>, output: Float32Array<ArrayBufferLike>): string {
  for (let i = 0; i < expected.length; i += 8) {
    if (Math.round(expected[i]) !== Math.round(output[i])) {
      return `, row ${i / 8} expected [${formatVecN(expected, i, 8)}] got [${formatVecN(output, i, 8)}]`;
    }
  }
  let max = 0;
  let offset = 0;
  for (let i = 0; i < expected.length; i++) {
    const diff = Math.abs(expected[i] - output[i]);
    if (diff > max) {
      max = diff;
      offset = i - i % 8;
    }
  }
  return max > 1e-2 ? `, max row ${offset / 8} expected [${formatVecN(expected, offset, 8)}] got [${formatVecN(output, offset, 8)}]` : '';
}

function formatVecN(values: Float32Array<ArrayBufferLike>, offset: number, length: number): string {
  return Array.from({ length }, (_, i) => values[offset + i].toPrecision(4)).join(', ');
}

function drawCompositePreview(
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
  output: Float32Array<ArrayBufferLike>,
) {
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) return;
  const image = context.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const row = i * COMPOSITE_OUTPUT_FLOATS_PER_RAY;
    const pixel = i * 4;
    image.data[pixel] = toDisplayByte(output[row + 4]);
    image.data[pixel + 1] = toDisplayByte(output[row + 5]);
    image.data[pixel + 2] = toDisplayByte(output[row + 6]);
    image.data[pixel + 3] = 255;
  }
  context.putImageData(image, 0, 0);
}

function toDisplayByte(value: number): number {
  const mapped = Math.max(0, value) / (1 + Math.max(0, value));
  return Math.round(Math.pow(mapped, 1 / 2.2) * 255);
}

function normalize3(v: { x: number; y: number; z: number }) {
  const length = Math.hypot(v.x, v.y, v.z) || 1;
  return { x: v.x / length, y: v.y / length, z: v.z / length };
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
