import { buildTetrad, eulerianObserver, normalize3, spatial, type Tetrad, type Vec3 } from './kerr';
import { createPlanner } from './minimap';
import {
  HORIZON,
  INNER_HORIZON,
  PARAMS,
  PRESETS,
  SINGULARITY_CUTOFF,
  constraintResidual,
  fourVelocity,
  launchLocal,
  localSpeed,
  previewPath,
  stepPlayer,
  type PlayerState,
  type PreviewPoint,
} from './player';
import { FallfableRenderer, type RendererFrameTiming } from './renderer';

// ----------------------------------------------------------------- layout

document.body.classList.add('fallfable-page');
const layoutStyle = document.createElement('style');
layoutStyle.textContent = `
  .fallfable-page {
    margin: 0;
    background: #000;
    overflow: hidden;
    overscroll-behavior: none;
    touch-action: none;
  }
  .fallfable-canvas {
    position: fixed;
    inset: 0;
    width: 100vw;
    height: 100dvh;
    display: block;
    background: #000;
    cursor: grab;
    touch-action: none;
    user-select: none;
  }
  .fallfable-panel {
    position: fixed;
    right: max(14px, env(safe-area-inset-right));
    top: max(14px, env(safe-area-inset-top));
    display: grid;
    gap: 6px;
    justify-items: end;
    color: #c9cdd6;
    font: 12px ui-monospace, SFMono-Regular, Consolas, monospace;
    user-select: none;
    z-index: 2;
  }
  .fallfable-glass {
    background: #080910cc;
    border: 1px solid #23262e;
    border-radius: 6px;
    backdrop-filter: blur(6px);
  }
  .fallfable-summary {
    justify-self: end;
    display: flex;
    gap: 8px;
    align-items: center;
    background: #080910b8;
    border: 1px solid #23262e;
    border-radius: 6px;
    padding: 5px 11px;
    color: #c9cdd6;
    cursor: pointer;
    font: inherit;
    backdrop-filter: blur(6px);
  }
  .fallfable-details {
    display: none;
    grid-template-rows: auto auto;
    gap: 6px;
    justify-items: stretch;
  }
  .fallfable-readout {
    padding: 9px 11px;
    line-height: 1.6;
    min-width: 300px;
  }
  .fallfable-controls {
    padding: 9px 11px;
    display: grid;
    gap: 8px;
  }
  .fallfable-presets {
    position: fixed;
    left: max(14px, env(safe-area-inset-left));
    right: max(329px, calc(315px + env(safe-area-inset-right)));
    bottom: max(14px, env(safe-area-inset-bottom));
    transform: none;
    display: flex;
    gap: 6px;
    flex-wrap: nowrap;
    justify-content: flex-start;
    max-width: none;
    overflow-x: auto;
    scrollbar-width: none;
    -webkit-overflow-scrolling: touch;
    z-index: 2;
    font: 11px ui-monospace, SFMono-Regular, Consolas, monospace;
    user-select: none;
  }
  .fallfable-presets::-webkit-scrollbar {
    display: none;
  }
  .fallfable-preset {
    background: #080910b8;
    color: #c9cdd6;
    border: 1px solid #23262e;
    border-radius: 999px;
    padding: 5px 11px;
    font: inherit;
    cursor: pointer;
    backdrop-filter: blur(6px);
    white-space: nowrap;
  }
  .fallfable-row {
    display: flex;
    gap: 8px;
    align-items: center;
    justify-content: space-between;
  }
  .fallfable-select,
  .fallfable-button {
    background: #171a22;
    color: #e6eaf4;
    border: 1px solid #3b4150;
    border-radius: 5px;
    padding: 4px 6px;
    font: 12px ui-monospace, monospace;
  }
  .fallfable-button {
    cursor: pointer;
  }
  .fallfable-map {
    position: fixed;
    right: max(14px, env(safe-area-inset-right));
    bottom: max(14px, env(safe-area-inset-bottom));
    background: #08091040;
    border-color: #3b415088;
    border-radius: 50%;
    overflow: hidden;
    box-shadow: 0 10px 30px #0008, inset 0 0 0 1px #ffffff0f;
  }
  .fallfable-map canvas {
    border-radius: 50%;
  }

  @media (hover: none) and (pointer: coarse) {
    .fallfable-summary,
    .fallfable-button,
    .fallfable-select,
    .fallfable-preset {
      min-height: 34px;
    }
  }

  @media (max-width: 720px), (max-height: 560px) {
    .fallfable-canvas {
      height: 100dvh !important;
    }
    .fallfable-presets {
      top: max(8px, env(safe-area-inset-top));
      bottom: auto !important;
      left: 0 !important;
      right: 0 !important;
      transform: none !important;
      max-width: none !important;
      justify-content: flex-start !important;
      flex-wrap: nowrap !important;
      overflow-x: auto;
      padding: 0 10px 8px;
      scrollbar-width: none;
      -webkit-overflow-scrolling: touch;
    }
    .fallfable-presets::-webkit-scrollbar {
      display: none;
    }
    .fallfable-preset {
      padding: 8px 12px !important;
      flex: 0 0 auto;
    }
    .fallfable-panel {
      top: auto !important;
      left: max(10px, env(safe-area-inset-left)) !important;
      right: max(10px, env(safe-area-inset-right)) !important;
      bottom: max(10px, env(safe-area-inset-bottom)) !important;
      justify-items: stretch !important;
      gap: 7px !important;
      max-height: calc(100dvh - 64px);
    }
    .fallfable-summary {
      order: 2;
      width: 100% !important;
      justify-content: flex-start !important;
      padding: 9px 12px !important;
      box-sizing: border-box;
    }
    .fallfable-summary-text {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .fallfable-details {
      order: 1;
      grid-template-rows: auto auto auto !important;
      max-height: min(calc(100dvh - 118px), 560px);
      overflow: auto;
      padding-right: 2px;
      -webkit-overflow-scrolling: touch;
    }
    .fallfable-readout,
    .fallfable-controls {
      min-width: 0 !important;
      width: 100% !important;
      box-sizing: border-box;
      padding: 10px 12px !important;
    }
    .fallfable-readout {
      font-size: 11px;
    }
    .fallfable-row {
      display: grid !important;
      grid-template-columns: minmax(82px, auto) minmax(0, 1fr);
      gap: 8px !important;
    }
    .fallfable-select,
    .fallfable-button {
      font-size: 12px !important;
      padding: 7px 8px !important;
    }
    .fallfable-select {
      width: 100%;
      min-width: 0;
    }
    .fallfable-controls input[type="range"] {
      width: 100% !important;
      min-width: 0;
    }
    .fallfable-exposure,
    .fallfable-benchmark-controls {
      width: 100%;
    }
    .fallfable-benchmark-controls {
      display: grid !important;
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .fallfable-map {
      position: static;
      order: 1;
      justify-self: end;
      width: min(46vw, 172px);
      max-width: 172px;
    }
    .fallfable-panel.is-open .fallfable-map {
      display: none;
    }
    .fallfable-map canvas {
      width: 100% !important;
      height: auto !important;
      aspect-ratio: 1;
    }
  }

  @media (max-width: 420px) {
    .fallfable-map {
      width: min(54vw, 158px);
    }
    .fallfable-details {
      max-height: 76dvh;
    }
  }
`;
document.head.appendChild(layoutStyle);

const canvas = document.createElement('canvas');
canvas.className = 'fallfable-canvas';
document.body.appendChild(canvas);

const panel = document.createElement('section');
panel.className = 'fallfable-panel';
document.body.appendChild(panel);

// Slim status chip; click to expand the full stats and controls.
let statsOpen = false;
const summaryBar = document.createElement('button');
summaryBar.className = 'fallfable-summary';
summaryBar.setAttribute('aria-expanded', 'false');
const summaryChevron = document.createElement('span');
summaryChevron.textContent = '▸';
summaryChevron.style.cssText = 'color:#7d8290;';
const summaryText = document.createElement('span');
summaryText.className = 'fallfable-summary-text';
summaryText.textContent = 'falling';
summaryBar.append(summaryChevron, summaryText);
panel.appendChild(summaryBar);

const details = document.createElement('div');
details.className = 'fallfable-details';
panel.appendChild(details);

summaryBar.addEventListener('click', () => {
  statsOpen = !statsOpen;
  panel.classList.toggle('is-open', statsOpen);
  summaryBar.setAttribute('aria-expanded', String(statsOpen));
  summaryChevron.textContent = statsOpen ? '▾' : '▸';
  details.style.display = statsOpen ? 'grid' : 'none';
});

const readout = document.createElement('div');
readout.className = 'fallfable-readout fallfable-glass';
details.appendChild(readout);

const controls = document.createElement('div');
controls.className = 'fallfable-controls fallfable-glass';
details.appendChild(controls);

// Presets live in their own slim bar along the bottom of the screen.
const presetRow = document.createElement('nav');
presetRow.className = 'fallfable-presets';
document.body.appendChild(presetRow);

const qualitySelect = document.createElement('select');
qualitySelect.className = 'fallfable-select';
[
  ['auto', 'auto'],
  ['half', '0.5'],
  ['three-quarter', '0.75'],
  ['full', '1'],
].forEach(([label, value]) => {
  const option = document.createElement('option');
  option.textContent = label;
  option.value = value;
  qualitySelect.appendChild(option);
});
controls.appendChild(row('resolution', qualitySelect));

const diagnosticSelect = document.createElement('select');
diagnosticSelect.className = 'fallfable-select';
[
  ['normal', '0'],
  ['term', '1'],
  ['cost', '2'],
  ['cost+term', '3'],
  ['classifier', '4'],
  ['tile-classifier', '5'],
  ['shadow-skip', '6'],
  ['shadow-skip+tint', '7'],
  ['sky-skip', '8'],
  ['sky-skip+tint', '9'],
].forEach(([label, value]) => {
  const option = document.createElement('option');
  option.textContent = label;
  option.value = value;
  diagnosticSelect.appendChild(option);
});
controls.appendChild(row('visualize', diagnosticSelect));

const exposureControl = document.createElement('div');
exposureControl.className = 'fallfable-exposure';
exposureControl.style.cssText = 'display:flex;gap:6px;align-items:center;';
const exposureInput = document.createElement('input');
exposureInput.type = 'range';
exposureInput.min = '0.05';
exposureInput.max = '2';
exposureInput.step = '0.05';
exposureInput.value = '1';
exposureInput.style.cssText = 'width:104px;accent-color:#e8b873;';
const exposureReadout = document.createElement('span');
exposureReadout.style.cssText = 'color:#7d8290;min-width:42px;text-align:right;';
exposureControl.append(exposureInput, exposureReadout);
controls.appendChild(row('exposure', exposureControl));

const freezeButton = controlButton('freeze');
const huntButton = controlButton('hunt');
const benchmarkButton = controlButton('bench');
const benchmarkControls = document.createElement('div');
benchmarkControls.className = 'fallfable-benchmark-controls';
benchmarkControls.style.cssText = 'display:flex;gap:6px;';
benchmarkControls.append(freezeButton, huntButton, benchmarkButton);
controls.appendChild(row('measure', benchmarkControls));

const benchmarkOutput = document.createElement('div');
benchmarkOutput.style.cssText = 'color:#7d8290;min-height:15px;';
benchmarkOutput.textContent = 'bench idle';
controls.appendChild(benchmarkOutput);

const paceInput = document.createElement('input');
paceInput.type = 'range';
paceInput.min = '0.1';
paceInput.max = '2.5';
paceInput.step = '0.1';
paceInput.value = '0.9';
paceInput.style.cssText = 'width:130px;accent-color:#e8b873;';
controls.appendChild(row('fall rate', paceInput));

const heightInput = document.createElement('input');
heightInput.type = 'range';
heightInput.min = '-3';
heightInput.max = '3';
heightInput.step = '0.1';
heightInput.value = '0';
heightInput.style.cssText = 'width:130px;accent-color:#e8b873;';
controls.appendChild(row('launch height', heightInput));

const tutorialLink = document.createElement('a');
tutorialLink.href = '/fallfable/tutorial.html';
tutorialLink.textContent = 'how it works →';
tutorialLink.style.cssText = 'color:#e8b873;text-decoration:none;justify-self:end;';
controls.appendChild(tutorialLink);

const mapBox = document.createElement('div');
mapBox.className = 'fallfable-map fallfable-glass';
panel.appendChild(mapBox);

function row(text: string, control: HTMLElement): HTMLElement {
  const label = document.createElement('label');
  label.className = 'fallfable-row';
  const span = document.createElement('span');
  span.textContent = text;
  label.append(span, control);
  return label;
}

function controlButton(text: string): HTMLButtonElement {
  const button = document.createElement('button');
  button.textContent = text;
  button.className = 'fallfable-button';
  return button;
}

// ------------------------------------------------------------------ state

let state: PlayerState = PRESETS[0].create();
let running = true;
let yaw = Math.PI * 1.25; // look inward from the plunge preset
let pitch = 0;
let trajectoryCameraFollow = false;
let preview: PreviewPoint[] = [];
let previewDirty = true;
let lastPreviewAt = 0;
let lastFrameAt = performance.now();
let fpsEma = 60;
let renderer: FallfableRenderer | null = null;
let rendererMessage = 'starting WebGPU…';
let benchmarkRunning = false;
let spikeHuntRunning = false;
let activeQuality: QualityMode = 'auto';
let latestBenchmarkResult: FallfableBenchmarkResult | null = null;
let latestSpikeHuntResult: FallfableSpikeHuntResult | null = null;
let latestBenchmarkSuiteResult: FallfableBenchmarkSuiteResult | null = null;

setExposure(currentExposure());

function setRunning(next: boolean): void {
  running = next;
  updateFreezeButton();
}

function updateFreezeButton(): void {
  freezeButton.textContent = running ? 'freeze' : 'resume';
  freezeButton.disabled = isTimedSamplingActive();
}

updateFreezeButton();

function isTimedSamplingActive(): boolean {
  return benchmarkRunning || spikeHuntRunning;
}

function setSamplingControlsActive(active: boolean): void {
  benchmarkButton.disabled = active;
  huntButton.disabled = active;
  freezeButton.disabled = active;
  benchmarkButton.style.cursor = active ? 'wait' : 'pointer';
  huntButton.style.cursor = active ? 'wait' : 'pointer';
}

function setRunningFromControl(next: boolean): void {
  if (isTimedSamplingActive()) return;
  setRunning(next);
}

for (const preset of PRESETS) {
  const button = document.createElement('button');
  button.textContent = preset.label;
  button.title = preset.description;
  button.className = 'fallfable-preset';
  button.addEventListener('mouseenter', () => { button.style.borderColor = '#e8b873'; });
  button.addEventListener('mouseleave', () => { button.style.borderColor = '#23262e'; });
  button.addEventListener('click', () => applyPreset(preset.id));
  presetRow.appendChild(button);
}

function applyPreset(id: string): void {
  const preset = PRESETS.find((p) => p.id === id);
  if (!preset) return;
  state = preset.create();
  setExposure(preset.exposure ?? 1);
  setRunning(true);
  trajectoryCameraFollow = false;
  faceInward();
  previewDirty = true;
}

function faceInward(): void {
  yaw = Math.atan2(-state.position.y, -state.position.x);
  const rho = Math.hypot(state.position.x, state.position.y);
  pitch = -Math.atan2(state.position.z, Math.max(rho, 0.3)) * 0.8;
}

function faceLaunchHeading(heading: PreviewPoint | null): void {
  if (!heading || Math.hypot(heading.x, heading.y) < 1e-6) {
    faceInward();
    return;
  }
  yaw = Math.atan2(heading.y, heading.x);
  pitch = 0;
}

function followTrajectoryCamera(dt: number): void {
  const u = fourVelocity(state);
  const planarSpeed = Math.hypot(u.x, u.y);
  if (planarSpeed < 1e-6) return;
  const targetYaw = Math.atan2(u.y, u.x);
  const targetPitch = Math.max(-1.2, Math.min(1.2, Math.atan2(u.z, planarSpeed)));
  const alpha = 1 - Math.exp(-Math.max(dt, 0) * 4.5);
  yaw += Math.atan2(Math.sin(targetYaw - yaw), Math.cos(targetYaw - yaw)) * alpha;
  pitch += (targetPitch - pitch) * alpha;
}

const planner = createPlanner(mapBox, {
  launchHeight: () => Number(heightInput.value),
  onPreview(next, heading) {
    state = next;
    setRunning(false);
    trajectoryCameraFollow = false;
    faceLaunchHeading(heading);
    previewDirty = true;
  },
  onCommit(next, heading) {
    state = next;
    setRunning(true);
    trajectoryCameraFollow = Boolean(heading);
    faceLaunchHeading(heading);
    previewDirty = true;
  },
});

// ------------------------------------------------------------------ input

let looking = false;
let lookX = 0;
let lookY = 0;
canvas.addEventListener('pointerdown', (event) => {
  trajectoryCameraFollow = false;
  looking = true;
  lookX = event.clientX;
  lookY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
  canvas.style.cursor = 'grabbing';
});
canvas.addEventListener('pointermove', (event) => {
  if (!looking) return;
  yaw -= (event.clientX - lookX) * 0.0042;
  pitch = Math.max(-1.35, Math.min(1.35, pitch - (event.clientY - lookY) * 0.0042));
  lookX = event.clientX;
  lookY = event.clientY;
});
const stopLook = (event?: PointerEvent) => {
  looking = false;
  if (event && canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  canvas.style.cursor = 'grab';
};
canvas.addEventListener('pointerup', stopLook);
canvas.addEventListener('pointercancel', stopLook);

qualitySelect.addEventListener('change', () => {
  setQuality(qualitySelect.value === 'auto' ? 'auto' : Number(qualitySelect.value));
});

diagnosticSelect.addEventListener('change', () => {
  setDiagnosticMode(currentDiagnosticMode());
});

exposureInput.addEventListener('input', () => {
  setExposure(currentExposure());
});

freezeButton.addEventListener('click', () => {
  setRunningFromControl(!running);
});

benchmarkButton.addEventListener('click', () => {
  void runBenchmark().catch((error: unknown) => {
    benchmarkOutput.textContent = `bench failed: ${errorMessage(error)}`;
  });
});

huntButton.addEventListener('click', () => {
  void huntFrameSpike().catch((error: unknown) => {
    benchmarkOutput.textContent = `hunt failed: ${errorMessage(error)}`;
  });
});

// ------------------------------------------------------------------ loop

void FallfableRenderer.create(canvas, {
  spin: PARAMS.spin,
  mass: PARAMS.mass,
  exposure: currentExposure(),
  baseStep: 0.035,
  escapeRadius: 30,
  maxSteps: 1200,
  singularityCutoff: SINGULARITY_CUTOFF,
  disk: {
    innerRadius: 1.32,
    outerRadius: 13,
    innerTemperature: 7200,
    emissivity: 1.0,
    boostPower: 3.0,
    spinDirection: 1,
    scaleHeight: 0.075,
    absorption: 4.0,
    animationScale: 5,
    hotspotIntensity: 1,
  },
  sky: {
    starIntensity: 1,
    milkyWayIntensity: 0.5,
    ambient: 0.45,
    debugStatus: currentDiagnosticMode(),
  },
}).then((created) => {
  renderer = created;
  setQuality(activeQuality);
  setDiagnosticMode(currentDiagnosticMode());
  setExposure(currentExposure());
  rendererMessage = created ? 'past-directed Kerr GRRT · analytic geodesics' : 'WebGPU is unavailable in this browser';
});

function cameraTetrad(): Tetrad {
  const cosPitch = Math.cos(pitch);
  const forward: Vec3 = normalize3({
    x: Math.cos(yaw) * cosPitch,
    y: Math.sin(yaw) * cosPitch,
    z: Math.sin(pitch),
  });
  const up: Vec3 = { x: -Math.cos(yaw) * Math.sin(pitch), y: -Math.sin(yaw) * Math.sin(pitch), z: cosPitch };
  const p = spatial(state.position);
  try {
    return buildTetrad(p, PARAMS, fourVelocity(state), forward, up);
  } catch {
    // Degenerate hints (e.g. exactly aligned with a boosted axis): retry with
    // the always-valid Eulerian frame.
    return buildTetrad(p, PARAMS, eulerianObserver(p, PARAMS), forward, up);
  }
}

function frame(nowMs: number): void {
  const rawDt = (nowMs - lastFrameAt) / 1000;
  const dt = Math.min(rawDt, 0.04);
  lastFrameAt = nowMs;
  fpsEma += (1 / Math.max(rawDt, 1e-3) - fpsEma) * 0.06;

  const atSingularity = Boolean(state.ended) || state.r <= SINGULARITY_CUTOFF + 1e-9;
  if (running && !atSingularity) {
    // Inside the horizon proper time runs out in a blink; stretch the playback
    // there so the past-light view is actually watchable.
    const insidePace = state.r < HORIZON * 1.5
      ? Math.max(0.08, (state.r - INNER_HORIZON) / (HORIZON * 1.5 - INNER_HORIZON))
      : 1;
    state = stepPlayer(state, dt * Number(paceInput.value) * insidePace);
    if (trajectoryCameraFollow) followTrajectoryCamera(dt);
    if (nowMs - lastPreviewAt > 300) previewDirty = true;
  }

  if (previewDirty) {
    previewDirty = false;
    lastPreviewAt = nowMs;
    preview = previewPath(state, 34);
  }

  if (renderer) {
    renderer.render({
      time: state.position.t,
      position: spatial(state.position),
      tetrad: cameraTetrad(),
      verticalFovRadians: (70 * Math.PI) / 180,
    });
  }

  planner.draw({
    state,
    preview,
    running,
    lookDirection: { x: Math.cos(yaw), y: Math.sin(yaw) },
  });
  updateReadout(atSingularity);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

let lastReadoutAt = 0;
function updateReadout(atSingularity: boolean): void {
  const now = performance.now();
  if (now - lastReadoutAt < 120) return;
  lastReadoutAt = now;
  const stats = renderer?.stats;
  const status = atSingularity
    ? (state.r <= SINGULARITY_CUTOFF + 1e-9 ? 'singularity reached' : 'worldline lost (integration limit)')
    : state.r <= INNER_HORIZON
      ? 'past the Cauchy horizon — geometry unstable'
      : state.r <= HORIZON
        ? 'inside the horizon · past-light view'
        : running ? 'falling' : 'planning';
  summaryText.innerHTML =
    `<span style="color:#e8b873">${status}</span>` +
    `<span style="color:#7d8290"> · r ${state.r.toFixed(2)}</span>`;
  if (!statsOpen) return;
  readout.innerHTML =
    `<div>status: <span style="color:#e8b873">${status}</span></div>` +
    `<div>r: ${state.r.toFixed(3)} · horizon ${HORIZON.toFixed(3)} · a/M ${(PARAMS.spin / PARAMS.mass).toFixed(2)}</div>` +
    `<div>proper time τ: ${state.tau.toFixed(1)} · coordinate t: ${state.position.t.toFixed(1)}</div>` +
    `<div>local speed: ${(localSpeed(state) * 100).toFixed(1)}% c</div>` +
    `<div>constraint |H+½|: ${constraintResidual(state).toExponential(1)}</div>` +
    (stats
      ? stats.gpuTimerAvailable
        ? `<div>render: ${stats.width}×${stats.height} · gpu ${stats.gpuMs.toFixed(1)} ms · ${fpsEma.toFixed(0)} fps</div>`
        : `<div>render: ${stats.width}×${stats.height} · gpu timer unavailable · ${fpsEma.toFixed(0)} fps</div>`
      : '') +
    `<div>viz: ${diagnosticLabel()}</div>` +
    (latestBenchmarkResult
      ? `<div>bench: median ${latestBenchmarkResult.medianGpuMs.toFixed(1)} ms · p95 ${latestBenchmarkResult.p95GpuMs.toFixed(1)} ms · ${latestBenchmarkResult.gpuFramesPerSecond.toFixed(1)} gpu fps</div>`
      : '') +
    (latestSpikeHuntResult
      ? `<div>hotspot: ${latestSpikeHuntResult.spikeGpuMs.toFixed(1)} ms spike · r ${latestSpikeHuntResult.radius.toFixed(2)}</div>`
      : '') +
    `<div>exposure: ${currentExposure().toFixed(currentExposure() < 1 ? 2 : 1)}x</div>` +
    `<div style="color:#7d8290;margin-top:6px">${rendererMessage}</div>` +
    `<div style="color:#7d8290">drag view · map: press-drag-release to launch</div>`;
}

// ------------------------------------------------------------- benchmarking

type QualityMode = 'auto' | number;
type DiagnosticMode = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9;

interface FallfableViewSnapshot {
  state: PlayerState;
  running: boolean;
  yaw: number;
  pitch: number;
  trajectoryCameraFollow?: boolean;
  quality: QualityMode;
  diagnosticMode: DiagnosticMode;
  exposure: number;
}

interface FallfableBenchmarkOptions {
  warmupFrames?: number;
  sampleFrames?: number;
  quality?: number;
  timeoutMs?: number;
  restoreQuality?: boolean;
  restoreRunning?: boolean;
}

interface FallfableBenchmarkResult {
  warmupFrames: number;
  sampleFrames: number;
  quality: number;
  width: number;
  height: number;
  scale: number;
  radius: number;
  coordinateTime: number;
  meanGpuMs: number;
  medianGpuMs: number;
  p95GpuMs: number;
  minGpuMs: number;
  maxGpuMs: number;
  totalGpuMs: number;
  gpuFramesPerSecond: number;
  passMedianGpuMs: Record<string, number>;
  passP95GpuMs: Record<string, number>;
  sampleDurationMs: number;
  completedFramesPerSecond: number;
  samplesGpuMs: number[];
}

interface FallfableBenchmarkPointInfo {
  id: string;
  label: string;
  description: string;
}

interface FallfableBenchmarkPoint extends FallfableBenchmarkPointInfo {
  create(): FallfableViewSnapshot;
}

interface FallfableBenchmarkPointResult extends FallfableBenchmarkPointInfo {
  benchmark: FallfableBenchmarkResult;
}

interface FallfableBenchmarkPointOptions extends FallfableBenchmarkOptions {
  restoreView?: boolean;
}

interface FallfableBenchmarkSuiteOptions extends FallfableBenchmarkOptions {
  points?: string[];
  restoreView?: boolean;
}

interface FallfableBenchmarkSuiteResult {
  points: FallfableBenchmarkPointResult[];
}

interface FallfableSpikeHuntOptions {
  quality?: number;
  /** Let the observer move this long before spike detection starts. */
  minDelayMs?: number;
  timeoutMs?: number;
  baselineFrames?: number;
  thresholdMultiplier?: number;
  /** Absolute spike threshold; overrides the baseline multiplier when set. */
  minSpikeGpuMs?: number;
  consecutiveFrames?: number;
  benchmark?: boolean;
  benchmarkWarmupFrames?: number;
  benchmarkSampleFrames?: number;
  restoreQuality?: boolean;
}

interface FallfableSpikeHuntResult {
  snapshot: FallfableViewSnapshot;
  benchmark: FallfableBenchmarkResult | null;
  quality: number;
  baselineGpuMs: number;
  thresholdGpuMs: number;
  spikeGpuMs: number;
  observedFrames: number;
  elapsedMs: number;
  radius: number;
  coordinateTime: number;
}

function currentQuality(): QualityMode {
  return activeQuality;
}

function setQuality(mode: QualityMode): void {
  activeQuality = mode === 'auto' ? 'auto' : Math.max(0.2, Math.min(1, Number.isFinite(mode) ? mode : 0.75));
  renderer?.setQuality(activeQuality);
  const value = activeQuality === 'auto' ? 'auto' : String(activeQuality);
  if (Array.from(qualitySelect.options).some((option) => option.value === value)) {
    qualitySelect.value = value;
  }
}

function currentDiagnosticMode(): DiagnosticMode {
  return normalizeDiagnosticMode(diagnosticSelect.value);
}

function normalizeDiagnosticMode(mode: unknown): DiagnosticMode {
  const value = Number(mode);
  return (Number.isFinite(value) && value >= 0 && value <= 9 ? Math.floor(value) : 0) as DiagnosticMode;
}

function setDiagnosticMode(mode: number): void {
  const normalized = normalizeDiagnosticMode(mode);
  const value = String(normalized);
  if (Array.from(diagnosticSelect.options).some((option) => option.value === value)) {
    diagnosticSelect.value = value;
  }
  if (renderer) renderer.options.sky.debugStatus = normalized;
}

function currentExposure(): number {
  const value = Number(exposureInput.value);
  return Number.isFinite(value) ? Math.max(value, 0) : 1;
}

function setExposure(exposure: number): void {
  const value = Math.max(0.05, Math.min(2, Number.isFinite(exposure) ? exposure : 1));
  exposureInput.value = value.toFixed(2);
  exposureReadout.textContent = `${value.toFixed(value < 1 ? 2 : 1)}x`;
  if (renderer) renderer.options.exposure = value;
}

function diagnosticLabel(): string {
  return diagnosticSelect.options[diagnosticSelect.selectedIndex]?.textContent ?? 'normal';
}

function clonePlayerState(source: PlayerState): PlayerState {
  return {
    ...source,
    position: { ...source.position },
    momentum: { ...source.momentum },
  };
}

function captureView(): FallfableViewSnapshot {
  return {
    state: clonePlayerState(state),
    running,
    yaw,
    pitch,
    trajectoryCameraFollow,
    quality: currentQuality(),
    diagnosticMode: currentDiagnosticMode(),
    exposure: currentExposure(),
  };
}

function restoreView(snapshot: FallfableViewSnapshot): void {
  state = clonePlayerState(snapshot.state);
  yaw = snapshot.yaw;
  pitch = snapshot.pitch;
  trajectoryCameraFollow = snapshot.trajectoryCameraFollow ?? false;
  setRunning(snapshot.running);
  setQuality(snapshot.quality);
  setDiagnosticMode(snapshot.diagnosticMode);
  setExposure(snapshot.exposure ?? 1);
  previewDirty = true;
}

const BENCHMARK_POINTS: FallfableBenchmarkPoint[] = [
  {
    id: 'outer-disk',
    label: 'outer disk',
    description: 'wide disk and sky view from above the outer disk',
    create: () => benchmarkSnapshot(launchLocal({
      r: 10,
      phi: Math.PI * 0.25,
      betaRadial: 0,
      betaTangential: 0.05,
      height: 1.6,
    })),
  },
  {
    id: 'disk-graze',
    label: 'disk graze',
    description: 'low-angle view through the bright disk atmosphere',
    create: () => benchmarkSnapshot(launchLocal({
      r: 7,
      phi: Math.PI / 2,
      betaRadial: -0.05,
      betaTangential: 0.27,
      height: 0.25,
    }), -0.05),
  },
  {
    id: 'adaptive-boundary',
    label: 'adaptive boundary',
    description: 'exterior view near the adaptive sky cutoff radius',
    create: () => benchmarkSnapshot(launchLocal({
      r: 4.1,
      phi: Math.PI * 0.38,
      betaRadial: -0.02,
      betaTangential: 0.18,
      height: 0.55,
    }), -0.04),
  },
  {
    id: 'isco-lens',
    label: 'ISCO lens',
    description: 'marginally stable orbit with strong near-disk lensing',
    create: () => benchmarkSnapshot(presetState('isco'), 0, presetExposure('isco')),
  },
  {
    id: 'photon-whirl',
    label: 'photon whirl',
    description: 'near photon-orbit view where many rays skim before escaping',
    create: () => benchmarkSnapshot(presetState('whirl'), 0, presetExposure('whirl')),
  },
  {
    id: 'horizon-graze',
    label: 'horizon graze',
    description: 'just outside the outer horizon, aimed across the disk',
    create: () => benchmarkSnapshot(launchLocal({
      r: HORIZON * 1.04,
      phi: -0.35,
      betaRadial: -0.18,
      betaTangential: 0.35,
      height: 0.08,
    }), -0.08),
  },
  {
    id: 'inner-horizon',
    label: 'inner horizon',
    description: 'inside the black hole near the Cauchy horizon stress region',
    create: () => benchmarkSnapshot(launchLocal({
      r: Math.max(INNER_HORIZON * 1.12, 0.28),
      phi: 2.1,
      betaRadial: -0.04,
      betaTangential: 0.12,
      height: 0.02,
    }), 0.04),
  },
  {
    id: 'polar-halo',
    label: 'polar halo',
    description: 'high-axis view where the disk forms a lensed halo',
    create: () => benchmarkSnapshot(presetState('polar')),
  },
];

function benchmarkSnapshot(pointState: PlayerState, pitchBias = 0, exposure = currentExposure()): FallfableViewSnapshot {
  const view = inwardView(pointState);
  return {
    state: clonePlayerState(pointState),
    running: false,
    yaw: view.yaw,
    pitch: Math.max(-1.35, Math.min(1.35, view.pitch + pitchBias)),
    quality: currentQuality(),
    diagnosticMode: currentDiagnosticMode(),
    exposure,
  };
}

function presetExposure(id: string): number {
  return PRESETS.find((entry) => entry.id === id)?.exposure ?? 1;
}

function inwardView(source: PlayerState): { yaw: number; pitch: number } {
  const rho = Math.hypot(source.position.x, source.position.y);
  return {
    yaw: Math.atan2(-source.position.y, -source.position.x),
    pitch: -Math.atan2(source.position.z, Math.max(rho, 0.3)) * 0.8,
  };
}

function presetState(id: string): PlayerState {
  const preset = PRESETS.find((entry) => entry.id === id);
  if (!preset) throw new Error(`Missing benchmark preset: ${id}`);
  return preset.create();
}

function listBenchmarkPoints(): FallfableBenchmarkPointInfo[] {
  return BENCHMARK_POINTS.map(({ id, label, description }) => ({ id, label, description }));
}

function benchmarkPointById(id: string): FallfableBenchmarkPoint {
  const point = BENCHMARK_POINTS.find((entry) => entry.id === id);
  if (!point) {
    const choices = BENCHMARK_POINTS.map((entry) => entry.id).join(', ');
    throw new Error(`Unknown benchmark point "${id}". Choose one of: ${choices}`);
  }
  return point;
}

async function runBenchmarkPoint(
  id: string,
  options: FallfableBenchmarkPointOptions = {},
): Promise<FallfableBenchmarkPointResult> {
  const point = benchmarkPointById(id);
  const previous = captureView();
  const { restoreView: shouldRestoreView = true, ...benchmarkOptions } = options;

  try {
    restoreView(point.create());
    const benchmark = await runBenchmark(benchmarkOptions);
    return {
      id: point.id,
      label: point.label,
      description: point.description,
      benchmark,
    };
  } finally {
    if (shouldRestoreView) restoreView(previous);
  }
}

async function runBenchmarkSuite(
  options: FallfableBenchmarkSuiteOptions = {},
): Promise<FallfableBenchmarkSuiteResult> {
  const previous = captureView();
  const { points, restoreView: shouldRestoreView = true, ...benchmarkOptions } = options;
  const ids = points?.length ? points : BENCHMARK_POINTS.map((point) => point.id);
  const results: FallfableBenchmarkPointResult[] = [];

  try {
    for (const id of ids) {
      results.push(await runBenchmarkPoint(id, { ...benchmarkOptions, restoreView: false }));
    }
    latestBenchmarkSuiteResult = { points: results };
    return latestBenchmarkSuiteResult;
  } finally {
    if (shouldRestoreView) restoreView(previous);
  }
}

async function runBenchmark(
  options: FallfableBenchmarkOptions = {},
  source: 'user' | 'spike-hunt' = 'user',
): Promise<FallfableBenchmarkResult> {
  const activeRenderer = renderer;
  if (!activeRenderer) throw new Error('Renderer is not ready');
  if (!activeRenderer.stats.gpuTimerAvailable) throw new Error('WebGPU timestamp-query is unavailable');
  if (benchmarkRunning) throw new Error('Benchmark already running');
  if (spikeHuntRunning && source !== 'spike-hunt') throw new Error('Spike hunt already running');

  benchmarkRunning = true;
  setSamplingControlsActive(true);
  const previous = captureView();
  const warmupFrames = Math.max(0, Math.floor(options.warmupFrames ?? 30));
  const sampleFrames = Math.max(1, Math.floor(options.sampleFrames ?? 120));
  const fallbackQuality = previous.quality === 'auto' ? 0.75 : previous.quality;
  const quality = Math.max(0.2, Math.min(1, options.quality ?? fallbackQuality));

  try {
    setRunning(false);
    setQuality(quality);
    benchmarkOutput.textContent = `bench warming 0/${warmupFrames} @ ${quality}`;
    activeRenderer.resetFrameTimings();
    if (warmupFrames > 0) {
      await collectRendererTimings(activeRenderer, warmupFrames, (count) => {
        benchmarkOutput.textContent = `bench warming ${count}/${warmupFrames} @ ${quality}`;
      }, options.timeoutMs);
    }

    benchmarkOutput.textContent = `bench sampling 0/${sampleFrames} @ ${quality}`;
    activeRenderer.resetFrameTimings();
    const timings = await collectRendererTimings(activeRenderer, sampleFrames, (count) => {
      benchmarkOutput.textContent = `bench sampling ${count}/${sampleFrames} @ ${quality}`;
    }, options.timeoutMs);
    const result = summarizeTimings(timings, warmupFrames, sampleFrames, quality);
    latestBenchmarkResult = result;
    benchmarkOutput.textContent =
      `bench ${quality}: median ${result.medianGpuMs.toFixed(1)} ms, p95 ${result.p95GpuMs.toFixed(1)} ms, ${result.gpuFramesPerSecond.toFixed(1)} fps`;
    return result;
  } finally {
    if (options.restoreQuality ?? true) setQuality(previous.quality);
    if (options.restoreRunning ?? true) setRunning(previous.running);
    benchmarkRunning = false;
    setSamplingControlsActive(isTimedSamplingActive());
  }
}

async function collectRendererTimings(
  activeRenderer: FallfableRenderer,
  count: number,
  onProgress?: (count: number) => void,
  timeoutMs = 60000,
): Promise<RendererFrameTiming[]> {
  const timings: RendererFrameTiming[] = [];
  const startedAt = performance.now();
  while (timings.length < count) {
    if (performance.now() - startedAt > timeoutMs) {
      throw new Error(`Timed out after ${timeoutMs} ms waiting for ${count} rendered frames`);
    }
    await animationFrame();
    const before = timings.length;
    timings.push(...activeRenderer.consumeFrameTimings());
    if (timings.length !== before) onProgress?.(Math.min(timings.length, count));
  }
  return timings.slice(0, count);
}

function animationFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

function summarizeTimings(
  timings: RendererFrameTiming[],
  warmupFrames: number,
  sampleFrames: number,
  quality: number,
): FallfableBenchmarkResult {
  const samples = timings.map((timing) => timing.gpuMs);
  const sorted = [...samples].sort((a, b) => a - b);
  const sum = samples.reduce((total, value) => total + value, 0);
  const first = timings[0];
  const last = timings[timings.length - 1];
  const sampleDurationMs = first && last ? Math.max(last.completedAt - first.completedAt, 1) : 0;
  return {
    warmupFrames,
    sampleFrames,
    quality,
    width: last?.width ?? 0,
    height: last?.height ?? 0,
    scale: last?.scale ?? quality,
    radius: state.r,
    coordinateTime: state.position.t,
    meanGpuMs: sum / Math.max(samples.length, 1),
    medianGpuMs: percentile(sorted, 0.5),
    p95GpuMs: percentile(sorted, 0.95),
    minGpuMs: sorted[0] ?? 0,
    maxGpuMs: sorted[sorted.length - 1] ?? 0,
    totalGpuMs: sum,
    gpuFramesPerSecond: sum > 0 ? (samples.length * 1000) / sum : 0,
    passMedianGpuMs: summarizePassTimings(timings, 0.5),
    passP95GpuMs: summarizePassTimings(timings, 0.95),
    sampleDurationMs,
    completedFramesPerSecond: sampleDurationMs > 0 ? (samples.length * 1000) / sampleDurationMs : 0,
    samplesGpuMs: samples,
  };
}

function summarizePassTimings(timings: RendererFrameTiming[], fraction: number): Record<string, number> {
  const passes = [
    ['classifier', 'classifierGpuMs'],
    ['trace', 'traceGpuMs'],
    ['output', 'outputGpuMs'],
    ['present', 'presentGpuMs'],
  ] as const;
  const summary: Record<string, number> = {};
  for (const [name, field] of passes) {
    const values = timings
      .map((timing) => timing[field])
      .filter((value): value is number => Number.isFinite(value));
    if (values.length > 0) {
      summary[name] = percentile([...values].sort((a, b) => a - b), fraction);
    }
  }
  return summary;
}

async function huntFrameSpike(options: FallfableSpikeHuntOptions = {}): Promise<FallfableSpikeHuntResult> {
  const activeRenderer = renderer;
  if (!activeRenderer) throw new Error('Renderer is not ready');
  if (!activeRenderer.stats.gpuTimerAvailable) throw new Error('WebGPU timestamp-query is unavailable');
  if (benchmarkRunning) throw new Error('Benchmark already running');
  if (spikeHuntRunning) throw new Error('Spike hunt already running');

  spikeHuntRunning = true;
  setSamplingControlsActive(true);

  const previous = captureView();
  const fallbackQuality = previous.quality === 'auto' ? 0.75 : previous.quality;
  const quality = Math.max(0.2, Math.min(1, options.quality ?? fallbackQuality));
  const minDelayMs = Math.max(0, options.minDelayMs ?? 2000);
  const timeoutMs = Math.max(minDelayMs + 1000, options.timeoutMs ?? 18000);
  const baselineFrames = Math.max(1, Math.floor(options.baselineFrames ?? 24));
  const thresholdMultiplier = Math.max(1, options.thresholdMultiplier ?? 1.45);
  const consecutiveFrames = Math.max(1, Math.floor(options.consecutiveFrames ?? 1));
  const baselineSamples: number[] = [];
  let consecutiveSpikes = 0;
  let observedFrames = 0;
  let baselineGpuMs = 0;
  let thresholdGpuMs = options.minSpikeGpuMs ?? 0;

  try {
    setQuality(quality);
    setRunning(true);
    activeRenderer.resetFrameTimings();
    const startedAt = performance.now();
    benchmarkOutput.textContent = `hunt waiting ${Math.round(minDelayMs)} ms @ ${quality}`;

    while (performance.now() - startedAt <= timeoutMs) {
      await animationFrame();
      const elapsedMs = performance.now() - startedAt;
      const timings = activeRenderer.consumeFrameTimings();
      if (elapsedMs < minDelayMs) {
        if (timings.length > 0) benchmarkOutput.textContent = `hunt waiting ${Math.round(minDelayMs - elapsedMs)} ms @ ${quality}`;
        continue;
      }

      for (const timing of timings) {
        observedFrames++;
        if (baselineSamples.length < baselineFrames && options.minSpikeGpuMs === undefined) {
          baselineSamples.push(timing.gpuMs);
          baselineGpuMs = percentile([...baselineSamples].sort((a, b) => a - b), 0.5);
          thresholdGpuMs = baselineGpuMs * thresholdMultiplier;
          benchmarkOutput.textContent = `hunt baseline ${baselineSamples.length}/${baselineFrames} @ ${quality}`;
          continue;
        }

        if (options.minSpikeGpuMs !== undefined) {
          thresholdGpuMs = options.minSpikeGpuMs;
          baselineGpuMs = baselineGpuMs || activeRenderer.stats.gpuMs;
        }

        const isSpike = timing.gpuMs >= thresholdGpuMs;
        consecutiveSpikes = isSpike ? consecutiveSpikes + 1 : 0;
        benchmarkOutput.textContent =
          `hunt gpu ${timing.gpuMs.toFixed(1)} / ${thresholdGpuMs.toFixed(1)} ms @ r ${state.r.toFixed(2)}`;

        if (consecutiveSpikes >= consecutiveFrames) {
          setRunning(false);
          const snapshot = captureView();
          let benchmark: FallfableBenchmarkResult | null = null;
          if (options.benchmark ?? true) {
            benchmark = await runBenchmark({
              warmupFrames: options.benchmarkWarmupFrames ?? 20,
              sampleFrames: options.benchmarkSampleFrames ?? 100,
              quality,
              restoreQuality: false,
              restoreRunning: false,
            }, 'spike-hunt');
          }

          const result: FallfableSpikeHuntResult = {
            snapshot,
            benchmark,
            quality,
            baselineGpuMs,
            thresholdGpuMs,
            spikeGpuMs: timing.gpuMs,
            observedFrames,
            elapsedMs: timing.completedAt - startedAt,
            radius: state.r,
            coordinateTime: state.position.t,
          };
          latestSpikeHuntResult = result;
          benchmarkOutput.textContent = benchmark
            ? `hunt r ${result.radius.toFixed(2)}: median ${benchmark.medianGpuMs.toFixed(1)} ms`
            : `hunt froze r ${result.radius.toFixed(2)} on ${result.spikeGpuMs.toFixed(1)} ms`;
          return result;
        }
      }
    }

    restoreView(previous);
    throw new Error(`No spike found after ${Math.round(timeoutMs)} ms`);
  } finally {
    if (options.restoreQuality ?? true) setQuality(previous.quality);
    spikeHuntRunning = false;
    setSamplingControlsActive(isTimedSamplingActive());
  }
}

function percentile(sorted: number[], fraction: number): number {
  if (sorted.length === 0) return 0;
  const index = (sorted.length - 1) * fraction;
  const lo = Math.floor(index);
  const hi = Math.ceil(index);
  const t = index - lo;
  return sorted[lo] * (1 - t) + sorted[hi] * t;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

// ------------------------------------------------------------------ debug

interface FallfableDebugApi {
  getState(): PlayerState;
  snapshot(): FallfableViewSnapshot;
  restore(snapshot: FallfableViewSnapshot): void;
  fastForward(tau: number): PlayerState;
  freeze(): void;
  resume(): void;
  pause(): void;
  preset(id: string): void;
  setQuality(mode: QualityMode): void;
  setExposure(exposure: number): void;
  setDiagnosticMode(mode: number): void;
  diagnosticMode(): DiagnosticMode;
  benchmark(options?: FallfableBenchmarkOptions): Promise<FallfableBenchmarkResult>;
  latestBenchmark(): FallfableBenchmarkResult | null;
  benchmarkPoints(): FallfableBenchmarkPointInfo[];
  benchmarkPoint(id: string, options?: FallfableBenchmarkPointOptions): Promise<FallfableBenchmarkPointResult>;
  benchmarkSuite(options?: FallfableBenchmarkSuiteOptions): Promise<FallfableBenchmarkSuiteResult>;
  latestBenchmarkSuite(): FallfableBenchmarkSuiteResult | null;
  huntSpike(options?: FallfableSpikeHuntOptions): Promise<FallfableSpikeHuntResult>;
  latestSpikeHunt(): FallfableSpikeHuntResult | null;
  renderer(): FallfableRenderer | null;
}

declare global {
  interface Window {
    __fallfable?: FallfableDebugApi;
  }
}

const fallfableDebugApi: FallfableDebugApi = {
  getState: () => state,
  snapshot: captureView,
  restore: restoreView,
  fastForward(tau: number) {
    state = stepPlayer(state, tau);
    previewDirty = true;
    return state;
  },
  freeze() {
    setRunningFromControl(false);
  },
  resume() {
    setRunningFromControl(true);
  },
  pause() {
    setRunningFromControl(false);
  },
  preset(id: string) {
    applyPreset(id);
  },
  setQuality,
  setExposure,
  setDiagnosticMode,
  diagnosticMode: currentDiagnosticMode,
  benchmark: runBenchmark,
  latestBenchmark: () => latestBenchmarkResult,
  benchmarkPoints: listBenchmarkPoints,
  benchmarkPoint: runBenchmarkPoint,
  benchmarkSuite: runBenchmarkSuite,
  latestBenchmarkSuite: () => latestBenchmarkSuiteResult,
  huntSpike: huntFrameSpike,
  latestSpikeHunt: () => latestSpikeHuntResult,
  renderer: () => renderer,
};

window.__fallfable = fallfableDebugApi;

if (import.meta.env.DEV) {
  installFallfableDevBridge(fallfableDebugApi);
}

interface FallfableDevBridgePayload {
  status: 'ready' | 'running' | 'complete' | 'error';
  command: string;
  nonce: string | null;
  updatedAt: number;
  result?: unknown;
  error?: string;
}

function installFallfableDevBridge(api: FallfableDebugApi): void {
  const output = document.createElement('script');
  output.id = 'fallfable-dev-result';
  output.type = 'application/json';
  document.head.appendChild(output);

  let lastCommand = '';
  const writePayload = (payload: FallfableDevBridgePayload) => {
    output.textContent = JSON.stringify(payload);
    document.documentElement.dataset.fallfableDevStatus = payload.status;
    document.documentElement.dataset.fallfableDevCommand = payload.command;
  };

  const runHashCommand = () => {
    const raw = location.hash.startsWith('#') ? location.hash.slice(1) : location.hash;
    const params = new URLSearchParams(raw);
    const command = params.get('fallfable');
    if (!command || raw === lastCommand) return;
    lastCommand = raw;
    void runDevCommand(api, command, params, writePayload);
  };

  window.addEventListener('hashchange', runHashCommand);
  writePayload({ status: 'ready', command: 'none', nonce: null, updatedAt: performance.now() });
  queueMicrotask(runHashCommand);
}

async function runDevCommand(
  api: FallfableDebugApi,
  command: string,
  params: URLSearchParams,
  writePayload: (payload: FallfableDevBridgePayload) => void,
): Promise<void> {
  const nonce = params.get('nonce');
  writePayload({ status: 'running', command, nonce, updatedAt: performance.now() });

  try {
    let result: unknown;
    switch (command) {
      case 'bench':
        await waitForRenderer();
        result = await api.benchmark(benchmarkOptionsFromParams(params));
        break;
      case 'benchPoint':
        await waitForRenderer();
        result = await api.benchmarkPoint(params.get('point') ?? BENCHMARK_POINTS[0].id, benchmarkPointOptionsFromParams(params));
        break;
      case 'benchSuite':
        await waitForRenderer();
        result = await api.benchmarkSuite(benchmarkSuiteOptionsFromParams(params));
        break;
      case 'hunt':
        await waitForRenderer();
        result = await api.huntSpike(spikeHuntOptionsFromParams(params));
        break;
      case 'diagnostic': {
        const mode = diagnosticModeFromParams(params);
        api.setDiagnosticMode(mode);
        result = { mode, snapshot: api.snapshot() };
        break;
      }
      case 'exposure': {
        const exposure = numberParam(params, 'value') ?? numberParam(params, 'exposure') ?? 1;
        api.setExposure(exposure);
        result = api.snapshot();
        break;
      }
      case 'freeze':
        api.freeze();
        result = api.snapshot();
        break;
      case 'resume':
        api.resume();
        result = api.snapshot();
        break;
      default:
        throw new Error(`Unknown fallfable dev command: ${command}`);
    }
    writePayload({ status: 'complete', command, nonce, updatedAt: performance.now(), result });
  } catch (error) {
    writePayload({ status: 'error', command, nonce, updatedAt: performance.now(), error: errorMessage(error) });
  }
}

async function waitForRenderer(timeoutMs = 10000): Promise<void> {
  const startedAt = performance.now();
  while (!renderer && rendererMessage === 'starting WebGPU…') {
    if (performance.now() - startedAt > timeoutMs) throw new Error('Timed out waiting for renderer');
    await animationFrame();
  }
  if (!renderer) throw new Error(rendererMessage);
}

function benchmarkOptionsFromParams(params: URLSearchParams): FallfableBenchmarkOptions {
  return {
    warmupFrames: numberParam(params, 'warmup') ?? numberParam(params, 'warmupFrames'),
    sampleFrames: numberParam(params, 'samples') ?? numberParam(params, 'sampleFrames'),
    quality: numberParam(params, 'quality'),
    timeoutMs: numberParam(params, 'timeout') ?? numberParam(params, 'timeoutMs'),
    restoreQuality: booleanParam(params, 'restoreQuality'),
    restoreRunning: booleanParam(params, 'restoreRunning'),
  };
}

function diagnosticModeFromParams(params: URLSearchParams): DiagnosticMode {
  const raw = params.get('mode') ?? params.get('viz') ?? params.get('diagnostic') ?? '0';
  if (raw === 'normal') return 0;
  if (raw === 'term' || raw === 'termination') return 1;
  if (raw === 'cost') return 2;
  if (raw === 'combined' || raw === 'cost+term' || raw === 'cost term') return 3;
  if (raw === 'class' || raw === 'classifier' || raw === 'grid-classifier') return 4;
  if (raw === 'tile' || raw === 'tile-classifier' || raw === 'adaptive-mask') return 5;
  if (raw === 'shadow' || raw === 'shadow-skip' || raw === 'shadow-probe') return 6;
  if (
    raw === 'shadow+tint' ||
    raw === 'shadow tint' ||
    raw === 'shadow-skip+tint' ||
    raw === 'shadow-skip tint' ||
    raw === 'shadow-probe-tint'
  ) return 7;
  if (raw === 'sky' || raw === 'sky-skip' || raw === 'sky-probe' || raw === 'adaptive-sky') return 8;
  if (
    raw === 'sky+tint' ||
    raw === 'sky tint' ||
    raw === 'sky-skip+tint' ||
    raw === 'sky-skip tint' ||
    raw === 'sky-skip-tint' ||
    raw === 'sky-probe-tint' ||
    raw === 'adaptive-sky-tint'
  ) return 9;
  return normalizeDiagnosticMode(raw);
}

function benchmarkPointOptionsFromParams(params: URLSearchParams): FallfableBenchmarkPointOptions {
  return {
    ...benchmarkOptionsFromParams(params),
    restoreView: booleanParam(params, 'restoreView'),
  };
}

function benchmarkSuiteOptionsFromParams(params: URLSearchParams): FallfableBenchmarkSuiteOptions {
  return {
    ...benchmarkOptionsFromParams(params),
    points: stringListParam(params, 'points'),
    restoreView: booleanParam(params, 'restoreView'),
  };
}

function spikeHuntOptionsFromParams(params: URLSearchParams): FallfableSpikeHuntOptions {
  return {
    quality: numberParam(params, 'quality'),
    minDelayMs: numberParam(params, 'minDelay') ?? numberParam(params, 'minDelayMs'),
    timeoutMs: numberParam(params, 'timeout') ?? numberParam(params, 'timeoutMs'),
    baselineFrames: numberParam(params, 'baseline') ?? numberParam(params, 'baselineFrames'),
    thresholdMultiplier: numberParam(params, 'threshold') ?? numberParam(params, 'thresholdMultiplier'),
    minSpikeGpuMs: numberParam(params, 'minSpike') ?? numberParam(params, 'minSpikeGpuMs'),
    consecutiveFrames: numberParam(params, 'consecutive') ?? numberParam(params, 'consecutiveFrames'),
    benchmark: booleanParam(params, 'benchmark'),
    benchmarkWarmupFrames: numberParam(params, 'benchmarkWarmup') ?? numberParam(params, 'benchmarkWarmupFrames'),
    benchmarkSampleFrames: numberParam(params, 'benchmarkSamples') ?? numberParam(params, 'benchmarkSampleFrames'),
    restoreQuality: booleanParam(params, 'restoreQuality'),
  };
}

function stringListParam(params: URLSearchParams, name: string): string[] | undefined {
  const raw = params.get(name);
  if (raw === null || raw.trim() === '') return undefined;
  const values = raw.split(',').map((value) => value.trim()).filter(Boolean);
  return values.length > 0 ? values : undefined;
}

function numberParam(params: URLSearchParams, name: string): number | undefined {
  const raw = params.get(name);
  if (raw === null || raw.trim() === '') return undefined;
  const value = Number(raw);
  return Number.isFinite(value) ? value : undefined;
}

function booleanParam(params: URLSearchParams, name: string): boolean | undefined {
  const raw = params.get(name);
  if (raw === null) return undefined;
  if (raw === '1' || raw === 'true') return true;
  if (raw === '0' || raw === 'false') return false;
  return undefined;
}
