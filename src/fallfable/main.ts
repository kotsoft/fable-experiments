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
  localSpeed,
  previewPath,
  stepPlayer,
  type PlayerState,
  type PreviewPoint,
} from './player';
import { FallfableRenderer } from './renderer';

// ----------------------------------------------------------------- layout

const canvas = document.createElement('canvas');
canvas.style.cssText =
  'position:fixed;inset:0;width:100vw;height:100vh;display:block;background:#000;cursor:grab;' +
  'touch-action:none;user-select:none;';
document.body.style.margin = '0';
document.body.style.background = '#000';
document.body.appendChild(canvas);

const panel = document.createElement('section');
panel.style.cssText =
  'position:fixed;right:14px;bottom:14px;display:grid;gap:6px;justify-items:end;color:#c9cdd6;' +
  'font:12px ui-monospace,SFMono-Regular,Consolas,monospace;user-select:none;z-index:2;';
document.body.appendChild(panel);

// Slim status chip; click to expand the full stats and controls.
let statsOpen = false;
const summaryBar = document.createElement('button');
summaryBar.style.cssText =
  'display:flex;gap:8px;align-items:center;background:#080910b8;border:1px solid #23262e;' +
  'border-radius:6px;padding:5px 11px;color:#c9cdd6;cursor:pointer;font:inherit;' +
  'backdrop-filter:blur(6px);';
const summaryChevron = document.createElement('span');
summaryChevron.textContent = '▸';
summaryChevron.style.cssText = 'color:#7d8290;';
const summaryText = document.createElement('span');
summaryText.textContent = 'falling';
summaryBar.append(summaryChevron, summaryText);
panel.appendChild(summaryBar);

const details = document.createElement('div');
details.style.cssText = 'display:none;grid-template-rows:auto auto;gap:6px;justify-items:stretch;';
panel.appendChild(details);

summaryBar.addEventListener('click', () => {
  statsOpen = !statsOpen;
  summaryChevron.textContent = statsOpen ? '▾' : '▸';
  details.style.display = statsOpen ? 'grid' : 'none';
});

const readout = document.createElement('div');
readout.style.cssText =
  'background:#080910cc;border:1px solid #23262e;border-radius:6px;padding:9px 11px;' +
  'backdrop-filter:blur(6px);line-height:1.6;min-width:300px;';
details.appendChild(readout);

const controls = document.createElement('div');
controls.style.cssText =
  'background:#080910cc;border:1px solid #23262e;border-radius:6px;padding:9px 11px;' +
  'display:grid;gap:8px;backdrop-filter:blur(6px);';
details.appendChild(controls);

// Presets live in their own slim bar along the bottom of the screen.
const presetRow = document.createElement('nav');
presetRow.style.cssText =
  'position:fixed;left:50%;bottom:14px;transform:translateX(-50%);display:flex;gap:6px;' +
  'flex-wrap:wrap;justify-content:center;max-width:min(92vw,760px);z-index:2;' +
  'font:11px ui-monospace,SFMono-Regular,Consolas,monospace;user-select:none;';
document.body.appendChild(presetRow);

const qualitySelect = document.createElement('select');
qualitySelect.style.cssText =
  'background:#171a22;color:#e6eaf4;border:1px solid #3b4150;border-radius:5px;padding:4px 6px;' +
  'font:12px ui-monospace,monospace;';
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
mapBox.style.cssText =
  'background:#080910cc;border:1px solid #23262e;border-radius:6px;overflow:hidden;' +
  'box-shadow:0 10px 30px #0008;';
panel.appendChild(mapBox);

function row(text: string, control: HTMLElement): HTMLElement {
  const label = document.createElement('label');
  label.style.cssText = 'display:flex;gap:8px;align-items:center;justify-content:space-between;';
  const span = document.createElement('span');
  span.textContent = text;
  label.append(span, control);
  return label;
}

// ------------------------------------------------------------------ state

let state: PlayerState = PRESETS[0].create();
let running = true;
let yaw = Math.PI * 1.25; // look inward from the plunge preset
let pitch = 0;
let preview: PreviewPoint[] = [];
let previewDirty = true;
let lastPreviewAt = 0;
let lastFrameAt = performance.now();
let fpsEma = 60;
let renderer: FallfableRenderer | null = null;
let rendererMessage = 'starting WebGPU…';

for (const preset of PRESETS) {
  const button = document.createElement('button');
  button.textContent = preset.label;
  button.title = preset.description;
  button.style.cssText =
    'background:#080910b8;color:#c9cdd6;border:1px solid #23262e;border-radius:999px;' +
    'padding:5px 11px;font:inherit;cursor:pointer;backdrop-filter:blur(6px);';
  button.addEventListener('mouseenter', () => { button.style.borderColor = '#e8b873'; });
  button.addEventListener('mouseleave', () => { button.style.borderColor = '#23262e'; });
  button.addEventListener('click', () => applyPreset(preset.id));
  presetRow.appendChild(button);
}

function applyPreset(id: string): void {
  const preset = PRESETS.find((p) => p.id === id);
  if (!preset) return;
  state = preset.create();
  running = true;
  faceInward();
  previewDirty = true;
}

function faceInward(): void {
  yaw = Math.atan2(-state.position.y, -state.position.x);
  const rho = Math.hypot(state.position.x, state.position.y);
  pitch = -Math.atan2(state.position.z, Math.max(rho, 0.3)) * 0.8;
}

const planner = createPlanner(mapBox, {
  launchHeight: () => Number(heightInput.value),
  onPreview(next) {
    state = next;
    running = false;
    faceInward();
    previewDirty = true;
  },
  onCommit(next) {
    state = next;
    running = true;
    faceInward();
    previewDirty = true;
  },
});

// ------------------------------------------------------------------ input

let looking = false;
let lookX = 0;
let lookY = 0;
canvas.addEventListener('pointerdown', (event) => {
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
  renderer?.setQuality(qualitySelect.value === 'auto' ? 'auto' : Number(qualitySelect.value));
});

// ------------------------------------------------------------------ loop

void FallfableRenderer.create(canvas, {
  spin: PARAMS.spin,
  mass: PARAMS.mass,
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
  },
  sky: {
    starIntensity: 1,
    milkyWayIntensity: 0.5,
    ambient: 0.45,
  },
}).then((created) => {
  renderer = created;
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
    const insidePace = state.r < HORIZON * 1.25
      ? Math.max(0.15, (state.r - INNER_HORIZON) / (HORIZON * 1.25 - INNER_HORIZON))
      : 1;
    state = stepPlayer(state, dt * Number(paceInput.value) * insidePace);
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
    ? 'worldline ended'
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
      ? `<div>render: ${stats.width}×${stats.height} · gpu ${stats.gpuMs.toFixed(1)} ms · ${fpsEma.toFixed(0)} fps</div>`
      : '') +
    `<div style="color:#7d8290;margin-top:6px">${rendererMessage}</div>` +
    `<div style="color:#7d8290">drag view · map: press-drag-release to launch</div>`;
}

// ------------------------------------------------------------------ debug

interface FallfableDebugApi {
  getState(): PlayerState;
  fastForward(tau: number): PlayerState;
  pause(): void;
  preset(id: string): void;
  renderer(): FallfableRenderer | null;
}

(window as Window & { __fallfable?: FallfableDebugApi }).__fallfable = {
  getState: () => state,
  fastForward(tau: number) {
    state = stepPlayer(state, tau);
    previewDirty = true;
    return state;
  },
  pause() {
    running = false;
  },
  preset(id: string) {
    applyPreset(id);
  },
  renderer: () => renderer,
};
