import { type CompositeCameraSampleOptions } from '../gr/compositeSamples';
import { renderWebGpuCompositeFromCameraToCanvas } from '../gr-demo/webgpuCompositeProbe';
import { createMinimap, type MapPosition, type MapVector } from './minimap';
import {
  FALL_PARAMS,
  HORIZON_RADIUS,
  SINGULARITY_CUTOFF,
  isHorizonCrossed,
  isSingularityReached,
  launchFromLocal,
  positionFromState,
  previewFall,
  spatialPositionFromState,
  stepFall,
  timelikeResidual,
  type FallState,
} from './physics';
import {
  cameraFlatForward,
  createPlayerCamera,
  observerFrameFromState,
  rotatePlayerCamera,
  setCameraLookDirection,
  tetradResidual,
} from './tetrad';

const canvas = document.createElement('canvas');
canvas.style.cssText =
  'position:fixed;inset:0;width:100vw;height:100vh;display:block;background:#000;cursor:grab;' +
  'touch-action:none;user-select:none;';
document.body.style.margin = '0';
document.body.style.background = '#000';
document.body.appendChild(canvas);

const panel = document.createElement('section');
panel.style.cssText =
  'position:fixed;right:14px;bottom:14px;display:grid;gap:8px;color:#c9cdd6;' +
  'font:12px ui-monospace,SFMono-Regular,Consolas,monospace;user-select:none;z-index:2;';
document.body.appendChild(panel);

const readout = document.createElement('div');
readout.style.cssText =
  'background:#080910cc;border:1px solid #23262e;border-radius:6px;padding:9px 10px;' +
  'backdrop-filter:blur(6px);line-height:1.55;min-width:286px;max-width:360px;';
panel.appendChild(readout);

const controls = document.createElement('div');
controls.style.cssText =
  'background:#080910cc;border:1px solid #23262e;border-radius:6px;padding:8px 10px;' +
  'display:grid;gap:8px;backdrop-filter:blur(6px);';
panel.appendChild(controls);

const diskToggle = labeledCheckbox('accretion disk', true);
controls.appendChild(diskToggle.label);

const qualitySelect = document.createElement('select');
qualitySelect.style.cssText =
  'background:#171a22;color:#e6eaf4;border:1px solid #3b4150;border-radius:5px;padding:5px;font:12px ui-monospace,monospace;';
[
  ['interactive', '320x180'],
  ['balanced', '480x270'],
  ['detail', '640x360'],
  ['native window', 'native'],
].forEach(([label, value]) => {
  const option = document.createElement('option');
  option.textContent = label;
  option.value = value;
  if (value === '320x180') option.selected = true;
  qualitySelect.appendChild(option);
});
controls.appendChild(labeledControl('render', qualitySelect));

const mapBox = document.createElement('div');
mapBox.style.cssText =
  'background:#080910cc;border:1px solid #23262e;border-radius:6px;overflow:hidden;' +
  'box-shadow:0 10px 30px #0008;';
panel.appendChild(mapBox);

let state: FallState = launchFromLocal({ r: 12, phi: 0, betaRadial: 0, betaTangential: 0.08 });
let lastRenderableState: FallState = state;
let running = true;
let crossed = false;
let singularityReached = false;
const camera = createPlayerCamera(-1, 0);
let plannedVector: MapVector = { x: -1, z: 0, pixels: 0 };
let lastFrame = performance.now() / 1000;
let looking = false;
let lastLookX = 0;
let lastLookY = 0;
let renderInFlight = false;
let pendingRender = false;
let renderMessage = 'WebGPU renderer warming up';
let lastRenderRequest = 0;

const minimap = createMinimap(mapBox, {
  onPlanStart(position) {
    running = false;
    crossed = false;
    singularityReached = false;
    state = launchFromLocal({ r: position.r, phi: Math.atan2(position.z, position.x), betaRadial: 0, betaTangential: 0 });
    lastRenderableState = state;
    const inward = inwardVector(position);
    setCameraLookDirection(camera, inward.x, inward.z);
    plannedVector = { x: inward.x, z: inward.z, pixels: 0 };
    updateMap();
    requestRender();
  },
  onPlanMove(position, vector) {
    plannedVector = vector.pixels < 8 ? { ...inwardVector(position), pixels: vector.pixels } : vector;
    const dir = normalize2(plannedVector.x, plannedVector.z);
    setCameraLookDirection(camera, dir.x, dir.z);
    state = plannedState(position, plannedVector);
    lastRenderableState = state;
    running = false;
    updateMap();
    requestRender();
  },
  onPlanCommit(position, vector) {
    plannedVector = vector.pixels < 8 ? { ...inwardVector(position), pixels: vector.pixels } : vector;
    const dir = normalize2(plannedVector.x, plannedVector.z);
    setCameraLookDirection(camera, dir.x, dir.z);
    state = plannedState(position, plannedVector);
    lastRenderableState = state;
    running = true;
    crossed = false;
    singularityReached = false;
    updateMap();
    requestRender();
  },
});

canvas.addEventListener('pointerdown', (event) => {
  looking = true;
  lastLookX = event.clientX;
  lastLookY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
  canvas.style.cursor = 'grabbing';
});
canvas.addEventListener('pointermove', (event) => {
  if (!looking) return;
  const dx = event.clientX - lastLookX;
  const dy = event.clientY - lastLookY;
  rotatePlayerCamera(camera, -dx * 0.004, -dy * 0.004);
  lastLookX = event.clientX;
  lastLookY = event.clientY;
  updateMap();
  requestRender();
});
function stopLooking(event?: PointerEvent) {
  looking = false;
  if (event && canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
  canvas.style.cursor = 'grab';
}
canvas.addEventListener('pointerup', stopLooking);
canvas.addEventListener('pointercancel', stopLooking);
canvas.addEventListener('lostpointercapture', () => {
  looking = false;
  canvas.style.cursor = 'grab';
});

diskToggle.input.addEventListener('change', () => requestRender());
qualitySelect.addEventListener('change', () => requestRender());
window.addEventListener('resize', () => requestRender());

updateMap();
requestRender();
requestAnimationFrame(frame);

function frame(nowMs: number) {
  const now = nowMs / 1000;
  const dt = Math.min(now - lastFrame, 0.035);
  lastFrame = now;

  if (running && !singularityReached) {
    state = stepFall(state, dt * 0.82);
    crossed = isHorizonCrossed(state);
    singularityReached = isSingularityReached(state);
    if (singularityReached) {
      running = false;
    } else {
      lastRenderableState = state;
    }
    updateMap();
    if (!singularityReached && now - lastRenderRequest > 0.1) {
      requestRender();
    }
  }

  renderReadout();
  requestAnimationFrame(frame);
}

function requestRender(): void {
  const now = performance.now() / 1000;
  if (now - lastRenderRequest < 0.08) return;
  lastRenderRequest = now;
  if (renderInFlight) {
    pendingRender = true;
    return;
  }
  renderInFlight = true;
  pendingRender = false;
  void renderFallView().finally(() => {
    renderInFlight = false;
    if (pendingRender) requestRender();
  });
}

async function renderFallView(): Promise<void> {
  if (singularityReached) {
    renderMessage = 'singularity cutoff reached; rendering paused';
    return;
  }
  const options = createFallRenderOptions(lastRenderableState);
  const result = await renderWebGpuCompositeFromCameraToCanvas(options, canvas);
  renderMessage = result.message;
}

function createFallRenderOptions(renderState: FallState): CompositeCameraSampleOptions {
  const { width, height } = selectedRenderSize();
  const observerFrame = observerFrameFromState(renderState, camera);
  return {
    width,
    height,
    position: spatialPositionFromState(renderState),
    tetrad: observerFrame.tetrad,
    observerVelocity: observerFrame.fourVelocity,
    params: FALL_PARAMS,
    verticalFovRadians: 72 * Math.PI / 180,
    traceOptions: {
      stepSize: renderState.r < HORIZON_RADIUS ? 0.018 : 0.026,
      maxSteps: renderState.r < HORIZON_RADIUS ? 820 : 260,
      escapeRadius: 38,
      singularityRadius: SINGULARITY_CUTOFF,
    },
    disk: {
      innerRadius: 1.65,
      outerRadius: 9.5,
    },
    radianceModel: {
      innerRadius: 1.65,
      outerRadius: 9.5,
      innerTemperature: 7800,
      emissivityScale: diskToggle.input.checked ? 0.68 : 0,
      boostPower: 3.2,
      spinDirection: 1,
      emissionPhase: renderState.t * 0.11,
    },
  };
}

function selectedRenderSize(): { width: number; height: number } {
  if (qualitySelect.value === 'native') {
    return {
      width: Math.max(1, Math.round(window.innerWidth)),
      height: Math.max(1, Math.round(window.innerHeight)),
    };
  }
  const [width, height] = qualitySelect.value.split('x').map((value) => Number(value));
  return { width, height };
}

function plannedState(position: MapPosition, vector: MapVector): FallState {
  const phi = Math.atan2(position.z, position.x);
  if (vector.pixels < 8) {
    return launchFromLocal({ r: position.r, phi, betaRadial: 0, betaTangential: 0 });
  }
  const dir = normalize2(vector.x, vector.z);
  const speed = Math.max(0.05, Math.min(0.85, vector.pixels / 160));
  const er = { x: Math.cos(phi), z: Math.sin(phi) };
  const ep = { x: -Math.sin(phi), z: Math.cos(phi) };
  return launchFromLocal({
    r: position.r,
    phi,
    betaRadial: speed * (dir.x * er.x + dir.z * er.z),
    betaTangential: speed * (dir.x * ep.x + dir.z * ep.z),
  });
}

function updateMap() {
  const pos = positionFromState(state);
  const flatForward = cameraFlatForward(camera);
  minimap.setState({
    position: pos,
    vector: plannedVector.pixels > 0 ? plannedVector : { x: flatForward.x, z: flatForward.z, pixels: 0 },
    preview: previewFall(state, 38),
    running,
    crossed,
  });
}

function inwardVector(position: MapPosition): MapVector {
  const dir = normalize2(-position.x, -position.z);
  return { x: dir.x, z: dir.z, pixels: 0 };
}

function normalize2(x: number, z: number): { x: number; z: number } {
  const len = Math.hypot(x, z) || 1;
  return { x: x / len, z: z / len };
}

function renderReadout() {
  const status = singularityReached
    ? 'singularity'
    : crossed
      ? 'inside Kerr horizon'
      : running ? 'falling' : 'planning';
  if (singularityReached) {
    readout.innerHTML =
      `<div>status: <span style="color:#e8b873">${status}</span></div>` +
      `<div>r: ${state.r.toFixed(3)} | cutoff: ${SINGULARITY_CUTOFF.toFixed(3)}</div>` +
      `<div>proper time: ${state.tau.toFixed(2)}</div>` +
      `<div>Kerr-Schild time: ${formatTime(state.t)}</div>` +
      `<div>H + 1/2: ${timelikeResidual(state).toExponential(1)}</div>` +
      `<div style="color:#7d8290;margin-top:6px">${renderMessage}</div>` +
      `<div style="color:#7d8290;margin-top:6px">minimap: choose a new start to run again</div>`;
    return;
  }

  const observerFrame = observerFrameFromState(state, camera);
  const residual = tetradResidual(observerFrame.tetrad, state);
  readout.innerHTML =
    `<div>status: <span style="color:#e8b873">${status}</span></div>` +
    `<div>r: ${state.r.toFixed(3)} | horizon: ${HORIZON_RADIUS.toFixed(3)}</div>` +
    `<div>proper time: ${state.tau.toFixed(2)}</div>` +
    `<div>Kerr-Schild time: ${formatTime(state.t)}</div>` +
    `<div>local speed: ${(observerFrame.speed * 100).toFixed(1)}% c</div>` +
    `<div>H + 1/2: ${timelikeResidual(state).toExponential(1)}</div>` +
    `<div>tetrad residual: ${residual.toExponential(1)}</div>` +
    `<div style="color:#7d8290;margin-top:6px">${renderMessage}</div>` +
    `<div style="color:#7d8290;margin-top:6px">main canvas: drag to look</div>` +
    `<div style="color:#7d8290">minimap: down = place, drag = aim, release = fall</div>`;
}

function formatTime(t: number): string {
  if (Math.abs(t) > 9999) return t.toExponential(2);
  return t.toFixed(2);
}

function labeledCheckbox(text: string, checked: boolean): { label: HTMLLabelElement; input: HTMLInputElement } {
  const label = document.createElement('label');
  label.style.cssText = 'display:flex;gap:8px;align-items:center;';
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = checked;
  input.style.cssText = 'accent-color:#e8b873;';
  const span = document.createElement('span');
  span.textContent = text;
  label.append(input, span);
  return { label, input };
}

function labeledControl(text: string, control: HTMLElement): HTMLElement {
  const label = document.createElement('label');
  label.style.cssText = 'display:flex;gap:8px;align-items:center;justify-content:space-between;';
  const span = document.createElement('span');
  span.textContent = text;
  label.append(span, control);
  return label;
}
