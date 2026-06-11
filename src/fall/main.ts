import { createFullscreenCanvas, createProgram, resizeCanvasToDisplaySize, uniforms } from '../common/webgl';
import { createMinimap, type MapPosition, type MapVector } from './minimap';
import {
  isHorizonCrossed,
  isSingularityReached,
  launchFromLocal,
  positionFromState,
  previewFall,
  stepFall,
  type FallState,
} from './physics';
import { FRAG_SRC, VERT_SRC } from './shaders';
import {
  cameraFlatForward,
  createPlayerCamera,
  observerFrameFromState,
  rotatePlayerCamera,
  setCameraLookDirection,
  tetradResidual,
  type Vec4,
} from './tetrad';

const canvas = createFullscreenCanvas('grab');
const maybeGl = canvas.getContext('webgl2');
if (!maybeGl) throw new Error('WebGL2 not supported');
const gl: WebGL2RenderingContext = maybeGl;

const prog = createProgram(gl, VERT_SRC, FRAG_SRC);
const U = uniforms(gl, prog);
const uRes = U('uRes');
const uTime = U('uTime');
const uObserverPos = U('uObserverPos');
const uTetradTime = U('uTetradTime');
const uTetradRight = U('uTetradRight');
const uTetradUp = U('uTetradUp');
const uTetradForward = U('uTetradForward');
const uObserverBeta = U('uObserverBeta');
const uInterior = U('uInterior');
const uSingularityFade = U('uSingularityFade');
const uDiskEnabled = U('uDiskEnabled');

const panel = document.createElement('section');
panel.style.cssText =
  'position:fixed;right:14px;bottom:14px;display:grid;gap:8px;color:#c9cdd6;' +
  'font:12px ui-monospace,SFMono-Regular,Consolas,monospace;user-select:none;';
document.body.appendChild(panel);

const readout = document.createElement('div');
readout.style.cssText =
  'background:#080910cc;border:1px solid #23262e;border-radius:6px;padding:9px 10px;' +
  'backdrop-filter:blur(6px);line-height:1.55;min-width:258px;';
panel.appendChild(readout);

const controls = document.createElement('label');
controls.style.cssText =
  'background:#080910cc;border:1px solid #23262e;border-radius:6px;padding:8px 10px;' +
  'display:flex;gap:8px;align-items:center;backdrop-filter:blur(6px);';
const diskToggle = document.createElement('input');
diskToggle.type = 'checkbox';
diskToggle.checked = true;
diskToggle.style.cssText = 'accent-color:#e8b873;';
const diskText = document.createElement('span');
diskText.textContent = 'accretion disk';
controls.appendChild(diskToggle);
controls.appendChild(diskText);
panel.appendChild(controls);

const mapBox = document.createElement('div');
mapBox.style.cssText =
  'background:#080910cc;border:1px solid #23262e;border-radius:6px;overflow:hidden;' +
  'box-shadow:0 10px 30px #0008;';
panel.appendChild(mapBox);

let state: FallState = launchFromLocal({ r: 12, phi: 0, betaRadial: 0, betaTangential: 0 });
let running = true;
let crossed = false;
let singularityReached = false;
const camera = createPlayerCamera(-1, 0);
let plannedVector: MapVector = { x: -1, z: 0, pixels: 0 };
let lastFrame = performance.now() / 1000;
let looking = false;
let lastLookX = 0;
let lastLookY = 0;

const minimap = createMinimap(mapBox, {
  onPlanStart(position) {
    running = false;
    crossed = false;
    singularityReached = false;
    state = launchFromLocal({ r: position.r, phi: Math.atan2(position.z, position.x), betaRadial: 0, betaTangential: 0 });
    const inward = inwardVector(position);
    setCameraLookDirection(camera, inward.x, inward.z);
    plannedVector = { x: inward.x, z: inward.z, pixels: 0 };
    updateMap();
  },
  onPlanMove(position, vector) {
    plannedVector = vector.pixels < 8 ? { ...inwardVector(position), pixels: vector.pixels } : vector;
    const dir = normalize2(plannedVector.x, plannedVector.z);
    setCameraLookDirection(camera, dir.x, dir.z);
    state = plannedState(position, plannedVector);
    running = false;
    updateMap();
  },
  onPlanCommit(position, vector) {
    plannedVector = vector.pixels < 8 ? { ...inwardVector(position), pixels: vector.pixels } : vector;
    const dir = normalize2(plannedVector.x, plannedVector.z);
    setCameraLookDirection(camera, dir.x, dir.z);
    state = plannedState(position, plannedVector);
    running = true;
    crossed = false;
    singularityReached = false;
    updateMap();
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

window.addEventListener('resize', () => resizeCanvasToDisplaySize(gl, canvas, 1.25));
updateMap();

function frame(nowMs: number) {
  const now = nowMs / 1000;
  const dt = Math.min(now - lastFrame, 0.04);
  lastFrame = now;
  resizeCanvasToDisplaySize(gl, canvas, 1.25);

  if (running && !singularityReached) {
    const tauRate = state.r < 1 ? 0.24 : 0.72;
    state = stepFall(state, dt * tauRate);
    crossed = isHorizonCrossed(state);
    singularityReached = isSingularityReached(state);
    if (singularityReached) running = false;
    updateMap();
  }

  const pos = positionFromState(state);
  const observerFrame = observerFrameFromState(state, camera);
  const interior = Math.min(1, Math.max(0, (1.0 - state.r) / 0.08));
  const singularityFade = Math.min(1, Math.max(0, (0.16 - state.r) / 0.08));

  gl.uniform2f(uRes, canvas.width, canvas.height);
  gl.uniform1f(uTime, now);
  gl.uniform3f(uObserverPos, pos.x, 0, pos.z);
  uniformVec4(uTetradTime, observerFrame.tetrad.eTime);
  uniformVec4(uTetradRight, observerFrame.tetrad.eRight);
  uniformVec4(uTetradUp, observerFrame.tetrad.eUp);
  uniformVec4(uTetradForward, observerFrame.tetrad.eForward);
  gl.uniform3f(uObserverBeta, observerFrame.beta.x, observerFrame.beta.y, observerFrame.beta.z);
  gl.uniform1f(uInterior, interior);
  gl.uniform1f(uSingularityFade, singularityFade);
  gl.uniform1f(uDiskEnabled, diskToggle.checked ? 1 : 0);
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  renderReadout(observerFrame.speed, tetradResidual(observerFrame.tetrad));
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

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
    preview: previewFall(state, 45),
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

function uniformVec4(location: WebGLUniformLocation | null, v: Vec4) {
  gl.uniform4f(location, v.t, v.x, v.y, v.z);
}

function renderReadout(speed: number, frameResidual: number) {
  const status = singularityReached
    ? 'singularity'
    : crossed
      ? 'inside horizon'
      : running ? 'falling' : 'planning';
  readout.innerHTML =
    `<div>status: <span style="color:#e8b873">${status}</span></div>` +
    `<div>r: ${state.r.toFixed(3)} rs</div>` +
    `<div>proper time: ${state.tau.toFixed(2)}</div>` +
    `<div>distant time: ${crossed ? 'unreachable' : formatDistantTime(state.t)}</div>` +
    `<div>local speed: ${(speed * 100).toFixed(1)}% c</div>` +
    `<div>tetrad residual: ${frameResidual.toExponential(1)}</div>` +
    `<div style="color:#7d8290;margin-top:6px">main canvas: drag to look</div>` +
    `<div style="color:#7d8290">minimap: down = place, drag = aim, release = fall</div>`;
}

function formatDistantTime(t: number): string {
  if (t > 9999) return t.toExponential(2);
  return t.toFixed(2);
}
