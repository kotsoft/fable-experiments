// Kerr (rotating) black hole renderer entry point — WebGL2, no React.
// Spin axis is +z; the camera orbits with z as "up".
import { VERT_SRC, FRAG_SRC } from './shaders';

document.body.style.margin = '0';
document.body.style.background = '#000';
document.body.style.overflow = 'hidden';

const canvas = document.createElement('canvas');
canvas.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;display:block;cursor:grab;';
document.body.appendChild(canvas);

// minimal spin control — the one physics parameter of this demo
const ui = document.createElement('label');
ui.style.cssText =
  'position:fixed;left:14px;bottom:12px;color:#9aa0ab;font:13px monospace;' +
  'display:flex;gap:10px;align-items:center;opacity:0.75;user-select:none;';
const spinInput = document.createElement('input');
spinInput.type = 'range';
spinInput.min = '0';
spinInput.max = '0.998';
spinInput.step = '0.002';
spinInput.value = '0.95';
spinInput.style.cssText = 'width:150px;accent-color:#e8b873;';
const spinText = document.createElement('span');
ui.appendChild(spinInput);
ui.appendChild(spinText);
document.body.appendChild(ui);

const gl = canvas.getContext('webgl2');
if (!gl) throw new Error('WebGL2 not supported');

function compile(type: number, src: string): WebGLShader {
  const sh = gl!.createShader(type)!;
  gl!.shaderSource(sh, src);
  gl!.compileShader(sh);
  if (!gl!.getShaderParameter(sh, gl!.COMPILE_STATUS)) {
    throw new Error(gl!.getShaderInfoLog(sh) ?? 'shader compile failed');
  }
  return sh;
}

const prog = gl.createProgram()!;
gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT_SRC));
gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG_SRC));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
  throw new Error(gl.getProgramInfoLog(prog) ?? 'program link failed');
}
gl.useProgram(prog);

const U = (n: string) => gl!.getUniformLocation(prog, n);
const uRes = U('uRes'), uTime = U('uTime');
const uCamPos = U('uCamPos'), uCamFwd = U('uCamFwd');
const uCamRight = U('uCamRight'), uCamUp = U('uCamUp');
const uSpin = U('uSpin'), uHorizon = U('uHorizon'), uIsco = U('uIsco');

// prograde ISCO (Bardeen); same formula validated in scripts/validate-kerr.mjs
function iscoRadius(a: number): number {
  const z1 = 1 + Math.cbrt(1 - a * a) * (Math.cbrt(1 + a) + Math.cbrt(1 - a));
  const z2 = Math.sqrt(3 * a * a + z1 * z1);
  return 3 + z2 - Math.sqrt((3 - z1) * (3 + z1 + 2 * z2));
}

function setSpin(a: number) {
  gl!.uniform1f(uSpin, a);
  gl!.uniform1f(uHorizon, 1 + Math.sqrt(Math.max(1 - a * a, 0)));
  gl!.uniform1f(uIsco, iscoRadius(a));
  spinText.textContent = `spin a = ${a.toFixed(3)}`;
}
spinInput.addEventListener('input', () => setSpin(parseFloat(spinInput.value)));
setSpin(0.95);

// --- camera orbit state (units of M; note rs = 2M here) ---
let yaw = 0.6;
let pitch = 0.10;
let dist = 30;
let dragging = false;
let lastX = 0, lastY = 0;
let lastInteract = -10;

canvas.addEventListener('pointerdown', (e) => {
  dragging = true;
  lastX = e.clientX; lastY = e.clientY;
  canvas.setPointerCapture(e.pointerId);
  canvas.style.cursor = 'grabbing';
});
canvas.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  yaw -= (e.clientX - lastX) * 0.005;
  pitch += (e.clientY - lastY) * 0.005;
  pitch = Math.max(-1.45, Math.min(1.45, pitch));
  lastX = e.clientX; lastY = e.clientY;
  lastInteract = performance.now() / 1000;
});
canvas.addEventListener('pointerup', (e) => {
  dragging = false;
  canvas.releasePointerCapture(e.pointerId);
  canvas.style.cursor = 'grab';
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  dist *= Math.exp(e.deltaY * 0.001);
  dist = Math.max(7, Math.min(60, dist));
  lastInteract = performance.now() / 1000;
}, { passive: false });

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 1.0);
  const w = Math.round(canvas.clientWidth * dpr);
  const h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    gl!.viewport(0, 0, w, h);
  }
}
window.addEventListener('resize', resize);

function frame(nowMs: number) {
  const t = nowMs / 1000;
  resize();
  if (!dragging && t - lastInteract > 4) yaw += 0.00045;

  // z-up look-at(origin) basis
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const px = dist * cp * cy, py = dist * cp * sy, pz = dist * sp;
  const fl = Math.hypot(px, py, pz);
  const fx = -px / fl, fy = -py / fl, fz = -pz / fl;
  // right = normalize(cross(fwd, z-up)) = (fy, -fx, 0)/|.|
  let rx = fy, ry = -fx;
  const rl = Math.hypot(rx, ry) || 1;
  rx /= rl; ry /= rl;
  // up = cross(right, fwd)
  const ux = ry * fz, uy = -rx * fz, uz = rx * fy - ry * fx;

  gl!.uniform2f(uRes, canvas.width, canvas.height);
  gl!.uniform1f(uTime, t);
  gl!.uniform3f(uCamPos, px, py, pz);
  gl!.uniform3f(uCamFwd, fx, fy, fz);
  gl!.uniform3f(uCamRight, rx, ry, 0);
  gl!.uniform3f(uCamUp, ux, uy, uz);

  gl!.drawArrays(gl!.TRIANGLES, 0, 3);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
