// Kerr (rotating) black hole renderer entry point — WebGL2, no React.
// Spin axis is +z; the camera orbits with z as "up".
import { advanceAutoOrbit, createOrbitControls } from '../common/orbitControls';
import { createFullscreenCanvas, createProgram, resizeCanvasToDisplaySize, uniforms } from '../common/webgl';
import { FRAG_SRC, VERT_SRC } from './shaders';

const canvas = createFullscreenCanvas('grab');

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

const maybeGl = canvas.getContext('webgl2');
if (!maybeGl) throw new Error('WebGL2 not supported');
const gl: WebGL2RenderingContext = maybeGl;

const prog = createProgram(gl, VERT_SRC, FRAG_SRC);
const U = uniforms(gl, prog);
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
  gl.uniform1f(uSpin, a);
  gl.uniform1f(uHorizon, 1 + Math.sqrt(Math.max(1 - a * a, 0)));
  gl.uniform1f(uIsco, iscoRadius(a));
  spinText.textContent = `spin a = ${a.toFixed(3)}`;
}
spinInput.addEventListener('input', () => setSpin(parseFloat(spinInput.value)));
setSpin(0.95);

const orbit = createOrbitControls(canvas, {
  yaw: 0.6,
  pitch: 0.10,
  dist: 30,
  minDist: 7,
  maxDist: 60,
});

window.addEventListener('resize', () => resizeCanvasToDisplaySize(gl, canvas, 1.0));

function frame(nowMs: number) {
  const t = nowMs / 1000;
  resizeCanvasToDisplaySize(gl, canvas, 1.0);
  advanceAutoOrbit(orbit);

  const cp = Math.cos(orbit.pitch), sp = Math.sin(orbit.pitch);
  const cy = Math.cos(orbit.yaw), sy = Math.sin(orbit.yaw);
  const px = orbit.dist * cp * cy, py = orbit.dist * cp * sy, pz = orbit.dist * sp;
  const fl = Math.hypot(px, py, pz);
  const fx = -px / fl, fy = -py / fl, fz = -pz / fl;
  let rx = fy, ry = -fx;
  const rl = Math.hypot(rx, ry) || 1;
  rx /= rl; ry /= rl;
  const ux = ry * fz, uy = -rx * fz, uz = rx * fy - ry * fx;

  gl.uniform2f(uRes, canvas.width, canvas.height);
  gl.uniform1f(uTime, t);
  gl.uniform3f(uCamPos, px, py, pz);
  gl.uniform3f(uCamFwd, fx, fy, fz);
  gl.uniform3f(uCamRight, rx, ry, 0);
  gl.uniform3f(uCamUp, ux, uy, uz);

  gl.drawArrays(gl.TRIANGLES, 0, 3);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
