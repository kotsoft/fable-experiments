// Black hole renderer entry point — WebGL2 fullscreen shader, no React.
import { VERT_SRC, FRAG_SRC } from './shaders';

document.body.style.margin = '0';
document.body.style.background = '#000';
document.body.style.overflow = 'hidden';

const canvas = document.createElement('canvas');
canvas.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;display:block;cursor:grab;';
document.body.appendChild(canvas);

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

const uRes = gl.getUniformLocation(prog, 'uRes');
const uTime = gl.getUniformLocation(prog, 'uTime');
const uCamPos = gl.getUniformLocation(prog, 'uCamPos');
const uCamFwd = gl.getUniformLocation(prog, 'uCamFwd');
const uCamRight = gl.getUniformLocation(prog, 'uCamRight');
const uCamUp = gl.getUniformLocation(prog, 'uCamUp');

// --- camera orbit state (units of rs) ---
let yaw = 0.6;
let pitch = 0.10;      // just above the disk plane, like the Interstellar shots
let dist = 17;
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
  dist = Math.max(4, Math.min(32, dist));
  lastInteract = performance.now() / 1000;
}, { passive: false });

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
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

  // slow auto-orbit when idle
  if (!dragging && t - lastInteract > 4) yaw += 0.00045;

  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const px = dist * cp * sy, py = dist * sp, pz = dist * cp * cy;

  // look-at(origin) basis
  const fl = Math.hypot(px, py, pz);
  const fx = -px / fl, fy = -py / fl, fz = -pz / fl;
  // right = normalize(cross(fwd, worldUp))
  let rx = -fz, rz = fx;
  const ry = 0;
  const rl = Math.hypot(rx, ry, rz) || 1;
  rx /= rl; rz /= rl;
  // up = cross(right, fwd)
  const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;

  gl!.uniform2f(uRes, canvas.width, canvas.height);
  gl!.uniform1f(uTime, t);
  gl!.uniform3f(uCamPos, px, py, pz);
  gl!.uniform3f(uCamFwd, fx, fy, fz);
  gl!.uniform3f(uCamRight, rx, ry, rz);
  gl!.uniform3f(uCamUp, ux, uy, uz);

  gl!.drawArrays(gl!.TRIANGLES, 0, 3);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
