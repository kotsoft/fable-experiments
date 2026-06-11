// Black hole renderer entry point — WebGL2 fullscreen shader, no React.
import { advanceAutoOrbit, createOrbitControls } from '../common/orbitControls';
import { createFullscreenCanvas, createProgram, resizeCanvasToDisplaySize, uniforms } from '../common/webgl';
import { FRAG_SRC, VERT_SRC } from './shaders';

const canvas = createFullscreenCanvas('grab');
const maybeGl = canvas.getContext('webgl2');
if (!maybeGl) throw new Error('WebGL2 not supported');
const gl: WebGL2RenderingContext = maybeGl;

const prog = createProgram(gl, VERT_SRC, FRAG_SRC);
const U = uniforms(gl, prog);
const uRes = U('uRes');
const uTime = U('uTime');
const uCamPos = U('uCamPos');
const uCamFwd = U('uCamFwd');
const uCamRight = U('uCamRight');
const uCamUp = U('uCamUp');

const orbit = createOrbitControls(canvas, {
  yaw: 0.6,
  pitch: 0.10,
  dist: 17,
  minDist: 4,
  maxDist: 32,
});

window.addEventListener('resize', () => resizeCanvasToDisplaySize(gl, canvas, 1.5));

function frame(nowMs: number) {
  const t = nowMs / 1000;
  resizeCanvasToDisplaySize(gl, canvas, 1.5);
  advanceAutoOrbit(orbit);

  const cp = Math.cos(orbit.pitch), sp = Math.sin(orbit.pitch);
  const cy = Math.cos(orbit.yaw), sy = Math.sin(orbit.yaw);
  const px = orbit.dist * cp * sy, py = orbit.dist * sp, pz = orbit.dist * cp * cy;

  const fl = Math.hypot(px, py, pz);
  const fx = -px / fl, fy = -py / fl, fz = -pz / fl;
  let rx = -fz, rz = fx;
  const ry = 0;
  const rl = Math.hypot(rx, ry, rz) || 1;
  rx /= rl; rz /= rl;
  const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;

  gl.uniform2f(uRes, canvas.width, canvas.height);
  gl.uniform1f(uTime, t);
  gl.uniform3f(uCamPos, px, py, pz);
  gl.uniform3f(uCamFwd, fx, fy, fz);
  gl.uniform3f(uCamRight, rx, ry, rz);
  gl.uniform3f(uCamUp, ux, uy, uz);

  gl.drawArrays(gl.TRIANGLES, 0, 3);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
