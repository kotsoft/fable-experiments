import { HORIZON_RADIUS, clampRadius, type PreviewPoint } from './physics';

export interface MapPosition {
  x: number;
  z: number;
  r: number;
}

export interface MapVector {
  x: number;
  z: number;
  pixels: number;
}

interface MinimapCallbacks {
  onPlanStart(position: MapPosition): void;
  onPlanMove(position: MapPosition, vector: MapVector): void;
  onPlanCommit(position: MapPosition, vector: MapVector): void;
}

interface MinimapState {
  position: MapPosition;
  vector: MapVector;
  preview: PreviewPoint[];
  running: boolean;
  crossed: boolean;
}

export interface MinimapApi {
  setState(state: MinimapState): void;
}

const SIZE = 280;
const PAD = 18;
const MAP_R = 18;
const SCALE = (SIZE - PAD * 2) / (MAP_R * 2);
const PHOTON_RING_R = 1.5;
const DISK_INNER_R = 1.65;

export function createMinimap(container: HTMLElement, callbacks: MinimapCallbacks): MinimapApi {
  const canvas = document.createElement('canvas');
  canvas.width = SIZE;
  canvas.height = SIZE;
  canvas.style.cssText =
    'width:280px;height:280px;display:block;touch-action:none;cursor:crosshair;';
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas not supported');

  let state: MinimapState = {
    position: { x: 12, z: 0, r: 12 },
    vector: { x: -1, z: 0, pixels: 0 },
    preview: [],
    running: false,
    crossed: false,
  };
  let planning = false;
  let planPosition = state.position;

  const draw = () => {
    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.fillStyle = '#07080c';
    ctx.fillRect(0, 0, SIZE, SIZE);
    drawGrid(ctx);
    drawRing(ctx, DISK_INNER_R, '#4a3f2e', [5, 4]);
    drawRing(ctx, PHOTON_RING_R, '#4d5361', [4, 5]);
    drawRing(ctx, HORIZON_RADIUS, '#000', []);

    if (state.preview.length > 1) {
      ctx.beginPath();
      state.preview.forEach((p, i) => {
        const s = toScreen(p.x, p.z);
        if (i === 0) ctx.moveTo(s.x, s.y);
        else ctx.lineTo(s.x, s.y);
      });
      ctx.strokeStyle = '#e8b873';
      ctx.globalAlpha = 0.8;
      ctx.lineWidth = 1.8;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    const pos = toScreen(state.position.x, state.position.z);
    const len = Math.hypot(state.vector.x, state.vector.z) || 1;
    const vx = state.vector.x / len;
    const vz = state.vector.z / len;
    drawArrow(ctx, pos.x, pos.y, vx, -vz);
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, 5.5, 0, Math.PI * 2);
    ctx.fillStyle = state.crossed ? '#e06050' : state.running ? '#ffffff' : '#e8b873';
    ctx.fill();

    ctx.fillStyle = '#7d8290';
    ctx.font = '11px ui-monospace, SFMono-Regular, Consolas, monospace';
    ctx.fillText('horizon', SIZE / 2 + 16, SIZE / 2 + 4);
    ctx.fillText('photon', SIZE / 2 + 24, SIZE / 2 - 18);
    ctx.fillText('disk in', SIZE / 2 + 44, SIZE / 2 - 43);
  };

  const pointerPosition = (event: PointerEvent): MapPosition => {
    const rect = canvas.getBoundingClientRect();
    const px = (event.clientX - rect.left) * (canvas.width / rect.width);
    const py = (event.clientY - rect.top) * (canvas.height / rect.height);
    const wx = (px - SIZE / 2) / SCALE;
    const wz = -(py - SIZE / 2) / SCALE;
    const r = Math.hypot(wx, wz);
    const clampedR = clampRadius(Math.min(MAP_R, r || HORIZON_RADIUS + 0.08));
    const s = clampedR / (r || 1);
    return { x: wx * s, z: wz * s, r: clampedR };
  };

  const pointerVector = (event: PointerEvent): MapVector => {
    const rect = canvas.getBoundingClientRect();
    const px = (event.clientX - rect.left) * (canvas.width / rect.width);
    const py = (event.clientY - rect.top) * (canvas.height / rect.height);
    const p0 = toScreen(planPosition.x, planPosition.z);
    const dx = px - p0.x;
    const dy = py - p0.y;
    return {
      x: dx / SCALE,
      z: -dy / SCALE,
      pixels: Math.hypot(dx, dy),
    };
  };

  canvas.addEventListener('pointerdown', (event) => {
    planning = true;
    canvas.setPointerCapture(event.pointerId);
    planPosition = pointerPosition(event);
    callbacks.onPlanStart(planPosition);
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!planning) return;
    callbacks.onPlanMove(planPosition, pointerVector(event));
  });
  const commit = (event: PointerEvent) => {
    if (!planning) return;
    planning = false;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    callbacks.onPlanCommit(planPosition, pointerVector(event));
  };
  canvas.addEventListener('pointerup', commit);
  canvas.addEventListener('pointercancel', commit);

  draw();
  return {
    setState(next) {
      state = next;
      draw();
    },
  };
}

function toScreen(x: number, z: number): { x: number; y: number } {
  return {
    x: SIZE / 2 + x * SCALE,
    y: SIZE / 2 - z * SCALE,
  };
}

function drawGrid(ctx: CanvasRenderingContext2D) {
  ctx.strokeStyle = '#181b23';
  ctx.lineWidth = 1;
  for (let r = 6; r <= 18; r += 6) drawRing(ctx, r, '#181b23', []);
  ctx.beginPath();
  ctx.moveTo(PAD, SIZE / 2);
  ctx.lineTo(SIZE - PAD, SIZE / 2);
  ctx.moveTo(SIZE / 2, PAD);
  ctx.lineTo(SIZE / 2, SIZE - PAD);
  ctx.stroke();
}

function drawRing(ctx: CanvasRenderingContext2D, r: number, color: string, dash: number[]) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(SIZE / 2, SIZE / 2, r * SCALE, 0, Math.PI * 2);
  ctx.setLineDash(dash);
  ctx.strokeStyle = color;
  ctx.lineWidth = r === 1 ? 2 : 1;
  if (r === 1) {
    ctx.fillStyle = '#000';
    ctx.fill();
  }
  ctx.stroke();
  ctx.restore();
}

function drawArrow(ctx: CanvasRenderingContext2D, x: number, y: number, vx: number, vy: number) {
  const length = 34;
  const tx = x + vx * length;
  const ty = y + vy * length;
  ctx.strokeStyle = '#ffffff';
  ctx.fillStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(tx, ty);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(tx + vx * 8, ty + vy * 8);
  ctx.lineTo(tx - vx * 7 - vy * 5, ty - vy * 7 + vx * 5);
  ctx.lineTo(tx - vx * 7 + vy * 5, ty - vy * 7 - vx * 5);
  ctx.closePath();
  ctx.fill();
}
