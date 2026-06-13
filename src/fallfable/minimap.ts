// Trajectory planner: a top-down equatorial map of the Kerr geometry. Press to
// place the player, drag to set the local launch velocity, release to fall.
// Because launches are measured against the Eulerian observer of the
// Kerr-Schild slicing, the planner works at any radius - including inside the
// ergosphere and even inside the horizon.

import { equatorialKsRadius, equatorialRho } from './kerr';
import {
  DISK_INNER,
  DISK_OUTER,
  ERGOSPHERE,
  HORIZON,
  ISCO,
  MAP_RADIUS,
  PARAMS,
  PHOTON_PROGRADE,
  PHOTON_RETROGRADE,
  launchLocal,
  type PlayerState,
  type PreviewPoint,
} from './player';

export interface PlannerCallbacks {
  onPreview(state: PlayerState, launchHeading: PreviewPoint | null): void;
  onCommit(state: PlayerState, launchHeading: PreviewPoint | null): void;
  /** Height above the equatorial plane applied to launches. */
  launchHeight?: () => number;
}

export interface PlannerView {
  state: PlayerState;
  preview: PreviewPoint[];
  running: boolean;
  /** Direction the camera looks, drawn when there is no drag in progress. */
  lookDirection: { x: number; y: number };
}

export interface PlannerApi {
  draw(view: PlannerView): void;
}

const SIZE = 300;
const CENTER = SIZE / 2;
const SCALE = (SIZE / 2 - 12) / MAP_RADIUS;
const MIN_KS_RADIUS = 0.14;

export function createPlanner(container: HTMLElement, callbacks: PlannerCallbacks): PlannerApi {
  const canvas = document.createElement('canvas');
  canvas.width = SIZE;
  canvas.height = SIZE;
  canvas.style.cssText = `width:${SIZE}px;height:${SIZE}px;display:block;touch-action:none;cursor:crosshair;`;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas unavailable');

  let view: PlannerView | null = null;
  let dragging = false;
  let anchor = { x: 0, y: 0, r: 0, phi: 0 };
  let dragVector: { x: number; y: number; px: number } | null = null;

  const toWorld = (event: PointerEvent) => {
    const rect = canvas.getBoundingClientRect();
    const sx = ((event.clientX - rect.left) / rect.width) * SIZE;
    const sy = ((event.clientY - rect.top) / rect.height) * SIZE;
    return { x: (sx - CENTER) / SCALE, y: -(sy - CENTER) / SCALE };
  };

  const currentLaunchHeading = (): PreviewPoint | null => {
    return dragVector && dragVector.px >= 10
      ? { x: dragVector.x, y: dragVector.y }
      : null;
  };

  const placedState = (): PlayerState => {
    const height = callbacks.launchHeight?.() ?? 0;
    if (!dragVector || dragVector.px < 10) {
      return launchLocal({ r: anchor.r, phi: anchor.phi, betaRadial: 0, betaTangential: 0, height });
    }
    const len = Math.hypot(dragVector.x, dragVector.y) || 1;
    const dirX = dragVector.x / len;
    const dirY = dragVector.y / len;
    const speed = Math.min(0.92, Math.max(0.05, dragVector.px / 130));
    const radialX = Math.cos(anchor.phi);
    const radialY = Math.sin(anchor.phi);
    return launchLocal({
      r: anchor.r,
      phi: anchor.phi,
      betaRadial: speed * (dirX * radialX + dirY * radialY),
      betaTangential: speed * (-dirX * radialY + dirY * radialX),
      height,
    });
  };

  canvas.addEventListener('pointerdown', (event) => {
    dragging = true;
    canvas.setPointerCapture(event.pointerId);
    const w = toWorld(event);
    const rho = Math.hypot(w.x, w.y);
    const r = Math.max(MIN_KS_RADIUS, Math.min(equatorialKsRadius(rho, PARAMS) || MIN_KS_RADIUS, MAP_RADIUS * 0.95));
    const clampedRho = equatorialRho(r, PARAMS);
    const phi = Math.atan2(w.y, w.x);
    anchor = { x: clampedRho * Math.cos(phi), y: clampedRho * Math.sin(phi), r, phi };
    dragVector = null;
    callbacks.onPreview(placedState(), currentLaunchHeading());
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!dragging) return;
    const w = toWorld(event);
    const dx = w.x - anchor.x;
    const dy = w.y - anchor.y;
    dragVector = { x: dx, y: dy, px: Math.hypot(dx, dy) * SCALE };
    callbacks.onPreview(placedState(), currentLaunchHeading());
  });
  const finish = (event: PointerEvent) => {
    if (!dragging) return;
    dragging = false;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    callbacks.onCommit(placedState(), currentLaunchHeading());
    dragVector = null;
  };
  canvas.addEventListener('pointerup', finish);
  canvas.addEventListener('pointercancel', finish);

  const px = (wx: number, wy: number) => ({ x: CENTER + wx * SCALE, y: CENTER - wy * SCALE });

  const ring = (ksR: number, style: string, dash: number[], width = 1) => {
    ctx.beginPath();
    ctx.arc(CENTER, CENTER, equatorialRho(ksR, PARAMS) * SCALE, 0, Math.PI * 2);
    ctx.setLineDash(dash);
    ctx.strokeStyle = style;
    ctx.lineWidth = width;
    ctx.stroke();
    ctx.setLineDash([]);
  };

  const draw = () => {
    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.save();
    ctx.beginPath();
    ctx.arc(CENTER, CENTER, CENTER - 1, 0, Math.PI * 2);
    ctx.clip();
    ctx.fillStyle = '#05060a8c';
    ctx.fillRect(0, 0, SIZE, SIZE);

    // Faint polar grid.
    ctx.strokeStyle = '#2b334766';
    ctx.lineWidth = 1;
    for (let r = 4; r <= MAP_RADIUS; r += 4) {
      ctx.beginPath();
      ctx.arc(CENTER, CENTER, r * SCALE, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Accretion disk band.
    const diskGradient = ctx.createRadialGradient(
      CENTER, CENTER, equatorialRho(DISK_INNER, PARAMS) * SCALE,
      CENTER, CENTER, equatorialRho(DISK_OUTER, PARAMS) * SCALE,
    );
    diskGradient.addColorStop(0, '#ffd38a4a');
    diskGradient.addColorStop(0.25, '#d3944a2b');
    diskGradient.addColorStop(1, '#d3944a0c');
    ctx.beginPath();
    ctx.arc(CENTER, CENTER, equatorialRho(DISK_OUTER, PARAMS) * SCALE, 0, Math.PI * 2);
    ctx.arc(CENTER, CENTER, equatorialRho(DISK_INNER, PARAMS) * SCALE, 0, Math.PI * 2, true);
    ctx.fillStyle = diskGradient;
    ctx.fill();

    // Ergosphere region between the horizon and the static limit.
    ctx.beginPath();
    ctx.arc(CENTER, CENTER, equatorialRho(ERGOSPHERE, PARAMS) * SCALE, 0, Math.PI * 2);
    ctx.arc(CENTER, CENTER, equatorialRho(HORIZON, PARAMS) * SCALE, 0, Math.PI * 2, true);
    ctx.fillStyle = '#6b5a9e38';
    ctx.fill();
    ring(ERGOSPHERE, '#a18bdfaa', [4, 4], 1.2);

    // Photon orbits and ISCO.
    ring(PHOTON_RETROGRADE, '#9aa3b46e', [2, 5], 1.1);
    ring(PHOTON_PROGRADE, '#aeb7c98c', [2, 5], 1.15);
    ring(ISCO, '#ffd38ab8', [6, 5], 1.25);

    // Horizon.
    ctx.beginPath();
    ctx.arc(CENTER, CENTER, equatorialRho(HORIZON, PARAMS) * SCALE, 0, Math.PI * 2);
    ctx.fillStyle = '#000000cc';
    ctx.fill();
    ctx.strokeStyle = '#b6c0d0aa';
    ctx.lineWidth = 1.6;
    ctx.stroke();

    if (!view) return;

    // Worldline preview, fading with look-ahead time.
    const pts = view.preview;
    for (let i = 1; i < pts.length; i++) {
      const a = px(pts[i - 1].x, pts[i - 1].y);
      const b = px(pts[i].x, pts[i].y);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = '#ffd38a';
      ctx.globalAlpha = 0.85 * (1 - i / pts.length) + 0.05;
      ctx.lineWidth = 1.9;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Player marker and heading.
    const pos = px(view.state.position.x, view.state.position.y);
    const heading = dragging && dragVector && dragVector.px >= 10
      ? { x: dragVector.x, y: dragVector.y }
      : view.lookDirection;
    const hLen = Math.hypot(heading.x, heading.y) || 1;
    arrow(ctx, pos.x, pos.y, heading.x / hLen, -heading.y / hLen);

    ctx.beginPath();
    ctx.arc(pos.x, pos.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = view.state.r <= HORIZON ? '#e06050' : view.running ? '#ffffff' : '#e8b873';
    ctx.shadowColor = ctx.fillStyle;
    ctx.shadowBlur = 9;
    ctx.fill();
    ctx.shadowBlur = 0;

    // Labels.
    ctx.fillStyle = '#c3cad8';
    ctx.shadowColor = '#000';
    ctx.shadowBlur = 4;
    ctx.font = '10px ui-monospace, SFMono-Regular, Consolas, monospace';
    ctx.fillText('ergosphere', CENTER + equatorialRho(ERGOSPHERE, PARAMS) * SCALE * 0.72, CENTER - equatorialRho(ERGOSPHERE, PARAMS) * SCALE * 0.78);
    ctx.fillText('ISCO', CENTER + equatorialRho(ISCO, PARAMS) * SCALE * 0.76, CENTER + equatorialRho(ISCO, PARAMS) * SCALE * 0.84);
    ctx.shadowBlur = 0;
    ctx.restore();
  };

  return {
    draw(next: PlannerView) {
      view = next;
      draw();
    },
  };
}

function arrow(ctx: CanvasRenderingContext2D, x: number, y: number, dx: number, dy: number): void {
  const len = 30;
  const tx = x + dx * len;
  const ty = y + dy * len;
  ctx.strokeStyle = '#ffffffd8';
  ctx.fillStyle = '#ffffffd8';
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(tx, ty);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(tx + dx * 7, ty + dy * 7);
  ctx.lineTo(tx - dx * 5 - dy * 4.5, ty - dy * 5 + dx * 4.5);
  ctx.lineTo(tx - dx * 5 + dy * 4.5, ty - dy * 5 - dx * 4.5);
  ctx.closePath();
  ctx.fill();
}
