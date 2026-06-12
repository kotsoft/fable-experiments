// 2D diagrams for the tutorial page. Every light path here is integrated with
// the same equation the renderer's shader uses (rs = 1):
//   a = -(3/2) h^2 x / r^5,  h = |x × v| conserved.

interface RayResult {
  pts: number[]; // flat x,y pairs
  captured: boolean;
  deflection: number; // accumulated turning of v, radians
  minR: number;
  horizonIndex: number | null;
}

function traceRay(px: number, py: number, vx: number, vy: number, escapeR = 30): RayResult {
  const vl = Math.hypot(vx, vy);
  vx /= vl; vy /= vl;
  const h = px * vy - py * vx;
  const h2 = h * h;
  const pts: number[] = [px, py];
  let captured = false;
  let deflection = 0;
  let minR = Math.hypot(px, py);
  let horizonIndex: number | null = null;

  const appendPoint = (x: number, y: number): number => {
    const last = pts.length - 2;
    if (last >= 0 && Math.hypot(pts[last] - x, pts[last + 1] - y) < 1e-7) {
      return last / 2;
    }
    pts.push(x, y);
    return pts.length / 2 - 1;
  };

  for (let i = 0; i < 30000; i++) {
    const r2 = px * px + py * py;
    const r = Math.sqrt(r2);
    if (r < minR) minR = r;
    if (!captured && r < 1.0) {
      captured = true;
      horizonIndex = appendPoint(px, py);
    }
    if (captured && r < 0.035) {
      appendPoint(px, py);
      break;
    }
    if (r2 > escapeR * escapeR && px * vx + py * vy > 0) break;

    const dt = captured
      ? Math.max(0.00001, Math.min(0.006, 0.012 * r * r))
      : Math.min(Math.max(0.03 * r, 0.00002), 0.12);
    const r5 = r2 * r2 * r;
    const a1x = -1.5 * h2 * px / r5;
    const a1y = -1.5 * h2 * py / r5;
    const pnx = px + vx * dt + 0.5 * a1x * dt * dt;
    const pny = py + vy * dt + 0.5 * a1y * dt * dt;
    const rn2 = pnx * pnx + pny * pny;
    const rn5 = rn2 * rn2 * Math.sqrt(rn2);
    const a2x = -1.5 * h2 * pnx / rn5;
    const a2y = -1.5 * h2 * pny / rn5;
    const nvx = vx + 0.5 * (a1x + a2x) * dt;
    const nvy = vy + 0.5 * (a1y + a2y) * dt;
    deflection += Math.atan2(vx * nvy - vy * nvx, vx * nvx + vy * nvy);
    px = pnx; py = pny; vx = nvx; vy = nvy;
    if (captured || i % 2 === 0) appendPoint(px, py);
  }
  appendPoint(px, py);
  return { pts, captured, deflection, minR, horizonIndex };
}

function interiorPts(result: RayResult): number[] {
  if (!result.captured || result.horizonIndex === null) return [];
  return result.pts.slice(result.horizonIndex * 2);
}

// ---------- canvas helpers ----------
const CSS = {
  fg: '#c9cdd6', dim: '#5a5f6b', accent: '#e8b873',
  captured: '#e06050', escaped: '#7a9fd4', highlight: '#ffffff',
  grid: '#181b23',
};

const SVG_NS = 'http://www.w3.org/2000/svg';
type Attrs = Record<string, string | number>;
type Point = [number, number];

function svgRoot(id: string): SVGSVGElement {
  return document.getElementById(id) as unknown as SVGSVGElement;
}

function svgEl(parent: Element, name: string, attrs: Attrs = {}): SVGElement {
  const node = document.createElementNS(SVG_NS, name) as SVGElement;
  for (const key in attrs) node.setAttribute(key, String(attrs[key]));
  parent.appendChild(node);
  return node;
}

function svgText(parent: Element, x: number, y: number, value: string, attrs: Attrs = {}): SVGElement {
  const node = svgEl(parent, 'text', {
    x,
    y,
    fill: CSS.dim,
    'font-size': 17,
    'font-family': 'system-ui, sans-serif',
    'text-anchor': 'middle',
    ...attrs,
  });
  node.textContent = value;
  return node;
}

function clearSvg(svg: SVGSVGElement): void {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
}

function svgPathD(points: Point[]): string {
  return points.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join('');
}

function svgArrow(svg: SVGSVGElement, x1: number, y1: number, x2: number, y2: number, color: string): void {
  svgEl(svg, 'path', { d: `M${x1} ${y1}L${x2} ${y2}`, stroke: color, 'stroke-width': 2.2, fill: 'none' });
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const a1 = angle + Math.PI * 0.82;
  const a2 = angle - Math.PI * 0.82;
  svgEl(svg, 'path', {
    d: `M${x2} ${y2}L${x2 + Math.cos(a1) * 10} ${y2 + Math.sin(a1) * 10}L${x2 + Math.cos(a2) * 10} ${y2 + Math.sin(a2) * 10}Z`,
    fill: color,
  });
}

function drawCanvasArrow(
  g: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  color: string,
): void {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const head = 13;
  g.save();
  g.strokeStyle = color;
  g.fillStyle = color;
  g.lineWidth = 2;
  g.lineCap = 'round';
  g.beginPath();
  g.moveTo(x1, y1);
  g.lineTo(x2, y2);
  g.stroke();
  g.beginPath();
  g.moveTo(x2, y2);
  g.lineTo(x2 - Math.cos(angle - 0.55) * head, y2 - Math.sin(angle - 0.55) * head);
  g.lineTo(x2 - Math.cos(angle + 0.55) * head, y2 - Math.sin(angle + 0.55) * head);
  g.closePath();
  g.fill();
  g.restore();
}

function ctx2d(id: string): CanvasRenderingContext2D {
  const c = document.getElementById(id) as HTMLCanvasElement;
  const logicalWidth = c.width;
  const logicalHeight = c.height;
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const pixelWidth = Math.round(logicalWidth * dpr);
  const pixelHeight = Math.round(logicalHeight * dpr);

  c.style.aspectRatio = `${logicalWidth} / ${logicalHeight}`;
  if (c.width !== pixelWidth || c.height !== pixelHeight) {
    c.width = pixelWidth;
    c.height = pixelHeight;
  }

  const g = c.getContext('2d')!;
  g.setTransform(pixelWidth / logicalWidth, 0, 0, pixelHeight / logicalHeight, 0, 0);
  return g;
}

interface View { cx: number; cy: number; scale: number }
interface ScreenPath {
  pts: Point[];
  lengths: number[];
  total: number;
}

function drawPath(g: CanvasRenderingContext2D, v: View, pts: number[]) {
  g.beginPath();
  g.moveTo(v.cx + pts[0] * v.scale, v.cy - pts[1] * v.scale);
  for (let i = 2; i < pts.length; i += 2) {
    g.lineTo(v.cx + pts[i] * v.scale, v.cy - pts[i + 1] * v.scale);
  }
  g.stroke();
}

function path2dFromWorldPts(pts: number[], v: View): Path2D | null {
  if (pts.length < 4) return null;
  const path = new Path2D();
  path.moveTo(v.cx + pts[0] * v.scale, v.cy - pts[1] * v.scale);
  for (let i = 2; i < pts.length; i += 2) {
    path.lineTo(v.cx + pts[i] * v.scale, v.cy - pts[i + 1] * v.scale);
  }
  return path;
}

function drawInteriorContinuation(
  g: CanvasRenderingContext2D,
  path: Path2D | null,
  color: string,
  width = 2,
  alpha = 0.5,
): void {
  if (!path) return;
  g.save();
  g.strokeStyle = color;
  g.globalAlpha = alpha;
  g.lineWidth = width;
  g.lineCap = 'round';
  g.stroke(path);
  g.restore();
}

function drawSingularityMarker(g: CanvasRenderingContext2D, v: View): void {
  g.save();
  g.fillStyle = CSS.captured;
  g.globalAlpha = 0.62;
  g.beginPath();
  g.arc(v.cx, v.cy, 4.5, 0, Math.PI * 2);
  g.fill();
  g.font = '15px system-ui';
  g.textAlign = 'center';
  g.fillText('singularity', v.cx, v.cy - 14);
  g.restore();
}

function screenPath(pts: number[], v: View): ScreenPath {
  const path: Point[] = [];
  const lengths: number[] = [0];
  let total = 0;
  for (let i = 0; i < pts.length; i += 2) {
    const p: Point = [v.cx + pts[i] * v.scale, v.cy - pts[i + 1] * v.scale];
    if (path.length > 0) {
      const prev = path[path.length - 1];
      total += Math.hypot(p[0] - prev[0], p[1] - prev[1]);
      lengths.push(total);
    }
    path.push(p);
  }
  return { pts: path, lengths, total };
}

function pointAt(path: ScreenPath, dist: number): Point {
  if (path.pts.length === 0) return [0, 0];
  if (dist <= 0) return path.pts[0];
  if (dist >= path.total) return path.pts[path.pts.length - 1];
  for (let i = 1; i < path.lengths.length; i++) {
    if (path.lengths[i] >= dist) {
      const prevDist = path.lengths[i - 1];
      const span = Math.max(path.lengths[i] - prevDist, 1e-6);
      const t = (dist - prevDist) / span;
      const a = path.pts[i - 1], b = path.pts[i];
      return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    }
  }
  return path.pts[path.pts.length - 1];
}

function drawPathSegment(
  g: CanvasRenderingContext2D,
  path: ScreenPath,
  fromDist: number,
  toDist: number,
  color: string,
  width: number,
  alpha: number,
): void {
  if (path.pts.length < 2 || toDist <= 0 || fromDist >= path.total) return;
  const from = Math.max(0, fromDist);
  const to = Math.min(path.total, toDist);
  const start = pointAt(path, from);
  g.save();
  g.globalAlpha = alpha;
  g.strokeStyle = color;
  g.lineWidth = width;
  g.lineCap = 'round';
  g.beginPath();
  g.moveTo(start[0], start[1]);
  for (let i = 1; i < path.pts.length; i++) {
    if (path.lengths[i] <= from) continue;
    if (path.lengths[i] >= to) {
      const end = pointAt(path, to);
      g.lineTo(end[0], end[1]);
      break;
    }
    g.lineTo(path.pts[i][0], path.pts[i][1]);
  }
  g.stroke();
  g.restore();
}

function drawPhotonPacket(
  g: CanvasRenderingContext2D,
  path: ScreenPath,
  elapsed: number,
  color: string,
  delay: number,
  speed = 360,
): void {
  if (path.total <= 0) return;
  const pause = 0.55;
  const travelTime = path.total / speed;
  const cycle = travelTime + pause;
  const age = (elapsed + delay) % cycle;
  const dist = Math.min(path.total, age * speed);
  drawPathSegment(g, path, dist - 92, dist, color, 4.5, 0.33);
  if (age <= travelTime) {
    const [x, y] = pointAt(path, dist);
    g.save();
    g.shadowBlur = 14;
    g.shadowColor = color;
    g.fillStyle = color;
    g.beginPath();
    g.arc(x, y, 6.5, 0, Math.PI * 2);
    g.fill();
    g.restore();
  }
}

function animateCanvasWhenVisible(
  canvas: HTMLCanvasElement,
  renderFrame: (elapsed: number) => void,
  renderStill: () => void = () => renderFrame(0),
): void {
  const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
  let raf = 0;
  let visible = false;
  let start = 0;

  const stop = (): void => {
    if (raf !== 0) cancelAnimationFrame(raf);
    raf = 0;
    start = 0;
    renderStill();
  };

  const loop = (now: number): void => {
    if (!visible || reduce) return;
    if (start === 0) start = now;
    renderFrame((now - start) / 1000);
    raf = requestAnimationFrame(loop);
  };

  const startLoop = (): void => {
    if (reduce) {
      renderStill();
      return;
    }
    if (raf === 0) raf = requestAnimationFrame(loop);
  };

  renderStill();
  if (!('IntersectionObserver' in window)) {
    visible = true;
    startLoop();
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    visible = entries.some((entry) => entry.isIntersecting);
    if (visible) startLoop();
    else stop();
  }, { rootMargin: '180px 0px' });
  observer.observe(canvas);
}

function drawHole(g: CanvasRenderingContext2D, v: View, labels = true) {
  // horizon
  g.beginPath();
  g.arc(v.cx, v.cy, v.scale, 0, Math.PI * 2);
  g.fillStyle = '#000';
  g.fill();
  g.strokeStyle = '#3a3f4a';
  g.lineWidth = 1.5;
  g.stroke();
  // photon sphere
  g.beginPath();
  g.arc(v.cx, v.cy, 1.5 * v.scale, 0, Math.PI * 2);
  g.setLineDash([6, 6]);
  g.strokeStyle = CSS.dim;
  g.lineWidth = 1;
  g.stroke();
  g.setLineDash([]);
  if (labels) {
    g.fillStyle = CSS.dim;
    g.font = '20px system-ui';
    g.textAlign = 'center';
    g.fillText('horizon', v.cx, v.cy + v.scale + 24);
    g.fillText('photon sphere', v.cx, v.cy - 1.5 * v.scale - 12);
  }
}

// ============================================================
// Diagram A: fan of rays + slider-controlled highlighted ray
// ============================================================
const fanG = ctx2d('rayFan');
const FAN_W = 1280;
const FAN_H = 1240;
const FAN_VIEW: View = { cx: FAN_W / 2, cy: 360, scale: FAN_W / 22 };
const CAPTURE_PANEL = {
  left: 42,
  right: FAN_W - 28,
  top: 790,
  bottom: 1168,
  plotLeft: 130,
  plotRight: 1130,
  plotTop: 862,
  plotBottom: 1124,
  rMin: 1,
  rMax: 10.8,
};

interface FanRay {
  b: number;
  result: RayResult;
  path: ScreenPath;
  interiorPath: Path2D | null;
  color: string;
  delay: number;
}

const fanBackgroundRays: FanRay[] = [];
for (let b = 0.6; b <= 6.01; b += 0.45) {
  const result = traceRay(10.5, b, -1, 0);
  fanBackgroundRays.push({
    b,
    result,
    path: screenPath(result.pts, FAN_VIEW),
    interiorPath: path2dFromWorldPts(interiorPts(result), FAN_VIEW),
    color: result.captured ? CSS.captured : CSS.escaped,
    delay: fanBackgroundRays.length * 0.29,
  });
}

function initialFanImpactParameter(): number {
  let lastCaptured: number | null = null;
  let firstEscaped: number | null = null;
  for (const ray of fanBackgroundRays) {
    if (ray.result.captured) lastCaptured = ray.b;
    else if (lastCaptured !== null) {
      firstEscaped = ray.b;
      break;
    }
  }
  return lastCaptured !== null && firstEscaped !== null
    ? (lastCaptured + firstEscaped) / 2
    : (3 * Math.sqrt(3)) / 2;
}

let fanSelectedB = initialFanImpactParameter();
let fanSelectedRay = traceRay(10.5, fanSelectedB, -1, 0);
let fanSelectedInteriorPath = path2dFromWorldPts(interiorPts(fanSelectedRay), FAN_VIEW);
let fanElapsed = 0;
let fanNextWhiteSpawn = 0;
let fanWhiteSpawnIndex = 0;

interface FanWhiteParticle {
  path: ScreenPath;
  born: number;
  speed: number;
}

const fanWhiteParticles: FanWhiteParticle[] = [];
const fanWhiteOffsets = [0, -0.12, 0.12, -0.24, 0.24];

function spawnFanWhiteParticle(elapsed: number): void {
  const offset = fanWhiteOffsets[fanWhiteSpawnIndex % fanWhiteOffsets.length];
  fanWhiteSpawnIndex++;
  const b = Math.min(6, Math.max(0.4, fanSelectedB + offset));
  const result = traceRay(10.5, b, -1, 0);
  fanWhiteParticles.push({
    path: screenPath(result.pts, FAN_VIEW),
    born: elapsed,
    speed: 430,
  });
}

function drawWhiteParticle(g: CanvasRenderingContext2D, particle: FanWhiteParticle, elapsed: number): void {
  const p = whiteParticlePoint(particle, elapsed);
  if (!p) return;
  drawPathSegment(g, particle.path, p.dist - 100, p.dist, CSS.highlight, 4.8, 0.38);
  g.save();
  g.shadowBlur = 18;
  g.shadowColor = CSS.highlight;
  g.fillStyle = CSS.highlight;
  g.beginPath();
  g.arc(p.x, p.y, 7, 0, Math.PI * 2);
  g.fill();
  g.restore();
}

function whiteParticlePoint(
  particle: FanWhiteParticle,
  elapsed: number,
): { x: number; y: number; dist: number; radius: number } | null {
  const age = elapsed - particle.born;
  if (age < 0) return null;
  const dist = age * particle.speed;
  if (dist > particle.path.total) return null;
  const [x, y] = pointAt(particle.path, dist);
  const wx = (x - FAN_VIEW.cx) / FAN_VIEW.scale;
  const wy = -(y - FAN_VIEW.cy) / FAN_VIEW.scale;
  return { x, y, dist, radius: Math.hypot(wx, wy) };
}

function renderFanFrame(elapsed: number, allowWhiteSpawns = true) {
  if (elapsed + 1e-3 < fanElapsed) {
    fanWhiteParticles.length = 0;
    fanNextWhiteSpawn = 0;
    fanWhiteSpawnIndex = 0;
  }
  fanElapsed = elapsed;
  const g = fanG;
  g.clearRect(0, 0, FAN_W, FAN_H);
  g.fillStyle = '#07080c';
  g.fillRect(0, 0, FAN_W, FAN_H);

  g.save();
  g.font = '22px system-ui';
  g.fillStyle = CSS.dim;
  g.textAlign = 'left';
  g.fillText('top-down ray plane', 70, 52);
  g.font = '18px system-ui';
  g.fillStyle = CSS.escaped;
  g.fillText('escape continues left', 70, 88);
  drawCanvasArrow(g, 285, 82, 174, 82, CSS.escaped);
  g.textAlign = 'right';
  g.fillStyle = CSS.accent;
  g.fillText('photons enter from right', FAN_W - 70, 88);
  drawCanvasArrow(g, FAN_W - 72, 82, FAN_W - 198, 82, CSS.accent);
  g.restore();

  // background fan, fixed impact parameters
  g.lineWidth = 1.2;
  for (const ray of fanBackgroundRays) {
    g.strokeStyle = ray.color;
    g.globalAlpha = 0.55;
    drawPath(g, FAN_VIEW, ray.result.pts);
  }
  g.globalAlpha = 1;

  drawHole(g, FAN_VIEW);
  for (const ray of fanBackgroundRays) {
    drawInteriorContinuation(g, ray.interiorPath, ray.color, 1.8, 0.42);
  }
  drawSingularityMarker(g, FAN_VIEW);

  // highlighted ray
  g.save();
  g.strokeStyle = CSS.highlight;
  g.globalAlpha = 0.42;
  g.lineWidth = 1.8;
  g.setLineDash([9, 8]);
  drawPath(g, FAN_VIEW, fanSelectedRay.pts);
  g.restore();
  drawInteriorContinuation(g, fanSelectedInteriorPath, CSS.highlight, 2.4, 0.5);

  // impact parameter marker
  g.strokeStyle = CSS.accent;
  g.lineWidth = 1;
  g.setLineDash([4, 4]);
  g.beginPath();
  g.moveTo(FAN_VIEW.cx + 10.5 * FAN_VIEW.scale, FAN_VIEW.cy);
  g.lineTo(FAN_VIEW.cx + 10.5 * FAN_VIEW.scale, FAN_VIEW.cy - fanSelectedB * FAN_VIEW.scale);
  g.stroke();
  g.setLineDash([]);
  g.fillStyle = CSS.accent;
  g.font = '20px system-ui';
  g.textAlign = 'left';
  g.fillText('b', FAN_VIEW.cx + 10.5 * FAN_VIEW.scale + 8, FAN_VIEW.cy - (fanSelectedB * FAN_VIEW.scale) / 2);

  for (const ray of fanBackgroundRays) {
    drawPhotonPacket(g, ray.path, elapsed, ray.color, ray.delay, 390);
  }

  if (allowWhiteSpawns) {
    while (elapsed >= fanNextWhiteSpawn) {
      spawnFanWhiteParticle(fanNextWhiteSpawn);
      fanNextWhiteSpawn += 0.22;
    }
  }
  for (let i = fanWhiteParticles.length - 1; i >= 0; i--) {
    const particle = fanWhiteParticles[i];
    const lifetime = (particle.path.total + 110) / particle.speed;
    if (elapsed - particle.born > lifetime) {
      fanWhiteParticles.splice(i, 1);
      continue;
    }
    drawWhiteParticle(g, particle, elapsed);
  }

  renderCaptureRulePanel(g, elapsed);
}

function setFanRay(bSel: number): RayResult {
  fanSelectedB = bSel;
  fanSelectedRay = traceRay(10.5, fanSelectedB, -1, 0);
  fanSelectedInteriorPath = path2dFromWorldPts(interiorPts(fanSelectedRay), FAN_VIEW);
  renderFanFrame(fanElapsed, false);
  return fanSelectedRay;
}

// ============================================================
// Lower panel: non-spatial radial capture rule
// ============================================================
function Veff(r: number): number {
  return (1 - 1 / r) / (r * r);
}

function captureX(r: number): number {
  const t = (r - CAPTURE_PANEL.rMin) / (CAPTURE_PANEL.rMax - CAPTURE_PANEL.rMin);
  return CAPTURE_PANEL.plotLeft + Math.max(0, Math.min(1, t)) * (CAPTURE_PANEL.plotRight - CAPTURE_PANEL.plotLeft);
}

function captureY(value: number): number {
  const peak = Veff(1.5);
  const max = peak * 1.36;
  const t = Math.min(value, max) / max;
  return CAPTURE_PANEL.plotBottom - t * (CAPTURE_PANEL.plotBottom - CAPTURE_PANEL.plotTop);
}

function outerTurningPoint(level: number): number | null {
  const peak = Veff(1.5);
  if (level >= peak) return null;
  let prevR = 1.5;
  let prev = Veff(prevR) - level;
  for (let i = 1; i <= 900; i++) {
    const r = 1.5 + ((CAPTURE_PANEL.rMax - 1.5) * i) / 900;
    const next = Veff(r) - level;
    if (prev >= 0 && next <= 0) {
      const t = prev / Math.max(prev - next, 1e-9);
      return prevR + (r - prevR) * t;
    }
    prevR = r;
    prev = next;
  }
  return null;
}

function renderCaptureRulePanel(g: CanvasRenderingContext2D, elapsed: number): void {
  const peak = Veff(1.5);
  const selected = 1 / (fanSelectedB * fanSelectedB);
  const captured = selected > peak;
  const selectedColor = captured ? CSS.captured : CSS.escaped;
  const yPeak = captureY(peak);
  const ySelected = captureY(selected);
  const turnR = outerTurningPoint(selected);

  g.save();
  g.fillStyle = '#090a0f';
  g.fillRect(
    CAPTURE_PANEL.left,
    CAPTURE_PANEL.top - 66,
    CAPTURE_PANEL.right - CAPTURE_PANEL.left,
    CAPTURE_PANEL.bottom - CAPTURE_PANEL.top + 118,
  );
  g.strokeStyle = '#23262e';
  g.lineWidth = 1;
  g.strokeRect(
    CAPTURE_PANEL.left,
    CAPTURE_PANEL.top - 66,
    CAPTURE_PANEL.right - CAPTURE_PANEL.left,
    CAPTURE_PANEL.bottom - CAPTURE_PANEL.top + 118,
  );

  g.font = '28px system-ui';
  g.fillStyle = CSS.fg;
  g.textAlign = 'center';
  g.fillText('capture rule, not space: the photon-sphere barrier', FAN_W / 2, CAPTURE_PANEL.top - 28);
  g.font = '20px system-ui';
  g.fillStyle = CSS.dim;
  g.textAlign = 'left';
  g.fillText('radial coordinate r / rs along bottom; higher line means more inward reach', CAPTURE_PANEL.left + 36, CAPTURE_PANEL.top + 2);
  g.fillText(captured ? '1/b² clears the peak -> captured' : '1/b² hits the curve -> turns around and escapes', CAPTURE_PANEL.left + 36, CAPTURE_PANEL.top + 30);

  g.save();
  g.beginPath();
  g.rect(
    CAPTURE_PANEL.plotLeft - 24,
    CAPTURE_PANEL.plotTop - 18,
    CAPTURE_PANEL.plotRight - CAPTURE_PANEL.plotLeft + 56,
    CAPTURE_PANEL.plotBottom - CAPTURE_PANEL.plotTop + 58,
  );
  g.clip();

  g.strokeStyle = '#303541';
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(CAPTURE_PANEL.plotLeft, CAPTURE_PANEL.plotTop);
  g.lineTo(CAPTURE_PANEL.plotLeft, CAPTURE_PANEL.plotBottom);
  g.lineTo(CAPTURE_PANEL.plotRight, CAPTURE_PANEL.plotBottom);
  g.stroke();

  for (const r of [1, 1.5, 3, 6, 10]) {
    const x = captureX(r);
    g.strokeStyle = r === 1.5 ? '#3a3f4a' : CSS.grid;
    g.lineWidth = r === 1.5 ? 1.3 : 1;
    g.beginPath();
    g.moveTo(x, CAPTURE_PANEL.plotTop);
    g.lineTo(x, CAPTURE_PANEL.plotBottom);
    g.stroke();
    g.fillStyle = CSS.dim;
    g.font = '17px system-ui';
    g.textAlign = 'center';
    g.fillText(r === 1.5 ? '1.5' : String(r), x, CAPTURE_PANEL.plotBottom + 26);
  }

  g.strokeStyle = CSS.fg;
  g.lineWidth = 2.5;
  g.beginPath();
  for (let i = 0; i <= 360; i++) {
    const r = CAPTURE_PANEL.rMin + ((CAPTURE_PANEL.rMax - CAPTURE_PANEL.rMin) * i) / 360;
    const x = captureX(r);
    const y = captureY(Veff(Math.max(r, 1.001)));
    if (i === 0) g.moveTo(x, y);
    else g.lineTo(x, y);
  }
  g.stroke();

  g.setLineDash([7, 6]);
  g.strokeStyle = CSS.dim;
  g.beginPath();
  g.moveTo(CAPTURE_PANEL.plotLeft, yPeak);
  g.lineTo(CAPTURE_PANEL.plotRight, yPeak);
  g.stroke();
  g.setLineDash([]);
  g.fillStyle = CSS.dim;
  g.font = '18px system-ui';
  g.textAlign = 'left';
  g.fillText('peak = 1/bc² = 4/27', captureX(1.62), yPeak - 12);

  g.strokeStyle = selectedColor;
  g.lineWidth = 2.2;
  g.setLineDash([10, 7]);
  g.beginPath();
  g.moveTo(CAPTURE_PANEL.plotLeft, ySelected);
  g.lineTo(CAPTURE_PANEL.plotRight, ySelected);
  g.stroke();
  g.setLineDash([]);
  g.fillStyle = selectedColor;
  g.font = '19px system-ui';
  g.textAlign = 'right';
  g.fillText(`slider: 1/b² - ${captured ? 'captured' : 'escapes'}`, CAPTURE_PANEL.plotRight, Math.max(CAPTURE_PANEL.plotTop + 22, ySelected - 12));

  g.fillStyle = CSS.highlight;
  g.font = '17px system-ui';
  g.textAlign = 'center';
  g.fillText('photon sphere', captureX(1.5), yPeak - 36);
  g.fillText('r = 1.5rs', captureX(1.5), CAPTURE_PANEL.plotBottom + 50);

  g.fillStyle = CSS.captured;
  g.textAlign = 'left';
  g.fillText('horizon', captureX(1) + 10, CAPTURE_PANEL.plotBottom - 16);

  if (turnR !== null) {
    const tx = captureX(turnR);
    const ty = captureY(selected);
    g.strokeStyle = CSS.escaped;
    g.lineWidth = 1.2;
    g.setLineDash([5, 5]);
    g.beginPath();
    g.moveTo(tx, ty);
    g.lineTo(tx, CAPTURE_PANEL.plotBottom);
    g.stroke();
    g.setLineDash([]);
    g.fillStyle = CSS.escaped;
    g.beginPath();
    g.arc(tx, ty, 6, 0, Math.PI * 2);
    g.fill();
    g.font = '18px system-ui';
    g.textAlign = 'left';
    g.fillText('turning point', tx + 12, ty - 10);
  }

  for (const particle of fanWhiteParticles) {
    const p = whiteParticlePoint(particle, elapsed);
    if (!p) continue;
    if (p.radius < CAPTURE_PANEL.rMin || p.radius > CAPTURE_PANEL.rMax) continue;
    const r = Math.max(1.001, p.radius);
    const x = captureX(r);
    const y = captureY(Veff(r));
    g.save();
    g.globalAlpha = 0.88;
    g.shadowBlur = 12;
    g.shadowColor = CSS.highlight;
    g.fillStyle = CSS.highlight;
    g.beginPath();
    g.arc(x, y, 5.5, 0, Math.PI * 2);
    g.fill();
    g.restore();
  }

  g.restore();
  g.fillStyle = CSS.dim;
  g.font = '18px system-ui';
  g.textAlign = 'center';
  g.fillText('r / rs', (CAPTURE_PANEL.plotLeft + CAPTURE_PANEL.plotRight) / 2, CAPTURE_PANEL.bottom + 36);
  g.save();
  g.translate(CAPTURE_PANEL.plotLeft - 70, (CAPTURE_PANEL.plotTop + CAPTURE_PANEL.plotBottom) / 2);
  g.rotate(-Math.PI / 2);
  g.fillText('V(r)', 0, 0);
  g.restore();
  g.restore();
}

// slider wiring
const slider = document.getElementById('bSlider') as HTMLInputElement;
const readout = document.getElementById('bReadout') as HTMLOutputElement;
slider.value = fanSelectedB.toFixed(3);

function update() {
  const b = parseFloat(slider.value);
  const hr = setFanRay(b);
  const deg = Math.abs((hr.deflection * 180) / Math.PI);
  readout.textContent = hr.captured
    ? `b = ${b.toFixed(2)} rs — captured`
    : `b = ${b.toFixed(2)} rs — deflected ${deg.toFixed(0)}°`;
}
slider.addEventListener('input', update);
update();
animateCanvasWhenVisible(
  fanG.canvas,
  (elapsed) => renderFanFrame(elapsed, true),
  () => renderFanFrame(fanElapsed, false),
);

// ============================================================
// Force-law comparison: Newtonian 1/r^2 vs photon h^2/r^4
// ============================================================
{
  const svg = svgRoot('forceLawCompare');
  const radiusSlider = document.getElementById('forceRadius') as HTMLInputElement;
  const hSlider = document.getElementById('forceH') as HTMLInputElement;
  const radiusReadout = document.getElementById('forceRadiusReadout') as HTMLOutputElement;
  const hReadout = document.getElementById('forceHReadout') as HTMLOutputElement;
  const W = 800;
  const H = 440;
  const plot = { x: 340, y: 74, w: 400, h: 278 };
  const rMin = 1.05;
  const rMax = 10;
  const logMin = -3.3;
  const logMax = 1.2;
  const xOf = (r: number): number => plot.x + ((r - rMin) / (rMax - rMin)) * plot.w;
  const yOf = (value: number): number => {
    const log = Math.log10(Math.max(value, 10 ** logMin));
    const t = (log - logMin) / (logMax - logMin);
    return plot.y + plot.h - Math.max(0, Math.min(1, t)) * plot.h;
  };
  const newton = (r: number): number => 1 / (r * r);
  const photon = (r: number, h: number): number => 1.5 * h * h / (r ** 4);
  const fmt = (value: number): string => value >= 10
    ? value.toFixed(1)
    : value >= 0.1
      ? value.toFixed(3)
      : value.toExponential(2);

  function curveD(fn: (r: number) => number): string {
    const pts: Point[] = [];
    for (let i = 0; i <= 180; i++) {
      const r = rMin + (i / 180) * (rMax - rMin);
      pts.push([xOf(r), yOf(fn(r))]);
    }
    return svgPathD(pts);
  }

  function equationCard(y: number, color: string, title: string, vectorEq: string, magnitudeEq: string): void {
    svgEl(svg, 'rect', { x: 42, y, width: 248, height: 112, rx: 7, fill: '#0b0d13', stroke: '#23262e' });
    svgText(svg, 62, y + 28, title, { fill: color, 'font-size': 17, 'text-anchor': 'start' });
    svgText(svg, 62, y + 58, vectorEq, { fill: CSS.fg, 'font-size': 16, 'text-anchor': 'start' });
    svgText(svg, 62, y + 86, magnitudeEq, { fill: CSS.dim, 'font-size': 15, 'text-anchor': 'start' });
  }

  function renderForceLawCompare(): void {
    clearSvg(svg);
    const r = parseFloat(radiusSlider.value);
    const h = parseFloat(hSlider.value);
    const n = newton(r);
    const p = photon(r, h);
    radiusReadout.textContent = `r = ${r.toFixed(2)} rs`;
    hReadout.textContent = `h = ${h.toFixed(2)} | h² = ${(h * h).toFixed(2)}`;

    svgEl(svg, 'rect', { x: 0, y: 0, width: W, height: H, fill: '#07080c' });
    svgText(svg, 400, 36, 'same central-acceleration loop, different radial falloff', { fill: CSS.fg, 'font-size': 19 });
    svgText(svg, 400, 62, 'The vector x contributes one power of r before you take the magnitude.', { fill: CSS.dim, 'font-size': 15 });

    equationCard(94, '#86c79a', 'Newtonian massive particle', 'a = -GM x / r^3', '|a| = GM / r^2');
    equationCard(232, CSS.accent, 'Schwarzschild photon track', 'a = -1.5 h^2 x / r^5', '|a| = 1.5 h^2 / r^4');

    svgEl(svg, 'path', { d: `M${plot.x} ${plot.y}L${plot.x} ${plot.y + plot.h}L${plot.x + plot.w} ${plot.y + plot.h}`, fill: 'none', stroke: '#3a3f4a', 'stroke-width': 1 });
    for (const tick of [1, 2, 4, 6, 8, 10]) {
      const x = xOf(tick);
      svgEl(svg, 'path', { d: `M${x} ${plot.y}L${x} ${plot.y + plot.h}`, stroke: tick === 1 ? '#3a3f4a' : CSS.grid, 'stroke-width': 1 });
      svgText(svg, x, plot.y + plot.h + 26, String(tick), { 'font-size': 13 });
    }
    for (const value of [10, 1, 0.1, 0.01, 0.001]) {
      const y = yOf(value);
      svgEl(svg, 'path', { d: `M${plot.x - 6} ${y}L${plot.x + plot.w} ${y}`, stroke: value === 1 ? '#3a3f4a' : CSS.grid, 'stroke-width': 1 });
      svgText(svg, plot.x - 16, y + 5, value >= 1 ? String(value) : value.toString(), { 'font-size': 12, 'text-anchor': 'end' });
    }

    svgEl(svg, 'path', { d: curveD(newton), fill: 'none', stroke: '#86c79a', 'stroke-width': 2.8 });
    svgEl(svg, 'path', { d: curveD((rr) => photon(rr, h)), fill: 'none', stroke: CSS.accent, 'stroke-width': 2.8 });

    const selectedX = xOf(r);
    svgEl(svg, 'path', { d: `M${selectedX} ${plot.y}L${selectedX} ${plot.y + plot.h}`, stroke: CSS.highlight, 'stroke-width': 1.2, 'stroke-dasharray': '5 5' });
    svgEl(svg, 'circle', { cx: selectedX, cy: yOf(n), r: 5, fill: '#86c79a' });
    svgEl(svg, 'circle', { cx: selectedX, cy: yOf(p), r: 5, fill: CSS.accent });

    svgText(svg, plot.x + plot.w - 8, yOf(newton(8.8)) - 10, '1/r^2', { fill: '#86c79a', 'font-size': 16, 'text-anchor': 'end' });
    svgText(svg, plot.x + plot.w - 8, yOf(photon(8.8, h)) + 20, 'h^2/r^4', { fill: CSS.accent, 'font-size': 16, 'text-anchor': 'end' });
    svgText(svg, plot.x + plot.w / 2, H - 28, 'radius r / rs', { fill: CSS.dim, 'font-size': 15 });
    svgText(svg, plot.x - 58, plot.y + plot.h / 2, 'magnitude, log scale', { fill: CSS.dim, 'font-size': 14, transform: `rotate(-90 ${plot.x - 58} ${plot.y + plot.h / 2})` });

    svgEl(svg, 'rect', { x: 340, y: 366, width: 400, height: 45, rx: 7, fill: '#0b0d13', stroke: '#23262e' });
    svgText(svg, 360, 394, `at r=${r.toFixed(2)}:  Newtonian ${fmt(n)}    photon ${fmt(p)}`, {
      fill: CSS.fg,
      'font-size': 16,
      'text-anchor': 'start',
    });
  }

  radiusSlider.addEventListener('input', renderForceLawCompare);
  hSlider.addEventListener('input', renderForceLawCompare);
  renderForceLawCompare();
}

// ============================================================
// Metric explorer: what each term in ds^2 contributes
// ============================================================
{
  const svg = svgRoot('metricExplorer');
  const massSlider = document.getElementById('metricMass') as HTMLInputElement;
  const radiusSlider = document.getElementById('metricRadius') as HTMLInputElement;
  const angleSlider = document.getElementById('metricAngle') as HTMLInputElement;
  const massReadout = document.getElementById('metricMassReadout') as HTMLOutputElement;
  const radiusReadout = document.getElementById('metricRadiusReadout') as HTMLOutputElement;
  const angleReadout = document.getElementById('metricAngleReadout') as HTMLOutputElement;
  const W = 800;
  const H = 500;
  const cx = 245;
  const cy = 268;
  const rsScale = 24;
  const rsPerSolarMassKm = 2.95325008;

  const compact = (value: number): string => {
    const abs = Math.abs(value);
    if (abs >= 1e9) return `${(value / 1e9).toFixed(abs >= 1e10 ? 1 : 2)}B`;
    if (abs >= 1e6) return `${(value / 1e6).toFixed(abs >= 1e7 ? 1 : 2)}M`;
    if (abs >= 1e3) return `${(value / 1e3).toFixed(abs >= 1e4 ? 1 : 2)}K`;
    if (abs >= 10) return value.toFixed(1);
    return value.toFixed(2);
  };

  function arcPath(radius: number, angle: number): string {
    const shown = Math.min(Math.max(angle, 0.001), Math.PI * 1.95);
    const sx = cx + radius;
    const sy = cy;
    const ex = cx + radius * Math.cos(shown);
    const ey = cy - radius * Math.sin(shown);
    const largeArc = shown > Math.PI ? 1 : 0;
    return `M${sx} ${sy}A${radius} ${radius} 0 ${largeArc} 0 ${ex} ${ey}`;
  }

  function labelValue(x: number, y: number, label: string, value: string, color: string): void {
    svgText(svg, x, y, label, { fill: CSS.dim, 'font-size': 14, 'text-anchor': 'start' });
    svgText(svg, x, y + 24, value, { fill: color, 'font-size': 19, 'text-anchor': 'start' });
  }

  function metricBar(
    y: number,
    color: string,
    label: string,
    value: string,
    fraction: number,
    term: string,
  ): void {
    const x = 500;
    const width = 232;
    svgText(svg, x, y, label, { fill: CSS.fg, 'font-size': 16, 'text-anchor': 'start' });
    svgText(svg, x, y + 20, term, { fill: CSS.dim, 'font-size': 13, 'text-anchor': 'start' });
    svgText(svg, x, y + 41, value, { fill: color, 'font-size': 15, 'text-anchor': 'start' });
    svgEl(svg, 'rect', { x, y: y + 52, width, height: 10, rx: 5, fill: '#151821' });
    svgEl(svg, 'rect', {
      x,
      y: y + 52,
      width: Math.max(3, Math.min(width, width * fraction)),
      height: 10,
      rx: 5,
      fill: color,
      opacity: 0.9,
    });
  }

  function renderMetricExplorer(): void {
    clearSvg(svg);
    const massSolar = 10 ** parseFloat(massSlider.value);
    const rsKm = rsPerSolarMassKm * massSolar;
    const rho = parseFloat(radiusSlider.value);
    const angleDeg = parseFloat(angleSlider.value);
    const angleRad = angleDeg * Math.PI / 180;
    const rKm = rho * rsKm;
    const f = 1 - 1 / rho;
    const timeFactor = Math.sqrt(f);
    const radialStretch = 1 / timeFactor;
    const arcKm = rKm * angleRad;

    massReadout.textContent = `M=${compact(massSolar)} solar | rs=${compact(rsKm)} km`;
    radiusReadout.textContent = `r=${rho.toFixed(2)} rs | ${compact(rKm)} km`;
    angleReadout.textContent = `dφ=${angleDeg.toFixed(1)}° | r dφ=${compact(arcKm)} km`;

    svgEl(svg, 'rect', { x: 0, y: 0, width: W, height: H, fill: '#07080c' });
    svgText(svg, 400, 36, 'Schwarzschild metric as three scale factors', { fill: CSS.fg, 'font-size': 19 });
    svgText(svg, 400, 62, 'ds² = -f c²dt²  +  dr²/f  +  r²dΩ²,    f = 1 - rs/r', { fill: CSS.dim, 'font-size': 16 });

    for (let r = 2; r <= 8; r += 2) {
      svgEl(svg, 'circle', { cx, cy, r: r * rsScale, fill: 'none', stroke: CSS.grid, 'stroke-width': 1 });
    }

    const selectedRadius = rho * rsScale;
    svgEl(svg, 'path', {
      d: `M${cx} ${cy}L${cx + selectedRadius} ${cy}`,
      stroke: CSS.dim,
      'stroke-width': 1,
      'stroke-dasharray': '5 5',
      fill: 'none',
    });
    svgEl(svg, 'circle', { cx, cy, r: rsScale, fill: '#000', stroke: '#3a3f4a', 'stroke-width': 1.5 });
    svgEl(svg, 'circle', {
      cx,
      cy,
      r: selectedRadius,
      fill: 'none',
      stroke: CSS.highlight,
      'stroke-width': 2.2,
    });
    svgEl(svg, 'path', { d: arcPath(selectedRadius, angleRad), fill: 'none', stroke: CSS.accent, 'stroke-width': 5, 'stroke-linecap': 'round' });

    const ex = cx + selectedRadius * Math.cos(angleRad);
    const ey = cy - selectedRadius * Math.sin(angleRad);
    svgEl(svg, 'circle', { cx: ex, cy: ey, r: 6, fill: CSS.accent });
    svgText(svg, cx, cy + rsScale + 24, 'horizon', { 'font-size': 14 });
    const selectedLabelX = selectedRadius > 150 ? cx + selectedRadius - 12 : cx + selectedRadius + 12;
    const selectedLabelAnchor = selectedRadius > 150 ? 'end' : 'start';
    svgText(svg, selectedLabelX, cy - 8, 'selected r', { fill: CSS.highlight, 'font-size': 14, 'text-anchor': selectedLabelAnchor });
    svgText(svg, cx + selectedRadius * 0.82, cy - selectedRadius * 0.26 - 14, 'r dφ', { fill: CSS.accent, 'font-size': 15 });

    labelValue(54, 105, 'physical horizon size', `rs = ${compact(rsKm)} km`, CSS.accent);
    labelValue(54, 158, 'selected coordinate radius', `r = ${rho.toFixed(2)} rs`, CSS.highlight);
    labelValue(54, 414, 'mass changes absolute scale', 'shape depends on r/rs', CSS.dim);

    metricBar(
      112,
      '#7a9fd4',
      'time part',
      `${timeFactor.toFixed(3)} local sec / distant sec`,
      timeFactor,
      '-(1 - rs/r)c²dt²',
    );
    metricBar(
      206,
      '#e8b873',
      'radial part',
      `${radialStretch.toFixed(2)} local km / coordinate km`,
      Math.min(radialStretch / 5, 1),
      'dr² / (1 - rs/r)',
    );
    metricBar(
      300,
      '#86c79a',
      'angular part',
      `${compact(arcKm)} km for ${angleDeg.toFixed(1)}°`,
      angleDeg / 45,
      'r²dΩ²  ->  arc length r dφ',
    );

    svgText(svg, 616, 420, `f = ${f.toFixed(3)} at r = ${rho.toFixed(2)}rs`, { fill: CSS.fg, 'font-size': 17 });
    svgText(svg, 616, 448, 'near the horizon: clocks slow, radial rulers stretch', { fill: CSS.dim, 'font-size': 14 });
  }

  massSlider.addEventListener('input', renderMetricExplorer);
  radiusSlider.addEventListener('input', renderMetricExplorer);
  angleSlider.addEventListener('input', renderMetricExplorer);
  renderMetricExplorer();
}

// ============================================================
// Diagram Ω: disk angular velocity and differential rotation
// ============================================================
{
  const svg = svgRoot('omegaDisk');
  const radiusSlider = document.getElementById('omegaRadius') as HTMLInputElement;
  const timeSlider = document.getElementById('omegaTime') as HTMLInputElement;
  const radiusReadout = document.getElementById('omegaRadiusReadout') as HTMLOutputElement;
  const timeReadout = document.getElementById('omegaTimeReadout') as HTMLOutputElement;
  const W = 800;
  const H = 470;
  const cx = 240;
  const cy = 245;
  const scale = 16.5;
  const diskIn = 3;
  const diskOut = 14;
  const omega = (r: number): number => Math.SQRT1_2 / Math.pow(r, 1.5);
  const screenPoint = (r: number, phi: number): Point => [
    cx + r * scale * Math.cos(phi),
    cy - r * scale * Math.sin(phi),
  ];

  function arcD(r: number, phi: number): string {
    const radius = r * scale;
    const shown = ((phi % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    const [sx, sy] = screenPoint(r, 0);
    const [ex, ey] = screenPoint(r, shown);
    const largeArc = shown > Math.PI ? 1 : 0;
    return `M${sx} ${sy}A${radius} ${radius} 0 ${largeArc} 0 ${ex} ${ey}`;
  }

  function renderOmega(): void {
    clearSvg(svg);
    const selectedR = parseFloat(radiusSlider.value);
    const time = parseFloat(timeSlider.value);
    const selectedOmega = omega(selectedR);
    const selectedPhi = selectedOmega * time;
    const selectedTurns = selectedPhi / (2 * Math.PI);
    radiusReadout.textContent = `r = ${selectedR.toFixed(2)} rs`;
    timeReadout.textContent = `t=${time.toFixed(1)} | Ω=${selectedOmega.toFixed(3)} | turns=${selectedTurns.toFixed(2)}`;

    svgEl(svg, 'rect', { x: 0, y: 0, width: W, height: H, fill: '#07080c' });

    for (let r = diskIn; r <= diskOut; r++) {
      svgEl(svg, 'circle', {
        cx,
        cy,
        r: r * scale,
        fill: 'none',
        stroke: r === diskIn ? CSS.accent : CSS.grid,
        'stroke-width': r === diskIn ? 1.4 : 1,
        'stroke-dasharray': r === diskIn ? '6 5' : '',
      });
    }
    svgEl(svg, 'circle', { cx, cy, r: 1 * scale, fill: '#000', stroke: '#3a3f4a', 'stroke-width': 1.5 });
    svgText(svg, cx, cy + scale + 22, 'horizon', { 'font-size': 14 });
    svgText(svg, cx + diskIn * scale + 22, cy + 5, 'ISCO', { fill: CSS.accent, 'font-size': 14, 'text-anchor': 'start' });

    svgEl(svg, 'path', {
      d: `M${cx} ${cy}L${cx + diskOut * scale + 18} ${cy}`,
      stroke: CSS.dim,
      'stroke-width': 1,
      'stroke-dasharray': '5 6',
      fill: 'none',
    });
    svgText(svg, cx + diskOut * scale - 12, cy + 23, 't = 0', { 'text-anchor': 'end', 'font-size': 14 });

    for (const r of [3, 4.2, 6, 8.5, 11, 14]) {
      const phi = omega(r) * time;
      const color = Math.abs(r - selectedR) < 0.35 ? CSS.highlight : r <= 6 ? CSS.accent : CSS.escaped;
      const width = Math.abs(r - selectedR) < 0.35 ? 3 : 1.8;
      const [x, y] = screenPoint(r, phi);
      svgEl(svg, 'path', {
        d: arcD(r, phi),
        fill: 'none',
        stroke: color,
        'stroke-width': width,
        opacity: Math.abs(r - selectedR) < 0.35 ? 1 : 0.72,
      });
      svgEl(svg, 'circle', {
        cx: x,
        cy: y,
        r: Math.abs(r - selectedR) < 0.35 ? 6.5 : 4.6,
        fill: color,
      });
    }

    const [sx, sy] = screenPoint(selectedR, selectedPhi);
    const tangent = selectedPhi + Math.PI / 2;
    svgArrow(svg, sx, sy, sx + Math.cos(tangent) * 34, sy - Math.sin(tangent) * 34, CSS.highlight);
    svgEl(svg, 'circle', { cx, cy, r: selectedR * scale, fill: 'none', stroke: CSS.highlight, 'stroke-width': 2.4 });

    svgText(svg, cx, 38, 'same elapsed time, different angular speeds', { fill: CSS.fg, 'font-size': 18 });
    svgText(svg, cx, 440, 'Ω(r) is the rate that advances a ring angle: φ(t) = Ωt', { fill: CSS.dim, 'font-size': 15 });

    const ml = 505;
    const mr = 42;
    const mt = 72;
    const mb = 78;
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;
    const xOf = (r: number): number => ml + ((r - diskIn) / (diskOut - diskIn)) * plotW;
    const yMax = omega(diskIn) * 1.08;
    const yOf = (value: number): number => H - mb - (value / yMax) * plotH;

    svgEl(svg, 'path', { d: `M${ml} ${mt}L${ml} ${H - mb}L${W - mr} ${H - mb}`, fill: 'none', stroke: '#3a3f4a', 'stroke-width': 1 });
    for (const r of [3, 5, 8, 11, 14]) {
      svgEl(svg, 'path', { d: `M${xOf(r)} ${H - mb}L${xOf(r)} ${H - mb + 6}`, stroke: '#3a3f4a', 'stroke-width': 1 });
      svgText(svg, xOf(r), H - mb + 28, String(r), { 'font-size': 14 });
    }
    for (const value of [0, 0.05, 0.1, 0.15]) {
      svgEl(svg, 'path', {
        d: `M${ml - 6} ${yOf(value)}L${ml} ${yOf(value)}M${ml} ${yOf(value)}L${W - mr} ${yOf(value)}`,
        stroke: value === 0 ? '#3a3f4a' : CSS.grid,
        'stroke-width': 1,
      });
      svgText(svg, ml - 24, yOf(value) + 5, value.toFixed(2), { 'font-size': 13 });
    }
    const curve: Point[] = [];
    for (let i = 0; i <= 180; i++) {
      const r = diskIn + (i / 180) * (diskOut - diskIn);
      curve.push([xOf(r), yOf(omega(r))]);
    }
    svgEl(svg, 'path', { d: svgPathD(curve), fill: 'none', stroke: CSS.accent, 'stroke-width': 2.2 });
    svgEl(svg, 'circle', { cx: xOf(selectedR), cy: yOf(selectedOmega), r: 6, fill: CSS.highlight });
    svgEl(svg, 'path', {
      d: `M${xOf(selectedR)} ${yOf(selectedOmega)}L${xOf(selectedR)} ${H - mb}`,
      stroke: CSS.highlight,
      'stroke-width': 1,
      'stroke-dasharray': '4 5',
      fill: 'none',
    });
    svgText(svg, ml + plotW / 2, H - 14, 'radius r / rs', { 'font-size': 15 });
    svgText(svg, 466, mt + plotH / 2, 'Ω', { transform: `rotate(-90 466 ${mt + plotH / 2})`, 'font-size': 16 });
    svgText(svg, ml + plotW / 2, 42, 'Keplerian disk rate', { fill: CSS.fg, 'font-size': 18 });
    svgText(svg, ml + plotW / 2, 62, 'Ω ∝ r^-3/2', { fill: CSS.accent, 'font-size': 15 });
  }

  radiusSlider.addEventListener('input', renderOmega);
  timeSlider.addEventListener('input', renderOmega);
  renderOmega();
}

// ============================================================
// Diagram E (geodesic primer): two "straight" paths on a sphere
// ============================================================
{
  const g = ctx2d('geoSphere');
  const W = 1280, H = 720;
  const cx = W / 2, cy = H / 2 + 40, R = 290;
  const tilt = (28 * Math.PI) / 180; // tip the pole toward the viewer
  g.clearRect(0, 0, W, H);

  // lat/lon (radians) -> screen point + depth; visible when depth > 0
  const proj = (lat: number, lon: number) => {
    const x = Math.cos(lat) * Math.sin(lon);
    const y = Math.sin(lat);
    const z = Math.cos(lat) * Math.cos(lon);
    return {
      sx: cx + R * x,
      sy: cy - R * (y * Math.cos(tilt) - z * Math.sin(tilt)),
      d: y * Math.sin(tilt) + z * Math.cos(tilt),
    };
  };

  const strokeCurve = (pts: { sx: number; sy: number; d: number }[]) => {
    let started = false;
    g.beginPath();
    for (const p of pts) {
      if (p.d <= 0.02) { started = false; continue; }
      if (!started) { g.moveTo(p.sx, p.sy); started = true; }
      else g.lineTo(p.sx, p.sy);
    }
    g.stroke();
  };
  const sample = (f: (t: number) => { sx: number; sy: number; d: number }) => {
    const pts = [];
    for (let i = 0; i <= 120; i++) pts.push(f(i / 120));
    return pts;
  };

  // sphere outline
  g.beginPath();
  g.arc(cx, cy, R, 0, Math.PI * 2);
  g.strokeStyle = '#3a3f4a';
  g.lineWidth = 1.5;
  g.stroke();

  // faint graticule
  g.strokeStyle = '#23262e';
  g.lineWidth = 1;
  for (const latDeg of [-30, 30, 60]) {
    const lat = (latDeg * Math.PI) / 180;
    strokeCurve(sample((t) => proj(lat, -Math.PI + t * 2 * Math.PI)));
  }
  for (let lonDeg = -90; lonDeg <= 90; lonDeg += 30) {
    const lon = (lonDeg * Math.PI) / 180;
    strokeCurve(sample((t) => proj(-Math.PI / 2 + t * Math.PI, lon)));
  }
  // equator, slightly brighter
  g.strokeStyle = '#363b47';
  strokeCurve(sample((t) => proj(0, -Math.PI + t * 2 * Math.PI)));

  // the two ant paths: meridians from the equator to the pole
  g.strokeStyle = CSS.accent;
  g.lineWidth = 3;
  for (const lonDeg of [-26, 26]) {
    const lon = (lonDeg * Math.PI) / 180;
    strokeCurve(sample((t) => proj(t * Math.PI / 2, lon)));
    // start dot + arrowhead pointing "north" along the path
    const a = proj(0, lon), b = proj(0.16, lon);
    g.fillStyle = CSS.accent;
    g.beginPath();
    g.arc(a.sx, a.sy, 7, 0, Math.PI * 2);
    g.fill();
    const dx = b.sx - a.sx, dy = b.sy - a.sy;
    const l = Math.hypot(dx, dy), ux = dx / l, uy = dy / l;
    const tipx = a.sx + ux * 52, tipy = a.sy + uy * 52;
    g.beginPath();
    g.moveTo(tipx + ux * 14, tipy + uy * 14);
    g.lineTo(tipx - uy * 8, tipy + ux * 8);
    g.lineTo(tipx + uy * 8, tipy - ux * 8);
    g.closePath();
    g.fill();
  }

  // meeting point at the pole
  const pole = proj(Math.PI / 2, 0);
  g.beginPath();
  g.arc(pole.sx, pole.sy, 8, 0, Math.PI * 2);
  g.fillStyle = '#fff';
  g.fill();

  g.font = '22px system-ui';
  g.textAlign = 'center';
  g.fillStyle = CSS.dim;
  const s1 = proj(0, (-26 * Math.PI) / 180), s2 = proj(0, (26 * Math.PI) / 180);
  g.fillText('start parallel, walk straight, never turn', (s1.sx + s2.sx) / 2, Math.max(s1.sy, s2.sy) + 56);
  g.fillStyle = '#fff';
  g.fillText('…and still meet', pole.sx, pole.sy - 26);
}

// ============================================================
// Diagram F (geodesic primer): same ray, flat vs curved space
// ============================================================
{
  const g = ctx2d('geoGrid');
  const W = 1280, H = 720;
  const v: View = { cx: W / 2, cy: H / 2, scale: W / 26 };
  g.clearRect(0, 0, W, H);

  // illustrative warped grid: points pulled inward near the mass
  const warp = (x: number, y: number) => {
    const r = Math.hypot(x, y);
    const s = 1 - 1.15 / (r + 1.5);
    return { x: x * s, y: y * s };
  };
  g.strokeStyle = '#1d212b';
  g.lineWidth = 1;
  for (let gx = -12; gx <= 12; gx += 2) {
    g.beginPath();
    for (let y = -7; y <= 7; y += 0.2) {
      const p = warp(gx, y);
      const sx = v.cx + p.x * v.scale, sy = v.cy - p.y * v.scale;
      if (y === -7) g.moveTo(sx, sy); else g.lineTo(sx, sy);
    }
    g.stroke();
  }
  for (let gy = -6; gy <= 6; gy += 2) {
    g.beginPath();
    for (let x = -12; x <= 12; x += 0.2) {
      const p = warp(x, gy);
      const sx = v.cx + p.x * v.scale, sy = v.cy - p.y * v.scale;
      if (x === -12) g.moveTo(sx, sy); else g.lineTo(sx, sy);
    }
    g.stroke();
  }

  const b = 3.4; // impact parameter for both rays

  // flat space: a straight line
  g.strokeStyle = CSS.dim;
  g.lineWidth = 2;
  g.setLineDash([8, 8]);
  g.beginPath();
  g.moveTo(v.cx + 12 * v.scale, v.cy - b * v.scale);
  g.lineTo(v.cx - 12 * v.scale, v.cy - b * v.scale);
  g.stroke();
  g.setLineDash([]);

  // curved space: the real geodesic, same start, same direction
  const ray = traceRay(12, b, -1, 0, 26);
  g.strokeStyle = CSS.accent;
  g.lineWidth = 3;
  drawPath(g, v, ray.pts);

  // the mass
  g.beginPath();
  g.arc(v.cx, v.cy, v.scale, 0, Math.PI * 2);
  g.fillStyle = '#000';
  g.fill();
  g.strokeStyle = '#3a3f4a';
  g.lineWidth = 1.5;
  g.stroke();

  // shared start marker
  g.beginPath();
  g.arc(v.cx + 12 * v.scale, v.cy - b * v.scale, 6, 0, Math.PI * 2);
  g.fillStyle = '#fff';
  g.fill();

  g.font = '22px system-ui';
  g.fillStyle = CSS.dim;
  g.textAlign = 'right';
  g.fillText('flat space: straight ahead stays straight', v.cx - 3 * v.scale, v.cy - (b + 0.7) * v.scale);
  g.fillStyle = CSS.accent;
  g.textAlign = 'left';
  g.fillText('curved space: same instructions', v.cx - 11.5 * v.scale, v.cy + 5.0 * v.scale);
  g.fillStyle = '#fff';
  g.textAlign = 'left';
  g.fillText('light starts here →', v.cx + 8.2 * v.scale, v.cy - (b + 0.7) * v.scale);
}

// ============================================================
// Diagram G (beaming primer): wavefronts from a moving source
// ============================================================
{
  const g = ctx2d('dopplerWaves');
  const W = 1280, H = 640;
  const cy = H / 2;
  g.clearRect(0, 0, W, H);

  const beta = 0.5;
  const N = 7;           // wavefronts emitted at t = 0..N-1, viewed at t = N
  const u = 41;          // px per light-unit
  const ox = W / 2 - 30; // where the source was at t = 0

  for (let i = 0; i < N; i++) {
    const cx = ox + beta * i * u;
    const r = (N - i) * u;
    g.beginPath();
    g.arc(cx, cy, r, 0, Math.PI * 2);
    g.strokeStyle = '#4a5060';
    g.lineWidth = 1.4;
    g.stroke();
  }

  // the source now, with its motion arrow
  const sx = ox + beta * N * u;
  g.beginPath();
  g.arc(sx, cy, 8, 0, Math.PI * 2);
  g.fillStyle = CSS.accent;
  g.fill();
  g.strokeStyle = CSS.accent;
  g.fillStyle = CSS.accent;
  g.lineWidth = 2.5;
  g.beginPath();
  g.moveTo(sx + 16, cy);
  g.lineTo(sx + 52, cy);
  g.stroke();
  g.beginPath();
  g.moveTo(sx + 64, cy);
  g.lineTo(sx + 48, cy - 8);
  g.lineTo(sx + 48, cy + 8);
  g.closePath();
  g.fill();

  g.font = '22px system-ui';
  g.textAlign = 'center';
  g.fillStyle = '#6db9ff';
  g.fillText('bunched ahead — bluer', ox + N * u, cy - N * u * 0.62 - 16);
  g.fillStyle = '#e08070';
  g.fillText('stretched behind — redder', ox - N * u * 0.55, cy + N * u * 0.62 + 32);
}

// ============================================================
// Diagram H (beaming primer): relativistic aberration of rays
// ============================================================
{
  const g = ctx2d('beamingArrows');
  const W = 1280, H = 640;
  g.clearRect(0, 0, W, H);

  const beta = 0.5;
  const R = 195;
  const panels = [
    { cx: 340, cy: 290, b: 0, label: 'at rest: glows evenly' },
    { cx: 940, cy: 290, b: beta, label: 'at 0.5c: the same rays, thrown forward' },
  ];

  for (const p of panels) {
    const pg = 1 / Math.sqrt(1 - p.b * p.b);
    g.strokeStyle = CSS.fg;
    g.fillStyle = CSS.fg;
    g.lineWidth = 2;
    for (let i = 0; i < 24; i++) {
      const th = (i / 24) * Math.PI * 2;
      // aberration: angles uniform in the blob's frame, seen from ours
      const den = 1 + p.b * Math.cos(th);
      const dx = (Math.cos(th) + p.b) / den;
      const dy = Math.sin(th) / (pg * den);
      const l = Math.hypot(dx, dy), ux = dx / l, uy = -dy / l;
      // arrow length ~ Doppler boost (normalized sideways), so the lobe reads
      const len = R * Math.pow(den, 0.75);
      const x1 = p.cx + ux * 26, y1 = p.cy + uy * 26;
      const x2 = p.cx + ux * len, y2 = p.cy + uy * len;
      g.beginPath();
      g.moveTo(x1, y1);
      g.lineTo(x2, y2);
      g.stroke();
      g.beginPath();
      g.moveTo(x2 + ux * 12, y2 + uy * 12);
      g.lineTo(x2 - uy * 6, y2 + ux * 6);
      g.lineTo(x2 + uy * 6, y2 - ux * 6);
      g.closePath();
      g.fill();
    }
    // the blob
    g.beginPath();
    g.arc(p.cx, p.cy, 12, 0, Math.PI * 2);
    g.fillStyle = CSS.accent;
    g.fill();
    g.font = '22px system-ui';
    g.textAlign = 'center';
    g.fillStyle = CSS.dim;
    g.fillText(p.label, p.cx, p.cy + R + 70);
    g.fillStyle = CSS.fg;
  }

  // motion arrow under the moving panel
  const mp = panels[1];
  g.strokeStyle = CSS.accent;
  g.fillStyle = CSS.accent;
  g.lineWidth = 2.5;
  g.beginPath();
  g.moveTo(mp.cx + 24, mp.cy);
  g.lineTo(mp.cx + 70, mp.cy);
  g.stroke();
  g.beginPath();
  g.moveTo(mp.cx + 82, mp.cy);
  g.lineTo(mp.cx + 66, mp.cy - 8);
  g.lineTo(mp.cx + 66, mp.cy + 8);
  g.closePath();
  g.fill();
}

// ============================================================
// Diagram C: side view — how the disk images form
// ============================================================
{
  const g = ctx2d('sideView');
  const W = 1280, H = 760;
  const v: View = { cx: W * 0.42, cy: H / 2, scale: W / 27 };
  const camX = 13, camY = 1.6;
  const DISK_IN = 3, DISK_OUT = 10.5;

  interface SideHit {
    x: number;
    y: number;
    order: number;
  }

  interface SideRay {
    result: RayResult;
    path: ScreenPath;
    hits: SideHit[];
    color: string;
    delay: number;
  }

  const baseAng = Math.atan2(-camY, -camX);
  const sideRays: SideRay[] = [];
  const hits: SideHit[] = [];
  for (let i = 0; i < 38; i++) {
    const ang = baseAng + (-0.30 + (i / 37) * 0.55);
    const r = traceRay(camX, camY, Math.cos(ang), Math.sin(ang), 26);
    const rayHits: SideHit[] = [];
    let order = 0;
    for (let j = 2; j < r.pts.length; j += 2) {
      const y0 = r.pts[j - 1], y1 = r.pts[j + 1];
      if (y0 === undefined || y1 === undefined) break;
      if (y0 * y1 < 0) {
        const t = y0 / (y0 - y1);
        const xc = r.pts[j - 2] + t * (r.pts[j] - r.pts[j - 2]);
        if (Math.abs(xc) >= DISK_IN && Math.abs(xc) <= DISK_OUT) {
          order++;
          const hit = { x: xc, y: 0, order };
          hits.push(hit);
          rayHits.push(hit);
        }
      }
    }
    const firstHit = rayHits[0];
    const color = firstHit ? (firstHit.x > 0 ? '#ffb24d' : '#6db9ff') : r.captured ? CSS.captured : '#8a93a6';
    sideRays.push({
      result: r,
      path: screenPath(r.pts, v),
      hits: rayHits,
      color,
      delay: i * 0.18,
    });
  }

  const takeEvenly = (rays: SideRay[], count: number): SideRay[] => {
    if (rays.length <= count) return rays;
    const picked: SideRay[] = [];
    for (let i = 0; i < count; i++) {
      picked.push(rays[Math.round((i / Math.max(count - 1, 1)) * (rays.length - 1))]);
    }
    return picked;
  };
  const animatedSideRays = [
    ...takeEvenly(sideRays.filter((ray) => ray.hits[0]?.x > 0), 4),
    ...takeEvenly(sideRays.filter((ray) => ray.hits[0]?.x < 0), 5),
  ];

  function drawSideDisk(): void {
    g.strokeStyle = CSS.accent;
    g.lineWidth = 5;
    g.lineCap = 'round';
    for (const s of [-1, 1]) {
      g.beginPath();
      g.moveTo(v.cx + s * DISK_IN * v.scale, v.cy);
      g.lineTo(v.cx + s * DISK_OUT * v.scale, v.cy);
      g.stroke();
    }
    g.lineCap = 'butt';
  }

  function drawSideHitMarkers(): void {
    for (const h of hits) {
      g.beginPath();
      g.arc(v.cx + h.x * v.scale, v.cy - h.y * v.scale, 6, 0, Math.PI * 2);
      g.fillStyle = h.x > 0 ? '#ffb24d' : '#6db9ff';
      g.fill();
    }
  }

  function drawSideCameraAndLabels(): void {
    g.beginPath();
    g.arc(v.cx + camX * v.scale, v.cy - camY * v.scale, 8, 0, Math.PI * 2);
    g.fillStyle = '#fff';
    g.fill();
    g.fillStyle = CSS.dim;
    g.font = '20px system-ui';
    g.textAlign = 'center';
    g.fillText('camera', v.cx + camX * v.scale, v.cy - camY * v.scale - 18);
    g.fillStyle = '#ffb24d';
    g.fillText('direct image', v.cx + 6.5 * v.scale, v.cy + 40);
    g.fillStyle = '#6db9ff';
    g.textAlign = 'left';
    g.fillText('lensed (arcs over / under the shadow)', v.cx - 10 * v.scale, v.cy - 1.9 * v.scale - 50);
  }

  function renderSideView(elapsed: number): void {
    g.clearRect(0, 0, W, H);
    drawSideDisk();
    for (const ray of sideRays) {
      g.strokeStyle = ray.result.captured ? CSS.captured : '#8a93a6';
      g.globalAlpha = 0.32;
      g.lineWidth = 1.1;
      drawPath(g, v, ray.result.pts);
    }
    g.globalAlpha = 1;
    drawHole(g, v, false);
    drawSideHitMarkers();
    drawSideCameraAndLabels();

    for (const ray of animatedSideRays) {
      drawPhotonPacket(g, ray.path, elapsed, ray.color, ray.delay, 500);
    }
  }

  animateCanvasWhenVisible(g.canvas, renderSideView, () => renderSideView(0));
}

// ============================================================
// Diagram D: Doppler beaming, top view
// ============================================================
{
  const g = ctx2d('doppler');
  const W = 1280, H = 700;
  const cx = W / 2, cy = H / 2 - 30;
  const scale = 36;
  g.clearRect(0, 0, W, H);

  const RIN = 3, ROUT = 8;
  const beta = 0.5;
  const gamma = 1 / Math.sqrt(1 - beta * beta);

  // blackbody-ish color (same approximation family as the shader)
  const bb = (t: number): [number, number, number] => {
    t = Math.max(t, 400);
    let r = 56100000 * Math.pow(t, -1.5) + 148;
    let gn = t > 6500 ? 35200000 * Math.pow(t, -1.5) + 184 : 100.04 * Math.log(t) - 623.6;
    let b = 194.18 * Math.log(t) - 1448.6;
    r = Math.min(Math.max(r, 0), 255);
    gn = Math.min(Math.max(gn, 0), 255);
    b = Math.min(Math.max(b, 0), 255);
    return [r, gn, b];
  };

  // annulus, wedge by wedge; observer is at the bottom: n = (0, -1)
  const N = 240;
  for (let i = 0; i < N; i++) {
    const th0 = (i / N) * Math.PI * 2;
    const th1 = ((i + 1.5) / N) * Math.PI * 2;
    const thm = (th0 + th1) / 2;
    // prograde CCW velocity direction: (-sin, cos); beta·n with n = (0,-1) is -beta*cos(thm)
    const dotBN = -beta * Math.cos(thm);
    const dop = 1 / (gamma * (1 - dotBN));
    const boost = Math.pow(dop, 3);
    const [r, gn, b] = bb(4200 * dop);
    const lum = Math.min(boost * 0.38, 1.25);
    g.fillStyle = `rgb(${Math.min(r * lum, 255)},${Math.min(gn * lum, 255)},${Math.min(b * lum, 255)})`;
    g.beginPath();
    g.arc(cx, cy, ROUT * scale, th0, th1 + 0.01);
    g.arc(cx, cy, RIN * scale, th1 + 0.01, th0, true);
    g.closePath();
    g.fill();
  }

  drawHole(g, { cx, cy, scale }, false);

  // rotation arrows (CCW)
  g.strokeStyle = '#fff';
  g.fillStyle = '#fff';
  g.lineWidth = 2;
  for (const th of [Math.PI / 2, (3 * Math.PI) / 2]) {
    const rr = (RIN + ROUT) / 2 * scale;
    const ax = cx + rr * Math.cos(th), ay = cy - rr * Math.sin(th);
    // tangent for CCW in screen coords (y flipped): direction (-sin, -cos) in canvas
    const tx = -Math.sin(th), ty = -Math.cos(th);
    g.beginPath();
    g.moveTo(ax - tx * 26, ay - ty * 26);
    g.lineTo(ax + tx * 26, ay + ty * 26);
    g.stroke();
    g.beginPath();
    g.moveTo(ax + tx * 38, ay + ty * 38);
    g.lineTo(ax + tx * 18 - ty * 9, ay + ty * 18 + tx * 9);
    g.lineTo(ax + tx * 18 + ty * 9, ay + ty * 18 - tx * 9);
    g.closePath();
    g.fill();
  }

  // observer arrow
  g.strokeStyle = CSS.dim;
  g.fillStyle = CSS.dim;
  g.lineWidth = 2;
  g.beginPath();
  g.moveTo(cx, cy + ROUT * scale + 16);
  g.lineTo(cx, cy + ROUT * scale + 56);
  g.stroke();
  g.beginPath();
  g.moveTo(cx, cy + ROUT * scale + 66);
  g.lineTo(cx - 8, cy + ROUT * scale + 50);
  g.lineTo(cx + 8, cy + ROUT * scale + 50);
  g.closePath();
  g.fill();
  g.font = '20px system-ui';
  g.textAlign = 'center';
  g.fillText('to camera', cx, cy + ROUT * scale + 92);
  g.fillStyle = '#cfe0ff';
  g.fillText('approaching: beamed g³ brighter, blueshifted', cx - ROUT * scale - 0, 40);
  g.textAlign = 'center';
  g.fillStyle = '#d89b7a';
  g.fillText('receding: dimmed, redshifted', cx + ROUT * scale - 60, H - 24);
}
