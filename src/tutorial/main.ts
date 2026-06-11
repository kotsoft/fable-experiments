// 2D diagrams for the tutorial page. Every light path here is integrated with
// the same equation the renderer's shader uses (rs = 1):
//   a = -(3/2) h^2 x / r^5,  h = |x × v| conserved.

interface RayResult {
  pts: number[]; // flat x,y pairs
  captured: boolean;
  deflection: number; // accumulated turning of v, radians
  minR: number;
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

  for (let i = 0; i < 12000; i++) {
    const r2 = px * px + py * py;
    const r = Math.sqrt(r2);
    if (r < minR) minR = r;
    if (r < 1.0) { captured = true; break; }
    if (r2 > escapeR * escapeR && px * vx + py * vy > 0) break;

    const dt = Math.min(Math.max(0.03 * r, 0.004), 0.12);
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
    if (i % 2 === 0) pts.push(px, py);
  }
  pts.push(px, py);
  return { pts, captured, deflection, minR };
}

// ---------- canvas helpers ----------
const CSS = {
  fg: '#c9cdd6', dim: '#5a5f6b', accent: '#e8b873',
  captured: '#e06050', escaped: '#7a9fd4', highlight: '#ffffff',
};

function ctx2d(id: string): CanvasRenderingContext2D {
  const c = document.getElementById(id) as HTMLCanvasElement;
  return c.getContext('2d')!;
}

interface View { cx: number; cy: number; scale: number }

function drawPath(g: CanvasRenderingContext2D, v: View, pts: number[]) {
  g.beginPath();
  g.moveTo(v.cx + pts[0] * v.scale, v.cy - pts[1] * v.scale);
  for (let i = 2; i < pts.length; i += 2) {
    g.lineTo(v.cx + pts[i] * v.scale, v.cy - pts[i + 1] * v.scale);
  }
  g.stroke();
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
const FAN_VIEW: View = { cx: 1280 / 2, cy: 800 / 2, scale: 1280 / 22 };

function renderFan(bSel: number) {
  const g = fanG;
  g.clearRect(0, 0, 1280, 800);

  // background fan, fixed impact parameters
  g.lineWidth = 1.2;
  for (let b = 0.6; b <= 6.01; b += 0.45) {
    const r = traceRay(10.5, b, -1, 0);
    g.strokeStyle = r.captured ? CSS.captured : CSS.escaped;
    g.globalAlpha = 0.55;
    drawPath(g, FAN_VIEW, r.pts);
  }
  g.globalAlpha = 1;

  drawHole(g, FAN_VIEW);

  // highlighted ray
  const hr = traceRay(10.5, bSel, -1, 0);
  g.strokeStyle = CSS.highlight;
  g.lineWidth = 2.5;
  drawPath(g, FAN_VIEW, hr.pts);

  // impact parameter marker
  g.strokeStyle = CSS.accent;
  g.lineWidth = 1;
  g.setLineDash([4, 4]);
  g.beginPath();
  g.moveTo(FAN_VIEW.cx + 10.5 * FAN_VIEW.scale, FAN_VIEW.cy);
  g.lineTo(FAN_VIEW.cx + 10.5 * FAN_VIEW.scale, FAN_VIEW.cy - bSel * FAN_VIEW.scale);
  g.stroke();
  g.setLineDash([]);
  g.fillStyle = CSS.accent;
  g.font = '20px system-ui';
  g.textAlign = 'left';
  g.fillText('b', FAN_VIEW.cx + 10.5 * FAN_VIEW.scale + 8, FAN_VIEW.cy - (bSel * FAN_VIEW.scale) / 2);

  return hr;
}

// ============================================================
// Diagram B: effective potential V(r) = (1 - 1/r) / r^2
// ============================================================
const potG = ctx2d('potential');

function Veff(r: number): number {
  return (1 - 1 / r) / (r * r);
}

function renderPotential(bSel: number) {
  const g = potG;
  const W = 1280, H = 560;
  g.clearRect(0, 0, W, H);

  const rMin = 1, rMax = 8;
  const vMax = 0.165;
  const ml = 80, mr = 30, mt = 30, mb = 60;
  const X = (r: number) => ml + ((r - rMin) / (rMax - rMin)) * (W - ml - mr);
  const Y = (v: number) => H - mb - (v / vMax) * (H - mt - mb);

  // axes
  g.strokeStyle = '#3a3f4a';
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(ml, mt); g.lineTo(ml, H - mb); g.lineTo(W - mr, H - mb);
  g.stroke();
  g.fillStyle = CSS.dim;
  g.font = '20px system-ui';
  g.textAlign = 'center';
  g.fillText('r / rs', (W + ml) / 2, H - 18);
  g.save();
  g.translate(26, H / 2);
  g.rotate(-Math.PI / 2);
  g.fillText('V(r)', 0, 0);
  g.restore();
  for (let r = 1; r <= 8; r++) {
    g.fillText(String(r), X(r), H - mb + 28);
    g.beginPath();
    g.moveTo(X(r), H - mb); g.lineTo(X(r), H - mb + 6);
    g.stroke();
  }

  // potential curve
  g.strokeStyle = CSS.fg;
  g.lineWidth = 2;
  g.beginPath();
  for (let i = 0; i <= 400; i++) {
    const r = rMin + (i / 400) * (rMax - rMin);
    const x = X(r), y = Y(Veff(r));
    if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
  }
  g.stroke();

  // peak marker at r = 1.5
  const peakV = Veff(1.5);
  g.setLineDash([4, 4]);
  g.strokeStyle = CSS.dim;
  g.beginPath();
  g.moveTo(X(1.5), H - mb); g.lineTo(X(1.5), Y(peakV));
  g.stroke();
  g.setLineDash([]);
  g.fillStyle = CSS.dim;
  g.fillText('r = 1.5 rs', X(1.5), Y(peakV) - 36);
  g.fillText('1/bc²', X(1.5), Y(peakV) - 12);

  // 1/b^2 line
  const E = 1 / (bSel * bSel);
  const captured = E > peakV;
  g.strokeStyle = captured ? CSS.captured : CSS.escaped;
  g.lineWidth = 2;
  const yE = Y(Math.min(E, vMax));
  g.beginPath();
  g.moveTo(ml, yE); g.lineTo(W - mr, yE);
  g.stroke();
  g.fillStyle = captured ? CSS.captured : CSS.escaped;
  g.textAlign = 'right';
  g.fillText(`1/b² ${captured ? '— over the peak: captured' : ''}`, W - mr - 8, yE - 10);

  // turning point: outermost root of V(r) = E beyond the peak
  if (!captured) {
    let lo = 1.5, hi = rMax;
    if (Veff(hi) < E) {
      for (let i = 0; i < 60; i++) {
        const mid = (lo + hi) / 2;
        if (Veff(mid) > E) lo = mid; else hi = mid;
      }
      const rt = (lo + hi) / 2;
      g.beginPath();
      g.arc(X(rt), Y(E), 7, 0, Math.PI * 2);
      g.fillStyle = CSS.highlight;
      g.fill();
      g.textAlign = 'left';
      g.fillStyle = CSS.dim;
      g.fillText('turning point', X(rt) + 14, Y(E) + 28);
    }
  }
}

// slider wiring
const slider = document.getElementById('bSlider') as HTMLInputElement;
const readout = document.getElementById('bReadout') as HTMLOutputElement;

function update() {
  const b = parseFloat(slider.value);
  const hr = renderFan(b);
  renderPotential(b);
  const deg = Math.abs((hr.deflection * 180) / Math.PI);
  readout.textContent = hr.captured
    ? `b = ${b.toFixed(2)} rs — captured`
    : `b = ${b.toFixed(2)} rs — deflected ${deg.toFixed(0)}°`;
}
slider.addEventListener('input', update);
update();

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
// Diagram C: side view — how the disk images form
// ============================================================
{
  const g = ctx2d('sideView');
  const W = 1280, H = 760;
  const v: View = { cx: W * 0.42, cy: H / 2, scale: W / 27 };
  g.clearRect(0, 0, W, H);

  const camX = 13, camY = 1.6;
  const DISK_IN = 3, DISK_OUT = 10.5;

  // disk cross-section
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

  // rays from the camera; record equatorial-plane crossings inside the disk
  const baseAng = Math.atan2(-camY, -camX);
  const hits: { x: number; y: number; order: number }[] = [];
  for (let i = 0; i < 38; i++) {
    const ang = baseAng + (-0.30 + (i / 37) * 0.55);
    const r = traceRay(camX, camY, Math.cos(ang), Math.sin(ang), 26);
    // find crossings
    let order = 0;
    for (let j = 2; j < r.pts.length; j += 2) {
      const y0 = r.pts[j - 1], y1 = r.pts[j + 1];
      if (y0 === undefined || y1 === undefined) break;
      if (y0 * y1 < 0) {
        const t = y0 / (y0 - y1);
        const xc = r.pts[j - 2] + t * (r.pts[j] - r.pts[j - 2]);
        if (Math.abs(xc) >= DISK_IN && Math.abs(xc) <= DISK_OUT) {
          order++;
          hits.push({ x: xc, y: 0, order });
        }
      }
    }
    g.strokeStyle = r.captured ? CSS.captured : '#8a93a6';
    g.globalAlpha = 0.32;
    g.lineWidth = 1.1;
    drawPath(g, v, r.pts);
  }
  g.globalAlpha = 1;

  drawHole(g, v, false);

  // hit markers: near side = direct image, far side = the lensed arcs
  for (const h of hits) {
    g.beginPath();
    g.arc(v.cx + h.x * v.scale, v.cy - h.y * v.scale, 6, 0, Math.PI * 2);
    g.fillStyle = h.x > 0 ? '#ffb24d' : '#6db9ff';
    g.fill();
  }

  // camera marker
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
