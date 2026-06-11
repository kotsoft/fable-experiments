// SVG diagrams for the Kerr tutorial. Everything is generated dynamically and
// computed from the real formulas (M = 1): the ray fan uses the same
// Hamiltonian integrator as the shader, the shadow uses Bardeen's exact
// parametrization, the radii plot uses the standard r+(a), r_ph(a), r_isco(a).

const NS = 'http://www.w3.org/2000/svg';
const C = {
  fg: '#c9cdd6', dim: '#5a5f6b', line: '#23262e', accent: '#e8b873',
  red: '#e06050', blue: '#7a9fd4', white: '#ffffff', shade: '#e8b87322',
};

type Attrs = Record<string, string | number>;
function el(parent: Element, name: string, attrs: Attrs = {}): SVGElement {
  const n = document.createElementNS(NS, name) as SVGElement;
  for (const k in attrs) n.setAttribute(k, String(attrs[k]));
  parent.appendChild(n);
  return n;
}
function txt(parent: Element, x: number, y: number, s: string, attrs: Attrs = {}): SVGElement {
  const t = el(parent, 'text', {
    x, y, fill: C.dim, 'font-size': 19, 'font-family': 'system-ui, sans-serif',
    'text-anchor': 'middle', ...attrs,
  });
  t.textContent = s;
  return t;
}
function pathD(pts: [number, number][]): string {
  return pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join('');
}
function svgRoot(id: string): SVGSVGElement {
  return document.getElementById(id) as unknown as SVGSVGElement;
}

// ---------- Kerr formulas (M = 1) ----------
const horizonR = (a: number) => 1 + Math.sqrt(Math.max(1 - a * a, 0));
const phPro = (a: number) => 2 * (1 + Math.cos((2 / 3) * Math.acos(-a)));
const phRet = (a: number) => 2 * (1 + Math.cos((2 / 3) * Math.acos(a)));
function isco(a: number, prograde: boolean): number {
  const z1 = 1 + Math.cbrt(1 - a * a) * (Math.cbrt(1 + a) + Math.cbrt(1 - a));
  const z2 = Math.sqrt(3 * a * a + z1 * z1);
  const root = Math.sqrt((3 - z1) * (3 + z1 + 2 * z2));
  return 3 + z2 + (prograde ? -root : root);
}

// ============================================================
// Diagram A: frame dragging and the ergosphere (top view, a = 0.95)
// ============================================================
{
  const a = 0.95;
  const svg = svgRoot('svgErgo');
  const cx = 400, cy = 240, s = 52;
  // Cartesian (Kerr-Schild) radius of a BL-radius circle in the equator
  const RofBL = (r: number) => Math.sqrt(r * r + a * a);

  // ergosphere (equatorial boundary r = 2M)
  el(svg, 'circle', { cx, cy, r: RofBL(2) * s, fill: C.shade, stroke: C.accent, 'stroke-width': 1, 'stroke-dasharray': '5 5' });
  // horizon
  el(svg, 'circle', { cx, cy, r: RofBL(horizonR(a)) * s, fill: '#000', stroke: '#3a3f4a', 'stroke-width': 1.5 });

  // dragging arrows: angular sweep proportional to the ZAMO rate omega(r)
  for (const r of [2.35, 3.1, 4.1, 5.6, 7.4]) {
    const delta = r * r - 2 * r + a * a;
    const A = (r * r + a * a) ** 2 - a * a * delta;
    const omega = (2 * a * r) / A;
    const span = omega * 9; // radians of arc, scaled for visibility
    const R = RofBL(r) * s;
    for (let k = 0; k < 3; k++) {
      const phi0 = (k / 3) * 2 * Math.PI + 0.5 + r;
      const pts: [number, number][] = [];
      for (let i = 0; i <= 24; i++) {
        const phi = phi0 + (i / 24) * span; // CCW = +phi
        pts.push([cx + R * Math.cos(phi), cy - R * Math.sin(phi)]);
      }
      el(svg, 'path', { d: pathD(pts), fill: 'none', stroke: C.fg, 'stroke-width': 2, opacity: 0.85 });
      // arrowhead along the local CCW tangent
      const phiE = phi0 + span;
      const tx = -Math.sin(phiE), ty = -Math.cos(phiE); // screen tangent for +phi
      const ex = cx + R * Math.cos(phiE), ey = cy - R * Math.sin(phiE);
      el(svg, 'path', {
        d: `M${ex + tx * 11} ${ey + ty * 11}L${ex - ty * 5.5} ${ey + tx * 5.5}L${ex + ty * 5.5} ${ey - tx * 5.5}Z`,
        fill: C.fg,
      });
    }
  }

  txt(svg, cx, cy + RofBL(horizonR(a)) * s + 24, 'horizon');
  txt(svg, cx, cy - RofBL(2) * s - 12, 'ergosphere — standing still is impossible inside', { fill: C.accent });
  txt(svg, cx, 462, 'arrow length ∝ dragging rate ω(r), falling as 1/r³');
}

// ============================================================
// 2D equatorial Hamiltonian integrator (same physics as the shader)
// ============================================================
function trace2D(a: number, b: number): { pts: [number, number][]; captured: boolean } {
  const rplus = horizonR(a);
  const H = (x: number, y: number, px: number, py: number) => {
    const r2 = Math.max(x * x + y * y - a * a, 1e-9);
    const r = Math.sqrt(r2);
    const f = 2 / r;
    const den = r2 + a * a;
    const lx = (r * x + a * y) / den, ly = (r * y - a * x) / den;
    const L = 1 + lx * px + ly * py; // forward photon, pt = -1
    return 0.5 * (-1 + px * px + py * py - f * L * L);
  };
  const deriv = (st: number[]) => {
    const [x, y, px, py] = st;
    const r2 = Math.max(x * x + y * y - a * a, 1e-9);
    const r = Math.sqrt(r2);
    const f = 2 / r;
    const den = r2 + a * a;
    const lx = (r * x + a * y) / den, ly = (r * y - a * x) / den;
    const L = 1 + lx * px + ly * py;
    const eps = 1e-6 * Math.max(1, Math.hypot(x, y));
    return [
      px - f * L * lx,
      py - f * L * ly,
      -(H(x + eps, y, px, py) - H(x - eps, y, px, py)) / (2 * eps),
      -(H(x, y + eps, px, py) - H(x, y - eps, px, py)) / (2 * eps),
    ];
  };
  // launch from far +x toward -x with offset y = b
  const x0 = 25;
  const r2i = x0 * x0 + b * b - a * a;
  const ri = Math.sqrt(r2i);
  const fi = 2 / ri;
  const deni = r2i + a * a;
  const c = -(ri * x0 + a * b) / deni;
  const A = 1 - fi * c * c, B = -2 * fi * c, Cc = -(1 + fi);
  const s0 = (-B + Math.sqrt(B * B - 4 * A * Cc)) / (2 * A);
  let st = [x0, b, -s0, 0];
  const pts: [number, number][] = [[x0, b]];
  let captured = false;
  for (let i = 0; i < 30000; i++) {
    const R = Math.hypot(st[0], st[1]);
    const rr = Math.sqrt(Math.max(R * R - a * a, 1e-9));
    if (rr < rplus * 1.005) { captured = true; break; }
    if (R > 27 && st[0] * st[2] + st[1] * st[3] > 0) break;
    const dl = Math.min(Math.max(0.012 * (rr - 0.9 * rplus), 0.003), 0.25);
    const k1 = deriv(st);
    const s2 = st.map((v, j) => v + 0.5 * dl * k1[j]);
    const k2 = deriv(s2);
    const s3 = st.map((v, j) => v + 0.5 * dl * k2[j]);
    const k3 = deriv(s3);
    const s4 = st.map((v, j) => v + dl * k3[j]);
    const k4 = deriv(s4);
    st = st.map((v, j) => v + (dl / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]));
    if (i % 4 === 0) pts.push([st[0], st[1]]);
  }
  pts.push([st[0], st[1]]);
  return { pts, captured };
}

// ============================================================
// Diagram B: prograde vs retrograde photons (a = 0.95)
// ============================================================
{
  const a = 0.95;
  const svg = svgRoot('svgFan');
  const cx = 430, cy = 280, s = 36;
  const toScreen = (pts: [number, number][]): [number, number][] =>
    pts.map(([x, y]) => [cx + x * s, cy - y * s]);

  // ergosphere + horizon (Cartesian radii)
  el(svg, 'circle', { cx, cy, r: Math.sqrt(4 + a * a) * s, fill: 'none', stroke: C.dim, 'stroke-width': 1, 'stroke-dasharray': '5 5' });

  const bs = [2.0, 2.45, 2.575, 2.7, 3.4, 4.6, -5.2, -6.4, -6.85, -6.95, -7.6, -8.8];
  for (const b of bs) {
    const r = trace2D(a, b);
    el(svg, 'path', {
      d: pathD(toScreen(r.pts)), fill: 'none',
      stroke: r.captured ? C.red : C.blue, 'stroke-width': 1.5, opacity: 0.8,
    });
  }

  el(svg, 'circle', { cx, cy, r: Math.sqrt(horizonR(a) ** 2 + a * a) * s, fill: '#000', stroke: '#3a3f4a', 'stroke-width': 1.5 });

  txt(svg, cx - 10, cy - 4.3 * s, 'prograde (with the spin): slips past at b ≈ 2.58 M', { fill: C.blue });
  txt(svg, cx, cy + 6.2 * s, 'retrograde (against the spin): captured out to b ≈ 6.92 M', { fill: C.red });
  txt(svg, cx + 7.9 * s, cy - 6.8 * s, 'spin: counterclockwise', { fill: C.dim, 'text-anchor': 'end' });
}

// ============================================================
// Diagram C: the shadow vs spin (Bardeen's exact rim) + slider
// ============================================================
{
  const svg = svgRoot('svgShadow');
  const cx = 380, cy = 300, s = 34;

  // Schwarzschild reference
  el(svg, 'circle', { cx, cy, r: 3 * Math.sqrt(3) * s, fill: 'none', stroke: C.dim, 'stroke-width': 1, 'stroke-dasharray': '6 6' });
  txt(svg, cx, cy - 3 * Math.sqrt(3) * s - 12, 'a = 0 reference (radius 3√3 M)');

  const rim = el(svg, 'path', { fill: '#000000aa', stroke: C.accent, 'stroke-width': 2 });
  const flatLabel = txt(svg, cx, cy, 'prograde side', { fill: C.accent, 'text-anchor': 'end' });
  const readout = document.getElementById('shadowReadout')!;

  function shadowPts(a: number): [number, number][] {
    if (a < 0.02) {
      const pts: [number, number][] = [];
      for (let i = 0; i <= 200; i++) {
        const t = (i / 200) * 2 * Math.PI;
        pts.push([3 * Math.sqrt(3) * Math.cos(t), 3 * Math.sqrt(3) * Math.sin(t)]);
      }
      return pts;
    }
    const r1 = phPro(a) + 1e-5, r2 = phRet(a) - 1e-5;
    const top: [number, number][] = [];
    for (let i = 0; i <= 400; i++) {
      const r = r1 + (i / 400) * (r2 - r1);
      const xi = (r * r * (3 - r) - a * a * (1 + r)) / (a * (r - 1));
      const eta = (r ** 3 * (4 * a * a - r * (r - 3) ** 2)) / (a * a * (r - 1) ** 2);
      if (eta < 0) continue;
      top.push([-xi, Math.sqrt(eta)]); // alpha = -xi for an equatorial observer
    }
    const bottom = top.slice().reverse().map(([x, y]) => [x, -y] as [number, number]);
    return top.concat(bottom);
  }

  function update(a: number) {
    const pts = shadowPts(a);
    rim.setAttribute('d', pathD(pts.map(([x, y]) => [cx + x * s, cy - y * s])) + 'Z');
    const xs = pts.map((p) => p[0]);
    const min = Math.min(...xs), max = Math.max(...xs);
    flatLabel.setAttribute('x', String(cx + min * s - 12));
    flatLabel.setAttribute('y', String(cy + 6));
    readout.textContent =
      `a = ${a.toFixed(3)} — width ${(max - min).toFixed(2)} M, center shifted ${((max + min) / 2).toFixed(2)} M off the hole`;
  }

  const slider = document.getElementById('shadowSpin') as HTMLInputElement;
  slider.addEventListener('input', () => update(parseFloat(slider.value)));
  update(parseFloat(slider.value));
}

// ============================================================
// Diagram D: characteristic radii vs spin
// ============================================================
{
  const svg = svgRoot('svgRadii');
  const W = 800, H = 500;
  const ml = 70, mr = 30, mt = 24, mb = 58;
  const X = (a: number) => ml + a * (W - ml - mr);
  const Y = (r: number) => H - mb - (r / 9.5) * (H - mt - mb);

  // axes
  el(svg, 'path', { d: `M${ml} ${mt}L${ml} ${H - mb}L${W - mr} ${H - mb}`, fill: 'none', stroke: '#3a3f4a', 'stroke-width': 1 });
  for (let r = 1; r <= 9; r++) {
    el(svg, 'path', { d: `M${ml - 6} ${Y(r)}L${ml} ${Y(r)}`, stroke: '#3a3f4a', 'stroke-width': 1 });
    txt(svg, ml - 22, Y(r) + 6, String(r));
  }
  for (const av of [0, 0.25, 0.5, 0.75, 1]) {
    el(svg, 'path', { d: `M${X(av)} ${H - mb}L${X(av)} ${H - mb + 6}`, stroke: '#3a3f4a', 'stroke-width': 1 });
    txt(svg, X(av), H - mb + 26, av.toFixed(2));
  }
  txt(svg, (W + ml) / 2, H - 12, 'spin a / M');
  txt(svg, 22, (H - mb + mt) / 2, 'r / M', { transform: `rotate(-90 22 ${(H - mb + mt) / 2})` });

  const curve = (f: (a: number) => number, stroke: string, dash = '') => {
    const pts: [number, number][] = [];
    for (let i = 0; i <= 200; i++) {
      const a = (i / 200) * 0.999;
      pts.push([X(a), Y(f(a))]);
    }
    el(svg, 'path', { d: pathD(pts), fill: 'none', stroke, 'stroke-width': 2, 'stroke-dasharray': dash });
  };
  curve((a) => isco(a, false), C.blue, '6 5');
  curve((a) => isco(a, true), C.blue);
  curve(phRet, C.accent, '6 5');
  curve(phPro, C.accent);
  curve(horizonR, C.fg);

  // demo default marker
  el(svg, 'path', { d: `M${X(0.95)} ${mt}L${X(0.95)} ${H - mb}`, stroke: C.dim, 'stroke-width': 1, 'stroke-dasharray': '3 5' });
  txt(svg, X(0.95), mt + 14, 'demo', { 'text-anchor': 'end' });

  // legend
  const lx = ml + 24, ly = mt + 10;
  const legend: [string, string, string][] = [
    ['ISCO retrograde', C.blue, '6 5'],
    ['ISCO prograde (disk inner edge)', C.blue, ''],
    ['photon orbit retrograde', C.accent, '6 5'],
    ['photon orbit prograde', C.accent, ''],
    ['horizon', C.fg, ''],
  ];
  legend.forEach(([label, color, dash], i) => {
    el(svg, 'path', { d: `M${lx} ${ly + i * 26}L${lx + 36} ${ly + i * 26}`, stroke: color, 'stroke-width': 2, 'stroke-dasharray': dash });
    txt(svg, lx + 46, ly + i * 26 + 6, label, { 'text-anchor': 'start', fill: C.fg, 'font-size': 17 });
  });
}
