// SVG diagrams for the fall tutorial. The dynamic plots mirror the compact
// Schwarzschild proper-time equations used by src/fall/physics.ts.

const NS = 'http://www.w3.org/2000/svg';
const C = {
  fg: '#c9cdd6',
  dim: '#5a5f6b',
  line: '#23262e',
  grid: '#181b23',
  accent: '#e8b873',
  red: '#e06050',
  blue: '#7a9fd4',
  green: '#86c79a',
  white: '#ffffff',
  shade: '#e8b87322',
};

type Attrs = Record<string, string | number>;
type Point = [number, number];

interface FallState {
  r: number;
  phi: number;
  tau: number;
  t: number;
  energy: number;
  angularMomentum: number;
  radialSign: number;
}

interface Sample extends FallState {
  x: number;
  z: number;
}

function el(parent: Element, name: string, attrs: Attrs = {}): SVGElement {
  const node = document.createElementNS(NS, name) as SVGElement;
  for (const key in attrs) node.setAttribute(key, String(attrs[key]));
  parent.appendChild(node);
  return node;
}

function txt(parent: Element, x: number, y: number, value: string, attrs: Attrs = {}): SVGElement {
  const node = el(parent, 'text', {
    x,
    y,
    fill: C.dim,
    'font-size': 18,
    'font-family': 'system-ui, sans-serif',
    'text-anchor': 'middle',
    ...attrs,
  });
  node.textContent = value;
  return node;
}

function clear(svg: SVGSVGElement): void {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
}

function svgRoot(id: string): SVGSVGElement {
  return document.getElementById(id) as unknown as SVGSVGElement;
}

function pathD(points: Point[]): string {
  return points.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join('');
}

function niceTick(raw: number): number {
  const exp = Math.floor(Math.log10(Math.max(raw, 1e-6)));
  const base = raw / 10 ** exp;
  const nice = base <= 1 ? 1 : base <= 2 ? 2 : base <= 5 ? 5 : 10;
  return nice * 10 ** exp;
}

function formatTick(value: number): string {
  if (value >= 10) return value.toFixed(0);
  if (value >= 1) return value.toFixed(1);
  return value.toFixed(2);
}

function fSchwarzschild(r: number): number {
  return 1 - 1 / Math.max(r, 0.08);
}

function radialPotential(r: number, energy: number, angularMomentum: number): number {
  const f = fSchwarzschild(r);
  return energy * energy - f * (1 + angularMomentum * angularMomentum / (r * r));
}

function effectivePotential(r: number, angularMomentum: number): number {
  return fSchwarzschild(r) * (1 + angularMomentum * angularMomentum / (r * r));
}

function launch(r: number, phi: number, betaRadial: number, betaTangential: number): FallState {
  const beta2 = Math.min(betaRadial * betaRadial + betaTangential * betaTangential, 0.85 * 0.85);
  const gamma = 1 / Math.sqrt(1 - beta2);
  return {
    r,
    phi,
    tau: 0,
    t: 0,
    energy: gamma * Math.sqrt(Math.max(fSchwarzschild(r), 1e-6)),
    angularMomentum: r * gamma * betaTangential,
    radialSign: Math.abs(betaRadial) < 1e-6 ? -1 : Math.sign(betaRadial),
  };
}

function deriv(state: FallState): Pick<FallState, 'r' | 'phi' | 't'> {
  const f = Math.max(fSchwarzschild(state.r), 1e-8);
  let radialSq = radialPotential(state.r, state.energy, state.angularMomentum);
  let sign = state.radialSign;
  if (radialSq < 1e-8 && sign > 0) {
    sign = -1;
    radialSq = 0;
  } else if (radialSq < 1e-10 && sign < 0) {
    radialSq = 1e-10;
  }
  return {
    r: sign * Math.sqrt(Math.max(radialSq, 0)),
    phi: state.angularMomentum / (state.r * state.r),
    t: state.r > 1 ? state.energy / f : 0,
  };
}

function offsetState(
  state: FallState,
  k: Pick<FallState, 'r' | 'phi' | 't'>,
  h: number,
): FallState {
  return {
    ...state,
    r: Math.max(state.r + k.r * h, 0.08),
    phi: state.phi + k.phi * h,
    t: state.t + k.t * h,
    tau: state.tau + h,
  };
}

function rk4Step(state: FallState, h: number): FallState {
  const k1 = deriv(state);
  const k2 = deriv(offsetState(state, k1, h * 0.5));
  const k3 = deriv(offsetState(state, k2, h * 0.5));
  const k4 = deriv(offsetState(state, k3, h));
  const r = state.r + h / 6 * (k1.r + 2 * k2.r + 2 * k3.r + k4.r);
  const phi = state.phi + h / 6 * (k1.phi + 2 * k2.phi + 2 * k3.phi + k4.phi);
  const t = state.t + h / 6 * (k1.t + 2 * k2.t + 2 * k3.t + k4.t);
  const radialSq = radialPotential(Math.max(r, 0.08), state.energy, state.angularMomentum);
  const radialSign = radialSq < 1e-8 && state.radialSign > 0 ? -1 : state.radialSign;
  return { ...state, r: Math.max(r, 0.08), phi, t, tau: state.tau + h, radialSign };
}

function simulate(initial: FallState, maxTau = 70, h = 0.025): Sample[] {
  const samples: Sample[] = [];
  let state = { ...initial };
  const steps = Math.ceil(maxTau / h);
  for (let i = 0; i <= steps; i++) {
    if (i % 2 === 0) {
      samples.push({
        ...state,
        x: state.r * Math.cos(state.phi),
        z: state.r * Math.sin(state.phi),
      });
    }
    if (state.r <= 0.09 || state.r > 22) break;
    state = rk4Step(state, h);
  }
  return samples;
}

function drawRing(svg: SVGSVGElement, cx: number, cy: number, r: number, attrs: Attrs): void {
  el(svg, 'circle', { cx, cy, r, fill: 'none', ...attrs });
}

function drawArrow(svg: SVGSVGElement, x1: number, y1: number, x2: number, y2: number, color: string): void {
  el(svg, 'path', { d: `M${x1} ${y1}L${x2} ${y2}`, stroke: color, 'stroke-width': 2.4, fill: 'none' });
  const ang = Math.atan2(y2 - y1, x2 - x1);
  const a1 = ang + Math.PI * 0.82;
  const a2 = ang - Math.PI * 0.82;
  el(svg, 'path', {
    d: `M${x2} ${y2}L${x2 + Math.cos(a1) * 12} ${y2 + Math.sin(a1) * 12}L${x2 + Math.cos(a2) * 12} ${y2 + Math.sin(a2) * 12}Z`,
    fill: color,
  });
}

function drawLegend(svg: SVGSVGElement, items: [string, string][], x: number, y: number): void {
  items.forEach(([label, color], i) => {
    const yy = y + i * 25;
    el(svg, 'path', { d: `M${x} ${yy}L${x + 34} ${yy}`, stroke: color, 'stroke-width': 2.5 });
    txt(svg, x + 45, yy + 6, label, { fill: C.fg, 'font-size': 16, 'text-anchor': 'start' });
  });
}

// Diagram 1: top-down fall trajectories.
{
  const svg = svgRoot('svgTrajectories');
  const cx = 400;
  const cy = 280;
  const scale = 21;
  const toScreen = (sample: Sample): Point => [cx + sample.x * scale, cy - sample.z * scale];

  for (let r = 3; r <= 18; r += 3) {
    drawRing(svg, cx, cy, r * scale, { stroke: C.grid, 'stroke-width': 1 });
  }
  el(svg, 'path', { d: `M${cx - 18 * scale} ${cy}L${cx + 18 * scale} ${cy}M${cx} ${cy - 12 * scale}L${cx} ${cy + 12 * scale}`, stroke: C.grid, 'stroke-width': 1 });
  drawRing(svg, cx, cy, 3 * scale, { stroke: '#4a3f2e', 'stroke-width': 1, 'stroke-dasharray': '6 5' });
  drawRing(svg, cx, cy, 1.5 * scale, { stroke: C.dim, 'stroke-width': 1, 'stroke-dasharray': '5 5' });
  el(svg, 'circle', { cx, cy, r: scale, fill: '#000', stroke: '#3a3f4a', 'stroke-width': 1.5 });

  const launches: [number, string, string][] = [
    [0, 'radial', C.white],
    [0.24, 'beta_phi = 0.24', C.green],
    [0.46, 'beta_phi = 0.46', C.blue],
    [0.66, 'beta_phi = 0.66', C.accent],
  ];
  for (const [beta, label, color] of launches) {
    const samples = simulate(launch(12, 0, 0, beta), 85);
    el(svg, 'path', { d: pathD(samples.map(toScreen)), fill: 'none', stroke: color, 'stroke-width': beta === 0 ? 2.6 : 2, opacity: 0.9 });
    const first = samples[0];
    el(svg, 'circle', { cx: cx + first.x * scale, cy: cy - first.z * scale, r: 4.5, fill: color });
    if (label === 'radial') txt(svg, cx + first.x * scale + 28, cy - first.z * scale + 5, 'start r=12', { fill: C.dim, 'text-anchor': 'start', 'font-size': 15 });
  }
  txt(svg, cx, cy + scale + 23, 'horizon');
  txt(svg, cx + 1.9 * scale, cy - 1.5 * scale - 8, 'photon ring');
  txt(svg, cx + 3.4 * scale, cy - 3 * scale - 8, 'ISCO');
  drawLegend(svg, launches.map(([, label, color]) => [label, color]), 38, 45);
}

// Diagram 2: effective potential, interactive tangential launch speed.
{
  const svg = svgRoot('svgPotential');
  const slider = document.getElementById('launchBeta') as HTMLInputElement;
  const readout = document.getElementById('launchReadout') as HTMLOutputElement;
  const W = 800;
  const H = 500;
  const ml = 70;
  const mr = 32;
  const mt = 28;
  const mb = 58;
  const rMin = 1.01;
  const rMax = 16;
  const X = (r: number) => ml + ((r - rMin) / (rMax - rMin)) * (W - ml - mr);

  function turningPoints(energy2: number, angularMomentum: number): number[] {
    const roots: number[] = [];
    let prevR = rMin;
    let prev = effectivePotential(prevR, angularMomentum) - energy2;
    for (let i = 1; i <= 900; i++) {
      const r = rMin + (i / 900) * (rMax - rMin);
      const cur = effectivePotential(r, angularMomentum) - energy2;
      if (prev * cur < 0) {
        let lo = prevR;
        let hi = r;
        for (let j = 0; j < 42; j++) {
          const mid = (lo + hi) / 2;
          const val = effectivePotential(mid, angularMomentum) - energy2;
          if ((effectivePotential(lo, angularMomentum) - energy2) * val <= 0) hi = mid;
          else lo = mid;
        }
        roots.push((lo + hi) / 2);
      }
      prevR = r;
      prev = cur;
    }
    return roots;
  }

  function render(): void {
    clear(svg);
    const beta = parseFloat(slider.value);
    const state = launch(12, 0, 0, beta);
    const energy2 = state.energy * state.energy;
    const curveValues: [number, number][] = [];
    let maxCurveValue = energy2;
    for (let i = 0; i <= 500; i++) {
      const r = rMin + (i / 500) * (rMax - rMin);
      const value = effectivePotential(r, state.angularMomentum);
      curveValues.push([r, value]);
      maxCurveValue = Math.max(maxCurveValue, value);
    }
    const yMax = Math.max(1.25, maxCurveValue * 1.08);
    const Y = (v: number) => H - mb - (v / yMax) * (H - mt - mb);

    el(svg, 'path', { d: `M${ml} ${mt}L${ml} ${H - mb}L${W - mr} ${H - mb}`, fill: 'none', stroke: '#3a3f4a', 'stroke-width': 1 });
    for (let r = 2; r <= 16; r += 2) {
      el(svg, 'path', { d: `M${X(r)} ${H - mb}L${X(r)} ${H - mb + 6}`, stroke: '#3a3f4a', 'stroke-width': 1 });
      txt(svg, X(r), H - mb + 28, String(r), { 'font-size': 15 });
    }
    const tickStep = niceTick(yMax / 4);
    for (let v = 0; v <= yMax + tickStep * 0.25; v += tickStep) {
      const y = Y(v);
      el(svg, 'path', { d: `M${ml - 6} ${y}L${ml} ${y}M${ml} ${y}L${W - mr} ${y}`, stroke: v === 0 ? '#3a3f4a' : C.grid, 'stroke-width': 1 });
      txt(svg, ml - 24, y + 5, formatTick(v), { 'font-size': 15 });
    }
    txt(svg, (W + ml) / 2, H - 12, 'r / rs');
    txt(svg, 22, 250, 'energy', { transform: 'rotate(-90 22 250)' });

    const pts = curveValues.map(([r, value]): Point => [X(r), Y(value)]);
    el(svg, 'path', { d: pathD(pts), fill: 'none', stroke: C.fg, 'stroke-width': 2 });
    el(svg, 'path', { d: `M${ml} ${Y(energy2)}L${W - mr} ${Y(energy2)}`, stroke: C.accent, 'stroke-width': 2, 'stroke-dasharray': '7 5' });
    txt(svg, W - mr - 8, Y(energy2) - 10, 'E²', { fill: C.accent, 'text-anchor': 'end' });

    const roots = turningPoints(energy2, state.angularMomentum);
    for (const r of roots) {
      el(svg, 'circle', { cx: X(r), cy: Y(energy2), r: 6, fill: C.white });
      txt(svg, X(r), Y(energy2) + 27, `turn r=${r.toFixed(2)}`, { fill: C.dim, 'font-size': 15 });
    }
    el(svg, 'circle', { cx: X(12), cy: Y(energy2), r: 5, fill: C.accent });
    txt(svg, X(12), Y(energy2) - 18, 'launch', { fill: C.accent, 'font-size': 15 });

    const samples = simulate(state, 95, 0.03);
    const outcome = samples.some((s) => s.r <= 1.01)
      ? 'crosses the horizon'
      : samples[samples.length - 1].r > 16
        ? 'escapes this plot'
        : 'keeps orbiting in preview';
    readout.textContent = `βφ=${beta.toFixed(3)}  E=${state.energy.toFixed(3)}  L=${state.angularMomentum.toFixed(2)} - ${outcome}`;
  }

  slider.addEventListener('input', render);
  render();
}

// Diagram 3: clock disagreement.
{
  const svg = svgRoot('svgClocks');
  const W = 800;
  const H = 500;
  const ml = 72;
  const mr = 34;
  const mt = 28;
  const mb = 62;
  const samples = simulate(launch(12, 0, 0, 0), 80, 0.02).filter((s) => s.r >= 1.015);
  const tauMax = samples[samples.length - 1].tau;
  const logTMax = Math.log10(samples[samples.length - 1].t + 1);
  const X = (tau: number) => ml + (tau / tauMax) * (W - ml - mr);
  const Yr = (r: number) => H - mb - ((r - 1) / 11) * (H - mt - mb);
  const Yt = (logT: number) => H - mb - (logT / logTMax) * (H - mt - mb);

  el(svg, 'path', { d: `M${ml} ${mt}L${ml} ${H - mb}L${W - mr} ${H - mb}`, fill: 'none', stroke: '#3a3f4a', 'stroke-width': 1 });
  for (let tau = 0; tau <= tauMax; tau += 5) {
    el(svg, 'path', { d: `M${X(tau)} ${H - mb}L${X(tau)} ${H - mb + 6}`, stroke: '#3a3f4a', 'stroke-width': 1 });
    txt(svg, X(tau), H - mb + 28, tau.toFixed(0), { 'font-size': 15 });
  }
  for (let r = 2; r <= 12; r += 2) {
    el(svg, 'path', { d: `M${ml - 6} ${Yr(r)}L${ml} ${Yr(r)}M${ml} ${Yr(r)}L${W - mr} ${Yr(r)}`, stroke: C.grid, 'stroke-width': 1 });
    txt(svg, ml - 24, Yr(r) + 5, String(r), { 'font-size': 15 });
  }
  txt(svg, (W + ml) / 2, H - 14, 'proper time τ');
  txt(svg, 20, 242, 'radius r', { transform: 'rotate(-90 20 242)' });

  const rPts = samples.map((s): Point => [X(s.tau), Yr(s.r)]);
  const tPts = samples.map((s): Point => [X(s.tau), Yt(Math.log10(s.t + 1))]);
  el(svg, 'path', { d: pathD(rPts), fill: 'none', stroke: C.accent, 'stroke-width': 2.4 });
  el(svg, 'path', { d: pathD(tPts), fill: 'none', stroke: C.blue, 'stroke-width': 2.4 });
  el(svg, 'path', { d: `M${ml} ${Yr(1)}L${W - mr} ${Yr(1)}`, stroke: C.red, 'stroke-width': 1.2, 'stroke-dasharray': '6 5' });
  txt(svg, W - mr - 10, Yr(1) - 10, 'horizon r=1', { fill: C.red, 'text-anchor': 'end', 'font-size': 15 });
  drawLegend(svg, [['radius r(τ)', C.accent], ['log10(distant time + 1)', C.blue]], 450, 52);
  txt(svg, 206, 68, `horizon reached at τ ≈ ${tauMax.toFixed(1)}`, { fill: C.fg, 'font-size': 17 });
}

// Diagram 4: tetrad camera schematic.
{
  const svg = svgRoot('svgTetrad');
  const leftX = 150;
  const cy = 215;
  const rightX = 610;

  el(svg, 'rect', { x: 50, y: 72, width: 245, height: 265, rx: 6, fill: '#0e1016', stroke: C.line });
  el(svg, 'rect', { x: 505, y: 72, width: 245, height: 265, rx: 6, fill: '#0e1016', stroke: C.line });
  txt(svg, 172, 112, 'player look basis', { fill: C.fg });
  txt(svg, 628, 112, 'boosted tetrad', { fill: C.fg });

  drawArrow(svg, leftX, cy, leftX + 86, cy, C.accent);
  drawArrow(svg, leftX, cy, leftX, cy - 82, C.blue);
  drawArrow(svg, leftX, cy, leftX - 64, cy + 62, C.green);
  txt(svg, leftX + 100, cy + 7, 'forward', { fill: C.accent, 'text-anchor': 'start', 'font-size': 15 });
  txt(svg, leftX + 4, cy - 94, 'up', { fill: C.blue, 'text-anchor': 'start', 'font-size': 15 });
  txt(svg, leftX - 78, cy + 78, 'right', { fill: C.green, 'font-size': 15 });

  drawArrow(svg, 317, cy, 480, cy, C.dim);
  txt(svg, 398, cy - 18, 'Lorentz boost by β', { fill: C.dim, 'font-size': 16 });

  drawArrow(svg, rightX, cy + 54, rightX + 86, cy + 54, C.accent);
  drawArrow(svg, rightX, cy + 54, rightX + 22, cy - 55, C.blue);
  drawArrow(svg, rightX, cy + 54, rightX - 72, cy + 2, C.green);
  drawArrow(svg, rightX, cy + 54, rightX + 44, cy - 102, C.white);
  txt(svg, rightX + 97, cy + 62, 'e_forward', { fill: C.accent, 'text-anchor': 'start', 'font-size': 15 });
  txt(svg, rightX + 26, cy - 69, 'e_up', { fill: C.blue, 'text-anchor': 'start', 'font-size': 15 });
  txt(svg, rightX - 86, cy + 2, 'e_right', { fill: C.green, 'font-size': 15 });
  txt(svg, rightX + 52, cy - 114, 'e_time', { fill: C.white, 'text-anchor': 'start', 'font-size': 15 });

  el(svg, 'rect', { x: 215, y: 355, width: 370, height: 42, rx: 6, fill: '#11131a', stroke: C.line });
  txt(svg, 400, 382, 'k = e_time + x e_right + y e_up + z e_forward', { fill: C.fg, 'font-size': 17 });
}

// Diagram 5: shader ray-tracing pipeline.
{
  const svg = svgRoot('svgPipeline');
  const box = (x: number, y: number, w: number, h: number, label: string, note: string, stroke = C.line): void => {
    el(svg, 'rect', { x, y, width: w, height: h, rx: 6, fill: '#0e1016', stroke, 'stroke-width': 1.2 });
    txt(svg, x + w / 2, y + 32, label, { fill: C.fg, 'font-size': 17 });
    txt(svg, x + w / 2, y + 59, note, { fill: C.dim, 'font-size': 14 });
  };
  const arrow = (x1: number, y1: number, x2: number, y2: number): void => drawArrow(svg, x1, y1, x2, y2, C.dim);

  box(52, 165, 155, 82, 'screen ray', 'rdLocal');
  box(250, 165, 155, 82, 'tetrad', 'make photon k');
  box(448, 70, 230, 92, 'outside r > 1', 'impact b + photon potential', C.accent);
  box(448, 268, 230, 92, 'inside r < 1', 'Hamiltonian RK4 in q,p', C.blue);
  box(610, 170, 135, 80, 'samples', 'stars + disk');

  arrow(207, 206, 250, 206);
  arrow(405, 196, 448, 132);
  arrow(405, 216, 448, 314);
  arrow(678, 116, 678, 170);
  arrow(678, 268, 678, 250);
  txt(svg, 426, 176, 'branch by camera radius', { fill: C.dim, 'font-size': 14 });

  el(svg, 'path', { d: 'M92 72C180 30 270 60 340 108S482 170 608 108', fill: 'none', stroke: C.grid, 'stroke-width': 2 });
  el(svg, 'circle', { cx: 355, cy: 112, r: 28, fill: '#000', stroke: '#3a3f4a', 'stroke-width': 1.5 });
  txt(svg, 356, 64, 'backward photon trace', { fill: C.dim, 'font-size': 15 });
}
