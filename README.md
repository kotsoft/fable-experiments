# Experiments with Fable

A growing collection of small, self-contained graphics experiments — each one
a live demo plus a tutorial page explaining the math behind it. First up: a
physically-based black hole renderer in the style of Interstellar's Gargantua.

![Black hole renderer](screenshot.jpg)

## Black hole renderer

A single WebGL2 fragment shader ray-traces null geodesics through the
Schwarzschild metric — the shadow, the photon ring, and the accretion disk
arcing over the hole all emerge from the integration rather than being painted
in. No textures, no geometry, no libraries.

- **Exact lensing** — each pixel integrates the Cartesian form of the Binet
  equation, `a = -(3/2) rs h² x / r⁵`, with velocity Verlet and adaptive steps
- **Volumetric accretion disk** from the ISCO outward: Shakura–Sunyaev
  temperature profile, blackbody emission, turbulence sheared into spiral
  streaks by differential Keplerian rotation
- **Relativistic light transport** — gravitational redshift, Doppler shift,
  and g³ beaming (the asymmetry Interstellar famously toned down is left on)
- **Lensed starfield** — escaped rays sample a procedural sky with their bent
  direction

Drag to orbit, scroll to zoom. Runs at interactive rates at high resolution on
a reasonable GPU.

### Tutorial

`/blackhole/tutorial.html` is a write-up of the math behind the renderer, with
2D diagrams computed live using the same geodesic integrator as the shader —
including an interactive impact-parameter slider showing photon capture at
b = 3√3/2 rs. Expandable primers explain geodesics, Doppler beaming, and the
curved-space-to-vector-math translation in plain language.

## Running

```sh
npm install
npm run dev
```

The root URL lists the experiments; each experiment lives at its own path
(`/blackhole/`, `/kerr/`) with a `tutorial.html` beside it.

## Spinning (Kerr) black hole

`/kerr/` renders a rotating black hole — no force-law shortcut exists for
Kerr, so the shader integrates the actual geodesics: Hamiltonian form in
Cartesian Kerr-Schild coordinates, RK4 with numerical gradients. Frame
dragging, the D-shaped shadow, and the spin-dependent ISCO all emerge, with a
slider to sweep the spin from 0 to 0.998. Its tutorial's diagrams are
dynamically generated SVG, and the integrator is validated against Bardeen's
exact critical impact parameters in
[`scripts/validate-kerr.mjs`](scripts/validate-kerr.mjs).

## Code layout

- [`index.html`](index.html) — the experiment list
- [`blackhole/`](blackhole/), [`kerr/`](kerr/) — demo + tutorial pages
- [`src/blackhole/`](src/blackhole/), [`src/kerr/`](src/kerr/) — renderers
  (GLSL with physics notes, WebGL2 setup, orbit camera)
- [`src/tutorial/`](src/tutorial/), [`src/kerr-tutorial/`](src/kerr-tutorial/)
  — the tutorials' computed diagrams (canvas and SVG respectively)
- [`scripts/validate-kerr.mjs`](scripts/validate-kerr.mjs) — physics
  validation for the Kerr integrator

## License

[MIT](LICENSE)
