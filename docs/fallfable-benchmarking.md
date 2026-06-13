# Fallfable Benchmarking

Fallfable exposes a small benchmark API for measuring renderer changes against
repeatable camera locations. The benchmark uses WebGPU timestamp queries when
the browser supports them; it does not use `performance.now()` as the GPU time.

## Console API

Open the Fallfable page and use the browser console:

```js
window.__fallfable.benchmarkPoints()
```

Run one point:

```js
await window.__fallfable.benchmarkPoint("horizon-graze", {
  quality: 0.75,
  warmupFrames: 10,
  sampleFrames: 60,
})
```

Run the whole suite:

```js
await window.__fallfable.benchmarkSuite({
  quality: 0.75,
  warmupFrames: 10,
  sampleFrames: 60,
})
```

Run a subset:

```js
await window.__fallfable.benchmarkSuite({
  points: ["outer-disk", "disk-graze", "horizon-graze"],
  quality: 0.75,
  warmupFrames: 10,
  sampleFrames: 60,
})
```

## Hash Bridge

In Vite dev mode, automation can avoid clicking overlay controls by navigating
to hash commands. Results are written as JSON to:

```js
document.querySelector("#fallfable-dev-result").textContent
```

Examples:

```text
/fallfable/#fallfable=benchPoint&point=horizon-graze&quality=0.75&warmup=10&samples=60
/fallfable/#fallfable=benchSuite&points=outer-disk,disk-graze,horizon-graze&quality=0.75&warmup=10&samples=60
/fallfable/#fallfable=hunt&quality=0.75&minDelay=1000&timeout=25000
```

Supported benchmark parameters:

- `quality`: render scale from `0.2` to `1`.
- `warmup` or `warmupFrames`: frames discarded before sampling.
- `samples` or `sampleFrames`: sampled frames to summarize.
- `timeout` or `timeoutMs`: maximum time to wait for sampled frames.
- `restoreView`: `true` or `false`; defaults to restoring the previous camera.
- `restoreQuality`: `true` or `false`.
- `restoreRunning`: `true` or `false`.

## Benchmark Points

| Id | What it stresses |
| --- | --- |
| `outer-disk` | Wide disk and sky view from above the outer disk. |
| `disk-graze` | Low-angle rays through disk atmosphere, absorption, and turbulence. |
| `adaptive-boundary` | Adaptive boundary tracking where grid resolution adapts to steep gradients. |
| `isco-lens` | Marginally stable orbit with strong near-disk lensing. |
| `photon-whirl` | Near photon-orbit view where many rays skim before escaping. |
| `horizon-graze` | Just outside the outer horizon, aimed across the disk. |
| `inner-horizon` | Inside the black hole near the Cauchy horizon stress region. |
| `polar-halo` | High-axis view where the disk forms a lensed halo. |

## Recommended Runs

For quick iteration:

```js
await window.__fallfable.benchmarkSuite({
  quality: 0.5,
  warmupFrames: 5,
  sampleFrames: 20,
})
```

For before/after comparisons:

```js
await window.__fallfable.benchmarkSuite({
  quality: 0.75,
  warmupFrames: 20,
  sampleFrames: 100,
})
```

Compare `medianGpuMs`, `p95GpuMs`, and `gpuFramesPerSecond` first. Use
`completedFramesPerSecond` as a sanity check for browser scheduling overhead.
