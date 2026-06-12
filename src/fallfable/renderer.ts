// Minimal WebGPU runtime for the fallfable demo.
//
// Built for a steady requestAnimationFrame loop: render() is synchronous
// (write uniforms, encode, submit) and never reads GPU buffers back. Frame
// pacing comes from a small frames-in-flight cap, and the trace resolution
// adapts to measured GPU frame time. The trace kernel writes linear HDR into
// an rgba16float texture; the present pass upsamples it with a filtering
// sampler and tonemaps at display resolution.

import { type Tetrad, type Vec4 } from './kerr';
import { PRESENT_WGSL, SKY_ATLAS_WGSL, TRACE_WGSL } from './shader';

export interface SceneFrame {
  /** Camera coordinate time (rays inherit it, so disk rotation sees delays). */
  time: number;
  position: { x: number; y: number; z: number };
  tetrad: Tetrad;
  verticalFovRadians: number;
}

export interface RendererOptions {
  spin: number;
  mass: number;
  baseStep: number;
  escapeRadius: number;
  maxSteps: number;
  singularityCutoff: number;
  disk: {
    innerRadius: number;
    outerRadius: number;
    innerTemperature: number;
    emissivity: number;
    boostPower: number;
    spinDirection: number;
    scaleHeight: number;
    absorption: number;
    /** Texture/hotspot time runs this much faster than coordinate time. */
    animationScale: number;
    hotspotIntensity: number;
  };
  sky: {
    starIntensity: number;
    milkyWayIntensity: number;
    ambient: number;
    /** 0 normal, 1 termination, 2 cost, 3 cost+term, 4 4x4 classifier, 5 8x8 tile classifier. */
    debugStatus?: number;
  };
}

export interface RendererStats {
  gpuMs: number;
  width: number;
  height: number;
  scale: number;
  gpuTimerAvailable: boolean;
}

export interface RendererFrameTiming extends RendererStats {
  completedAt: number;
}

const UNIFORM_FLOATS = 44;
const MAX_IN_FLIGHT = 2;
const MAX_DISPLAY_WIDTH = 1920;
const TIMING_SLOT_COUNT = 4;
const TIMESTAMP_QUERY_COUNT = 2;
const TIMESTAMP_RESULT_BYTES = TIMESTAMP_QUERY_COUNT * 8;
const SKY_ATLAS_WIDTH = 1024;
const SKY_ATLAS_HEIGHT = 512;

// The flag namespaces are runtime globals the TS lib in this project does not
// declare; fall back to the spec-fixed bit values when they are absent.
type GpuFlags = Record<string, number>;
const gpuGlobals = globalThis as unknown as { GPUBufferUsage?: GpuFlags; GPUMapMode?: GpuFlags; GPUTextureUsage?: GpuFlags };
const BUFFER_USAGE: GpuFlags = gpuGlobals.GPUBufferUsage ?? {
  MAP_READ: 0x01,
  COPY_SRC: 0x04,
  COPY_DST: 0x08,
  UNIFORM: 0x40,
  QUERY_RESOLVE: 0x200,
};
const MAP_MODE: GpuFlags = gpuGlobals.GPUMapMode ?? { READ: 0x01 };
const TEXTURE_USAGE: GpuFlags = gpuGlobals.GPUTextureUsage ?? { TEXTURE_BINDING: 0x04, STORAGE_BINDING: 0x08 };

interface TimestampSlot {
  querySet: GPUQuerySet;
  resolveBuffer: GPUBuffer;
  readbackBuffer: GPUBuffer;
  busy: boolean;
}

export class FallfableRenderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private canvas: HTMLCanvasElement;
  private tracePipeline: GPUComputePipeline;
  private classifierGridPipeline: GPUComputePipeline;
  private classifierTilePipeline: GPUComputePipeline;
  private skyPipeline: GPUComputePipeline;
  private presentPipeline: GPURenderPipeline;
  private uniformBuffer: GPUBuffer;
  private presentSampler: GPUSampler;
  private skySampler: GPUSampler;
  private skyMilkyTexture: GPUTexture;
  private uniforms = new Float32Array(UNIFORM_FLOATS);

  private traceTexture: GPUTexture | null = null;
  private traceBindGroup: GPUBindGroup | null = null;
  private classifierGridBindGroup: GPUBindGroup | null = null;
  private classifierTileBindGroup: GPUBindGroup | null = null;
  private presentBindGroup: GPUBindGroup | null = null;
  private traceWidth = 0;
  private traceHeight = 0;
  private displayAspect = 16 / 9;

  private inFlight = 0;
  private gpuMsEma = Number.NaN;
  private scale = 0.6;
  private autoScale = true;
  private lastScaleAdjust = 0;
  private frameTimings: RendererFrameTiming[] = [];
  private timestampSlots: TimestampSlot[] = [];
  private nextTimestampSlot = 0;

  options: RendererOptions;

  private constructor(
    device: GPUDevice,
    context: GPUCanvasContext,
    canvas: HTMLCanvasElement,
    format: GPUTextureFormat,
    options: RendererOptions,
  ) {
    this.device = device;
    this.context = context;
    this.canvas = canvas;
    this.options = options;

    const traceModule = device.createShaderModule({ code: TRACE_WGSL });
    this.tracePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: traceModule, entryPoint: 'main' },
    });
    this.classifierGridPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: traceModule, entryPoint: 'classify_grid_main' },
    });
    this.classifierTilePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: traceModule, entryPoint: 'classify_tile_main' },
    });
    this.skyPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: SKY_ATLAS_WGSL }), entryPoint: 'main' },
    });
    const presentModule = device.createShaderModule({ code: PRESENT_WGSL });
    this.presentPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: presentModule, entryPoint: 'vertex_main' },
      fragment: { module: presentModule, entryPoint: 'fragment_main', targets: [{ format }] },
    });
    this.uniformBuffer = device.createBuffer({
      size: UNIFORM_FLOATS * 4,
      usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST,
    });
    this.presentSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.skySampler = device.createSampler({
      addressModeU: 'repeat',
      addressModeV: 'clamp-to-edge',
      magFilter: 'linear',
      minFilter: 'linear',
    });
    this.skyMilkyTexture = this.createSkyAtlasTexture('fallfable milky way atlas');
    this.generateSkyAtlas();
    if (device.features.has('timestamp-query')) {
      this.timestampSlots = Array.from({ length: TIMING_SLOT_COUNT }, () => ({
        querySet: device.createQuerySet({ type: 'timestamp', count: TIMESTAMP_QUERY_COUNT }),
        resolveBuffer: device.createBuffer({
          size: TIMESTAMP_RESULT_BYTES,
          usage: BUFFER_USAGE.QUERY_RESOLVE | BUFFER_USAGE.COPY_SRC,
        }),
        readbackBuffer: device.createBuffer({
          size: TIMESTAMP_RESULT_BYTES,
          usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
        }),
        busy: false,
      }));
    }
  }

  static async create(canvas: HTMLCanvasElement, options: RendererOptions): Promise<FallfableRenderer | null> {
    if (!navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const requiredFeatures: GPUFeatureName[] = adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
    const device = await adapter.requestDevice({ requiredFeatures });
    if (!device) return null;
    const context = canvas.getContext('webgpu') as GPUCanvasContext | null;
    if (!context) return null;
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });
    return new FallfableRenderer(device, context, canvas, format, options);
  }

  get stats(): RendererStats {
    return {
      gpuMs: Number.isFinite(this.gpuMsEma) ? this.gpuMsEma : 0,
      width: this.traceWidth,
      height: this.traceHeight,
      scale: this.scale,
      gpuTimerAvailable: this.timestampSlots.length > 0,
    };
  }

  /** 'auto' adapts resolution to GPU frame time; a number pins the scale. */
  setQuality(mode: 'auto' | number): void {
    if (mode === 'auto') {
      this.autoScale = true;
    } else {
      this.autoScale = false;
      this.scale = Math.max(0.2, Math.min(1, mode));
    }
  }

  resetFrameTimings(): void {
    this.frameTimings = [];
  }

  consumeFrameTimings(): RendererFrameTiming[] {
    const timings = this.frameTimings;
    this.frameTimings = [];
    return timings;
  }

  /** Returns false when the frame was skipped because the GPU is behind. */
  render(frame: SceneFrame): boolean {
    if (this.inFlight >= MAX_IN_FLIGHT) return false;
    this.adjustScale();
    this.ensureTargets(frame);
    if (!this.traceTexture || !this.traceBindGroup || !this.presentBindGroup) return false;

    this.packUniforms(frame);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniforms);
    const timingSlot = this.takeTimestampSlot();
    const diagnosticMode = Math.floor(this.options.sky.debugStatus ?? 0);
    const classifierBlockSize = diagnosticMode === 4 ? 4 : diagnosticMode === 5 ? 8 : 0;
    if (classifierBlockSize === 4 && !this.classifierGridBindGroup) return false;
    if (classifierBlockSize === 8 && !this.classifierTileBindGroup) return false;

    const encoder = this.device.createCommandEncoder();
    const compute = encoder.beginComputePass(timingSlot
      ? {
          timestampWrites: {
            querySet: timingSlot.querySet,
            beginningOfPassWriteIndex: 0,
          },
        }
      : undefined);
    if (classifierBlockSize > 0) {
      const classifierBindGroup = classifierBlockSize === 4 ? this.classifierGridBindGroup : this.classifierTileBindGroup;
      compute.setPipeline(classifierBlockSize === 4 ? this.classifierGridPipeline : this.classifierTilePipeline);
      compute.setBindGroup(0, classifierBindGroup);
      compute.dispatchWorkgroups(
        Math.ceil(Math.ceil(this.traceWidth / classifierBlockSize) / 8),
        Math.ceil(Math.ceil(this.traceHeight / classifierBlockSize) / 8),
      );
    } else {
      compute.setPipeline(this.tracePipeline);
      compute.setBindGroup(0, this.traceBindGroup);
      compute.dispatchWorkgroups(Math.ceil(this.traceWidth / 8), Math.ceil(this.traceHeight / 8));
    }
    compute.end();

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      timestampWrites: timingSlot
        ? {
            querySet: timingSlot.querySet,
            endOfPassWriteIndex: 1,
          }
        : undefined,
    });
    pass.setPipeline(this.presentPipeline);
    pass.setBindGroup(0, this.presentBindGroup);
    pass.draw(3);
    pass.end();
    if (timingSlot) {
      encoder.resolveQuerySet(timingSlot.querySet, 0, TIMESTAMP_QUERY_COUNT, timingSlot.resolveBuffer, 0);
      encoder.copyBufferToBuffer(timingSlot.resolveBuffer, 0, timingSlot.readbackBuffer, 0, TIMESTAMP_RESULT_BYTES);
    }

    const submittedWidth = this.traceWidth;
    const submittedHeight = this.traceHeight;
    const submittedScale = this.scale;
    this.device.queue.submit([encoder.finish()]);
    this.inFlight += 1;
    void this.device.queue.onSubmittedWorkDone().finally(() => {
      this.inFlight = Math.max(0, this.inFlight - 1);
    });
    if (timingSlot) {
      void this.readTimestampSlot(timingSlot, submittedWidth, submittedHeight, submittedScale);
    }
    return true;
  }

  private takeTimestampSlot(): TimestampSlot | null {
    if (this.timestampSlots.length === 0) return null;
    for (let tries = 0; tries < this.timestampSlots.length; tries++) {
      const index = (this.nextTimestampSlot + tries) % this.timestampSlots.length;
      const slot = this.timestampSlots[index];
      if (!slot.busy) {
        slot.busy = true;
        this.nextTimestampSlot = (index + 1) % this.timestampSlots.length;
        return slot;
      }
    }
    return null;
  }

  private async readTimestampSlot(slot: TimestampSlot, width: number, height: number, scale: number): Promise<void> {
    let mapped = false;
    try {
      await slot.readbackBuffer.mapAsync(MAP_MODE.READ, 0, TIMESTAMP_RESULT_BYTES);
      mapped = true;
      const data = new BigUint64Array(slot.readbackBuffer.getMappedRange(0, TIMESTAMP_RESULT_BYTES));
      const begin = data[0];
      const end = data[1];
      const durationNs = end > begin ? end - begin : 0n;
      const gpuMs = Number(durationNs) / 1_000_000;
      const completedAt = performance.now();
      this.gpuMsEma = Number.isFinite(this.gpuMsEma) ? this.gpuMsEma + (gpuMs - this.gpuMsEma) * 0.12 : gpuMs;
      this.frameTimings.push({
        gpuMs,
        width,
        height,
        scale,
        gpuTimerAvailable: true,
        completedAt,
      });
      if (this.frameTimings.length > 900) this.frameTimings.splice(0, this.frameTimings.length - 900);
    } catch {
      // Timer readback can fail if the device is lost or a browser revokes the
      // optional timestamp-query feature; skip that sample instead of falling
      // back to CPU wall time.
    } finally {
      if (mapped) slot.readbackBuffer.unmap();
      slot.busy = false;
    }
  }

  private adjustScale(): void {
    if (!this.autoScale || this.timestampSlots.length === 0 || !Number.isFinite(this.gpuMsEma)) return;
    const now = performance.now();
    if (now - this.lastScaleAdjust < 600) return;
    this.lastScaleAdjust = now;
    if (this.gpuMsEma > 26 && this.scale > 0.25) {
      this.scale = Math.max(0.25, this.scale * 0.85);
    } else if (this.gpuMsEma < 15 && this.scale < 1) {
      this.scale = Math.min(1, this.scale * 1.1);
    }
  }

  private ensureTargets(frame: SceneFrame): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const displayWidth = Math.min(Math.max(Math.round(this.canvas.clientWidth * dpr), 64), MAX_DISPLAY_WIDTH);
    const displayHeight = Math.max(Math.round(displayWidth * (this.canvas.clientHeight / Math.max(this.canvas.clientWidth, 1))), 64);
    if (this.canvas.width !== displayWidth || this.canvas.height !== displayHeight) {
      this.canvas.width = displayWidth;
      this.canvas.height = displayHeight;
    }
    this.displayAspect = displayWidth / displayHeight;

    const w = Math.max(64, (Math.round((displayWidth * this.scale) / 16) * 16));
    const h = Math.max(64, (Math.round((displayHeight * this.scale) / 16) * 16));
    if (w === this.traceWidth && h === this.traceHeight && this.traceTexture) return;

    this.traceTexture?.destroy();
    this.classifierGridBindGroup = null;
    this.classifierTileBindGroup = null;
    this.traceTexture = this.device.createTexture({
      size: { width: w, height: h },
      format: 'rgba16float',
      usage: TEXTURE_USAGE.STORAGE_BINDING | TEXTURE_USAGE.TEXTURE_BINDING,
    });
    this.traceWidth = w;
    this.traceHeight = h;
    const view = this.traceTexture.createView();
    const traceEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: view },
      { binding: 2, resource: this.skySampler },
      { binding: 3, resource: this.skyMilkyTexture.createView() },
    ];
    this.traceBindGroup = this.device.createBindGroup({
      layout: this.tracePipeline.getBindGroupLayout(0),
      entries: traceEntries,
    });
    this.classifierGridBindGroup = this.device.createBindGroup({
      layout: this.classifierGridPipeline.getBindGroupLayout(0),
      entries: traceEntries,
    });
    this.classifierTileBindGroup = this.device.createBindGroup({
      layout: this.classifierTilePipeline.getBindGroupLayout(0),
      entries: traceEntries,
    });
    this.presentBindGroup = this.device.createBindGroup({
      layout: this.presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: view },
        { binding: 1, resource: this.presentSampler },
      ],
    });
    void frame;
  }

  private createSkyAtlasTexture(label: string): GPUTexture {
    return this.device.createTexture({
      label,
      size: { width: SKY_ATLAS_WIDTH, height: SKY_ATLAS_HEIGHT },
      format: 'rgba16float',
      usage: TEXTURE_USAGE.STORAGE_BINDING | TEXTURE_USAGE.TEXTURE_BINDING,
    });
  }

  private generateSkyAtlas(): void {
    const bindGroup = this.device.createBindGroup({
      layout: this.skyPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.skyMilkyTexture.createView() },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.skyPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(SKY_ATLAS_WIDTH / 8), Math.ceil(SKY_ATLAS_HEIGHT / 8));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  private packUniforms(frame: SceneFrame): void {
    const o = this.options;
    const u = this.uniforms;
    u[0] = frame.time;
    u[1] = frame.position.x;
    u[2] = frame.position.y;
    u[3] = frame.position.z;
    packVec4(u, 4, frame.tetrad.eTime);
    packVec4(u, 8, frame.tetrad.eRight);
    packVec4(u, 12, frame.tetrad.eUp);
    packVec4(u, 16, frame.tetrad.eForward);
    u[20] = o.spin;
    u[21] = o.mass;
    u[22] = o.baseStep;
    u[23] = o.escapeRadius;
    u[24] = o.maxSteps;
    u[25] = o.singularityCutoff;
    u[26] = Math.tan(frame.verticalFovRadians * 0.5);
    // The display aspect, not the trace aspect: trace dimensions are rounded
    // to workgroup multiples, and using them would shift the image slightly
    // every time the adaptive resolution steps.
    u[27] = this.displayAspect;
    u[28] = o.disk.innerRadius;
    u[29] = o.disk.outerRadius;
    u[30] = o.disk.innerTemperature;
    u[31] = o.disk.emissivity;
    u[32] = o.disk.boostPower;
    u[33] = o.disk.spinDirection;
    u[34] = o.disk.scaleHeight;
    u[35] = o.disk.absorption;
    u[36] = o.sky.starIntensity;
    u[37] = o.sky.milkyWayIntensity;
    u[38] = o.sky.ambient;
    u[39] = o.sky.debugStatus ?? 0;
    u[40] = o.disk.animationScale;
    u[41] = o.disk.hotspotIntensity;
    u[42] = 0;
    u[43] = 0;
  }
}

function packVec4(target: Float32Array, offset: number, v: Vec4): void {
  target[offset] = v.t;
  target[offset + 1] = v.x;
  target[offset + 2] = v.y;
  target[offset + 3] = v.z;
}
