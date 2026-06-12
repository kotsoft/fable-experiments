// Minimal WebGPU runtime for the fallfable demo.
//
// Built for a steady requestAnimationFrame loop: render() is synchronous
// (write uniforms, encode, submit) and never reads GPU buffers back. Frame
// pacing comes from a small frames-in-flight cap, and the trace resolution
// adapts to measured GPU frame time. The trace kernel writes linear HDR into
// an rgba16float texture; the present pass upsamples it with a filtering
// sampler and tonemaps at display resolution.

import { type Tetrad, type Vec4 } from './kerr';
import { PRESENT_WGSL, TRACE_WGSL } from './shader';

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
  };
  sky: {
    starIntensity: number;
    milkyWayIntensity: number;
    ambient: number;
    /** 1 = color-code ray termination branches instead of shading. */
    debugStatus?: number;
  };
}

export interface RendererStats {
  gpuMs: number;
  width: number;
  height: number;
  scale: number;
}

const UNIFORM_FLOATS = 40;
const MAX_IN_FLIGHT = 2;
const MAX_DISPLAY_WIDTH = 1920;

// The flag namespaces are runtime globals the TS lib in this project does not
// declare; fall back to the spec-fixed bit values when they are absent.
type GpuFlags = Record<string, number>;
const gpuGlobals = globalThis as unknown as { GPUBufferUsage?: GpuFlags; GPUTextureUsage?: GpuFlags };
const BUFFER_USAGE: GpuFlags = gpuGlobals.GPUBufferUsage ?? { UNIFORM: 0x40, COPY_DST: 0x08 };
const TEXTURE_USAGE: GpuFlags = gpuGlobals.GPUTextureUsage ?? { TEXTURE_BINDING: 0x04, STORAGE_BINDING: 0x08 };

export class FallfableRenderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private canvas: HTMLCanvasElement;
  private tracePipeline: GPUComputePipeline;
  private presentPipeline: GPURenderPipeline;
  private uniformBuffer: GPUBuffer;
  private sampler: GPUSampler;
  private uniforms = new Float32Array(UNIFORM_FLOATS);

  private traceTexture: GPUTexture | null = null;
  private traceBindGroup: GPUBindGroup | null = null;
  private presentBindGroup: GPUBindGroup | null = null;
  private traceWidth = 0;
  private traceHeight = 0;
  private displayAspect = 16 / 9;

  private inFlight = 0;
  private gpuMsEma = 16;
  private scale = 0.6;
  private autoScale = true;
  private lastScaleAdjust = 0;

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

    this.tracePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: TRACE_WGSL }), entryPoint: 'main' },
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
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  }

  static async create(canvas: HTMLCanvasElement, options: RendererOptions): Promise<FallfableRenderer | null> {
    if (!navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter?.requestDevice();
    if (!device) return null;
    const context = canvas.getContext('webgpu') as GPUCanvasContext | null;
    if (!context) return null;
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });
    return new FallfableRenderer(device, context, canvas, format, options);
  }

  get stats(): RendererStats {
    return { gpuMs: this.gpuMsEma, width: this.traceWidth, height: this.traceHeight, scale: this.scale };
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

  /** Returns false when the frame was skipped because the GPU is behind. */
  render(frame: SceneFrame): boolean {
    if (this.inFlight >= MAX_IN_FLIGHT) return false;
    this.adjustScale();
    this.ensureTargets(frame);
    if (!this.traceTexture || !this.traceBindGroup || !this.presentBindGroup) return false;

    this.packUniforms(frame);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniforms);

    const encoder = this.device.createCommandEncoder();
    const compute = encoder.beginComputePass();
    compute.setPipeline(this.tracePipeline);
    compute.setBindGroup(0, this.traceBindGroup);
    compute.dispatchWorkgroups(Math.ceil(this.traceWidth / 8), Math.ceil(this.traceHeight / 8));
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
    });
    pass.setPipeline(this.presentPipeline);
    pass.setBindGroup(0, this.presentBindGroup);
    pass.draw(3);
    pass.end();

    const submitted = performance.now();
    this.device.queue.submit([encoder.finish()]);
    this.inFlight += 1;
    void this.device.queue.onSubmittedWorkDone().then(() => {
      this.inFlight -= 1;
      const ms = performance.now() - submitted;
      this.gpuMsEma += (ms - this.gpuMsEma) * 0.12;
    });
    return true;
  }

  private adjustScale(): void {
    if (!this.autoScale) return;
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
    this.traceTexture = this.device.createTexture({
      size: { width: w, height: h },
      format: 'rgba16float',
      usage: TEXTURE_USAGE.STORAGE_BINDING | TEXTURE_USAGE.TEXTURE_BINDING,
    });
    this.traceWidth = w;
    this.traceHeight = h;
    const view = this.traceTexture.createView();
    this.traceBindGroup = this.device.createBindGroup({
      layout: this.tracePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: view },
      ],
    });
    this.presentBindGroup = this.device.createBindGroup({
      layout: this.presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: view },
        { binding: 1, resource: this.sampler },
      ],
    });
    void frame;
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
  }
}

function packVec4(target: Float32Array, offset: number, v: Vec4): void {
  target[offset] = v.t;
  target[offset + 1] = v.x;
  target[offset + 2] = v.y;
  target[offset + 3] = v.z;
}
