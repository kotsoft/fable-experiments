interface OrbitOptions {
  yaw: number;
  pitch: number;
  dist: number;
  minDist: number;
  maxDist: number;
  pitchMin?: number;
  pitchMax?: number;
  dragScale?: number;
  wheelScale?: number;
  autoYawSpeed?: number;
}

export interface OrbitState {
  yaw: number;
  pitch: number;
  dist: number;
  dragging: boolean;
}

export function createOrbitControls(
  canvas: HTMLCanvasElement,
  options: OrbitOptions,
): OrbitState {
  const state: OrbitState = {
    yaw: options.yaw,
    pitch: options.pitch,
    dist: options.dist,
    dragging: false,
  };
  const pitchMin = options.pitchMin ?? -1.45;
  const pitchMax = options.pitchMax ?? 1.45;
  const dragScale = options.dragScale ?? 0.005;
  const wheelScale = options.wheelScale ?? 0.001;
  let lastX = 0;
  let lastY = 0;

  canvas.addEventListener('pointerdown', (event) => {
    state.dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
    canvas.style.cursor = 'grabbing';
  });

  canvas.addEventListener('pointermove', (event) => {
    if (!state.dragging) return;
    state.yaw -= (event.clientX - lastX) * dragScale;
    state.pitch += (event.clientY - lastY) * dragScale;
    state.pitch = Math.max(pitchMin, Math.min(pitchMax, state.pitch));
    lastX = event.clientX;
    lastY = event.clientY;
  });

  const stopDragging = (event?: PointerEvent) => {
    state.dragging = false;
    if (event && canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
    canvas.style.cursor = 'grab';
  };

  canvas.addEventListener('pointerup', stopDragging);
  canvas.addEventListener('pointercancel', stopDragging);
  canvas.addEventListener('lostpointercapture', () => {
    state.dragging = false;
    canvas.style.cursor = 'grab';
  });

  canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    state.dist *= Math.exp(event.deltaY * wheelScale);
    state.dist = Math.max(options.minDist, Math.min(options.maxDist, state.dist));
  }, { passive: false });

  return state;
}

export function advanceAutoOrbit(state: OrbitState, speed = 0.00045): void {
  if (!state.dragging) state.yaw += speed;
}
