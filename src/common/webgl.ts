export function createFullscreenCanvas(cursor = 'default'): HTMLCanvasElement {
  document.body.style.margin = '0';
  document.body.style.background = '#000';
  document.body.style.overflow = 'hidden';

  const canvas = document.createElement('canvas');
  canvas.style.cssText =
    `position:fixed;inset:0;width:100%;height:100%;display:block;cursor:${cursor};`;
  document.body.appendChild(canvas);
  return canvas;
}

export function createProgram(
  gl: WebGL2RenderingContext,
  vertexSrc: string,
  fragmentSrc: string,
): WebGLProgram {
  const compile = (type: number, src: string): WebGLShader => {
    const shader = gl.createShader(type);
    if (!shader) throw new Error('shader allocation failed');
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader) ?? 'shader compile failed');
    }
    return shader;
  };

  const program = gl.createProgram();
  if (!program) throw new Error('program allocation failed');
  gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSrc));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSrc));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) ?? 'program link failed');
  }
  gl.useProgram(program);
  return program;
}

export function uniforms(gl: WebGL2RenderingContext, program: WebGLProgram) {
  return (name: string): WebGLUniformLocation | null => gl.getUniformLocation(program, name);
}

export function resizeCanvasToDisplaySize(
  gl: WebGL2RenderingContext,
  canvas: HTMLCanvasElement,
  maxDpr: number,
): void {
  const dpr = Math.min(window.devicePixelRatio || 1, maxDpr);
  const width = Math.round(canvas.clientWidth * dpr);
  const height = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    gl.viewport(0, 0, width, height);
  }
}
