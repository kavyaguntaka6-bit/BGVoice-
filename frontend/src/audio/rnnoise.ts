/**
 * Loader for the genuine RNNoise neural network (Xiph.Org), Jitsi WebAssembly
 * build, via its Emscripten glue script (public/assets/rnnoise.js +
 * public/assets/rnnoise.wasm). Frames are 480 float32 samples = 10 ms at
 * 48 kHz mono — RNNoise's native frame size.
 */

export const RNNOISE_FRAME = 480;
export const RNNOISE_SAMPLE_RATE = 48000;

interface EmscriptenModule {
  _rnnoise_create(vadProb: number): number;
  _rnnoise_process_frame(state: number, outPtr: number, inPtr: number): number;
  _rnnoise_destroy(state: number): void;
  _malloc(bytes: number): number;
  _free(ptr: number): void;
  HEAPF32: Float32Array;
  ready: Promise<unknown>;
}

declare global {
  interface Window {
    createRNNWasmModule?: (overrides?: Record<string, unknown>) => Promise<EmscriptenModule>;
  }
}

let glueLoad: Promise<void> | null = null;

function baseUrl(): string {
  const b = import.meta.env.BASE_URL as string | undefined;
  return b ?? '/';
}

function loadGlueScript(): Promise<void> {
  if (glueLoad) return glueLoad;
  glueLoad = new Promise<void>((resolve, reject) => {
    if (typeof window.createRNNWasmModule === 'function') {
      resolve();
      return;
    }
    const script = document.createElement('script');
    script.src = `${baseUrl()}assets/rnnoise.js`;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () =>
      reject(new Error('Could not load the RNNoise engine script (assets/rnnoise.js).'));
    document.head.appendChild(script);
  });
  return glueLoad;
}

export interface RNNoiseInstance {
  /** Process one 480-sample frame; returns a new Float32Array(480). */
  processFrame(frame: Float32Array): Float32Array;
  /** Last voice-activity probability returned by the network (0..1). */
  lastVad: number;
  destroy(): void;
}

/** Instantiate a real RNNoise denoiser. Throws with a clear message on failure. */
export async function createRNNoise(): Promise<RNNoiseInstance> {
  await loadGlueScript();
  const factory = window.createRNNWasmModule;
  if (typeof factory !== 'function') {
    throw new Error('The RNNoise engine script loaded but did not expose its factory.');
  }
  let wasmBinary: ArrayBuffer;
  try {
    // The 112KB wasm ships as two parts (hosting upload limits); reassemble.
    const [pa, pb] = await Promise.all(
      ['rnnoise.wasm.p1', 'rnnoise.wasm.p2'].map(async (f) => {
        const res = await fetch(`${baseUrl()}assets/${f}`);
        if (!res.ok) throw new Error(`HTTP ${res.status} for ${f}`);
        return res.arrayBuffer();
      }),
    );
    const out = new Uint8Array(pa.byteLength + pb.byteLength);
    out.set(new Uint8Array(pa), 0);
    out.set(new Uint8Array(pb), pa.byteLength);
    wasmBinary = out.buffer;
  } catch (e) {
    throw new Error(
      `Could not fetch the RNNoise model (assets/rnnoise.wasm.p1/p2): ${e instanceof Error ? e.message : 'network error'}`,
    );
  }
  let mod: EmscriptenModule;
  try {
    mod = await factory({ wasmBinary });
    await mod.ready;
  } catch (e) {
    throw new Error(
      `The RNNoise engine failed to start: ${e instanceof Error ? e.message : 'unknown error'}`,
    );
  }
  for (const fn of ['_rnnoise_create', '_rnnoise_process_frame', '_rnnoise_destroy', '_malloc', '_free'] as const) {
    if (typeof mod[fn] !== 'function') {
      throw new Error(`The RNNoise engine is missing its native entry point ${fn}.`);
    }
  }
  const state = mod._rnnoise_create(0);
  if (!state) throw new Error('RNNoise could not allocate its internal state.');
  const inPtr = mod._malloc(RNNOISE_FRAME * 4);
  const outPtr = mod._malloc(RNNOISE_FRAME * 4);
  if (!inPtr || !outPtr) {
    mod._rnnoise_destroy(state);
    throw new Error('RNNoise could not allocate its audio buffers.');
  }
  let destroyed = false;
  const instance: RNNoiseInstance = {
    lastVad: 0,
    processFrame(frame: Float32Array): Float32Array {
      if (destroyed) throw new Error('RNNoise instance already destroyed.');
      if (frame.length !== RNNOISE_FRAME) {
        throw new Error(`RNNoise needs exactly ${RNNOISE_FRAME} samples per frame.`);
      }
      // Fresh views each call — safe even if the wasm memory ever grows.
      new Float32Array(mod.HEAPF32.buffer, inPtr, RNNOISE_FRAME).set(frame);
      const vad = mod._rnnoise_process_frame(state, outPtr, inPtr);
      instance.lastVad = Math.min(1, Math.max(0, vad));
      return new Float32Array(
        new Float32Array(mod.HEAPF32.buffer, outPtr, RNNOISE_FRAME),
      );
    },
    destroy(): void {
      if (destroyed) return;
      destroyed = true;
      try {
        mod._free(inPtr);
        mod._free(outPtr);
        mod._rnnoise_destroy(state);
      } catch {
        /* best effort */
      }
    },
  };
  return instance;
}
