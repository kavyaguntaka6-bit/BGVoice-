/**
 * AudioWorklet processor source for the Live preview tab.
 *
 * The worklet itself does NO denoising — it is a low-latency courier:
 *  - every 128-sample input quantum is shipped to the main thread,
 *  - processed 480-sample frames come back over the port and are queued,
 *  - output is drained from the queue (silence on underrun, counted honestly).
 *
 * All RNNoise inference runs on the main thread via the real WASM module
 * (see audio/rnnoise.ts). Loaded via a Blob URL so no extra file routing is needed.
 */

export const PREVIEW_PROCESSOR_NAME = 'voiceclean-preview-processor';

export const PREVIEW_PROCESSOR_SOURCE = `
class VoiceCleanPreviewProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];        // Float32Array chunks of processed audio
    this.queuedSamples = 0;
    this.underruns = 0;
    this.port.onmessage = (e) => {
      const m = e.data;
      if (m && m.type === 'processed' && m.samples) {
        const chunk = new Float32Array(m.samples);
        this.queue.push(chunk);
        this.queuedSamples += chunk.length;
      } else if (m && m.type === 'clear') {
        this.queue = [];
        this.queuedSamples = 0;
      }
    };
  }

  process(inputs, outputs) {
    const input = inputs[0] && inputs[0][0];
    const output = outputs[0] && outputs[0][0];

    if (input) {
      // Ship a copy to the main thread for RNNoise processing.
      const copy = new Float32Array(input.length);
      copy.set(input);
      this.port.postMessage({ type: 'input', samples: copy }, [copy.buffer]);
    }

    if (output) {
      const n = output.length;
      let filled = 0;
      while (filled < n && this.queue.length > 0) {
        const head = this.queue[0];
        const take = Math.min(head.length, n - filled);
        output.set(head.subarray(0, take), filled);
        filled += take;
        if (take < head.length) {
          this.queue[0] = head.subarray(take);
        } else {
          this.queue.shift();
        }
        this.queuedSamples -= take;
      }
      if (filled < n) {
        output.fill(0, filled);
        this.underruns += 1;
        if (this.underruns % 200 === 1) {
          this.port.postMessage({ type: 'underrun', count: this.underruns });
        }
      }
    }
    return true;
  }
}

registerProcessor('${PREVIEW_PROCESSOR_NAME}', VoiceCleanPreviewProcessor);
`;
