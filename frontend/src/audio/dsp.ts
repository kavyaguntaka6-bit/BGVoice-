/**
 * Small DSP helpers for visualisation only: peak waveforms and log-frequency
 * spectrograms rendered to canvas. Nothing here affects the cleaned audio —
 * all real processing happens in the backend pipeline.
 */

/** In-place radix-2 FFT. re/im are Float64Arrays of power-of-two length. */
export function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr;
      const ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cwr = 1;
      let cwi = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k];
        const ui = im[i + k];
        const vr = re[i + k + len / 2] * cwr - im[i + k + len / 2] * cwi;
        const vi = re[i + k + len / 2] * cwi + im[i + k + len / 2] * cwr;
        re[i + k] = ur + vr;
        im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr;
        im[i + k + len / 2] = ui - vi;
        const nwr = cwr * wr - cwi * wi;
        cwi = cwr * wi + cwi * wr;
        cwr = nwr;
      }
    }
  }
}

export interface Peaks {
  min: Float32Array;
  max: Float32Array;
}

/** Min/max peaks per horizontal pixel column. */
export function computePeaks(channel: Float32Array, columns: number): Peaks {
  const min = new Float32Array(columns);
  const max = new Float32Array(columns);
  const n = channel.length;
  const per = Math.max(1, Math.floor(n / columns));
  for (let c = 0; c < columns; c++) {
    const start = Math.min(n - 1, c * per);
    const end = Math.min(n, start + per);
    let lo = 0;
    let hi = 0;
    for (let i = start; i < end; i++) {
      const v = channel[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    min[c] = lo;
    max[c] = hi;
  }
  return { min, max };
}

export interface Spectrogram {
  /** dB values, 0..1 normalised, size cols * bins (bins = fftSize/2, row-major per column). */
  data: Float32Array;
  cols: number;
  bins: number;
  sampleRate: number;
  fftSize: number;
}

const FFT_SIZE = 2048;
const MAX_COLS = 1100;
const DB_FLOOR = -90;
const DB_CEIL = -15;

/** STFT magnitude spectrogram, mono mix of the buffer. */
export function computeSpectrogram(buffer: AudioBuffer): Spectrogram {
  const sr = buffer.sampleRate;
  const ch0 = buffer.getChannelData(0);
  const ch1 = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : null;
  const total = ch0.length;
  const cols = Math.max(32, Math.min(MAX_COLS, Math.floor(total / 512)));
  const hop = Math.max(1, Math.floor(total / cols));
  const bins = FFT_SIZE / 2;
  const data = new Float32Array(cols * bins);
  const re = new Float64Array(FFT_SIZE);
  const im = new Float64Array(FFT_SIZE);
  const win = new Float64Array(FFT_SIZE);
  for (let i = 0; i < FFT_SIZE; i++) {
    win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / FFT_SIZE); // Hann
  }
  for (let c = 0; c < cols; c++) {
    const start = Math.min(total - FFT_SIZE, c * hop);
    re.fill(0);
    im.fill(0);
    for (let i = 0; i < FFT_SIZE; i++) {
      const s0 = ch0[start + i] ?? 0;
      const s1 = ch1 ? (ch1[start + i] ?? 0) : s0;
      re[i] = 0.5 * (s0 + s1) * win[i];
    }
    fft(re, im);
    const off = c * bins;
    for (let b = 0; b < bins; b++) {
      const mag = Math.sqrt(re[b] * re[b] + im[b] * im[b]) / (FFT_SIZE / 4);
      const db = 20 * Math.log10(mag + 1e-7);
      data[off + b] = Math.min(1, Math.max(0, (db - DB_FLOOR) / (DB_CEIL - DB_FLOOR)));
    }
  }
  return { data, cols, bins, sampleRate: sr, fftSize: FFT_SIZE };
}

// --- colour map (dark -> ember -> pale) -------------------------------------

const STOPS: Array<[number, [number, number, number]]> = [
  [0.0, [9, 9, 15]],
  [0.25, [49, 32, 96]],
  [0.5, [146, 52, 96]],
  [0.75, [232, 140, 70]],
  [1.0, [250, 244, 214]],
];

function colormap(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t));
  for (let i = 1; i < STOPS.length; i++) {
    if (x <= STOPS[i][0]) {
      const [x0, c0] = STOPS[i - 1];
      const [x1, c1] = STOPS[i];
      const f = (x - x0) / (x1 - x0 || 1);
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * f),
        Math.round(c0[1] + (c1[1] - c0[1]) * f),
        Math.round(c0[2] + (c1[2] - c0[2]) * f),
      ];
    }
  }
  return STOPS[STOPS.length - 1][1];
}

// --- canvas rendering -------------------------------------------------------

const FMIN = 60; // Hz — bottom of the log axis
const FMAX = 20000; // Hz — top of the log axis

/** Render the spectrogram to an offscreen canvas with a log-frequency axis
 *  (low = bottom). Returns the canvas; callers draw it scaled as needed. */
export function renderSpectrogramBase(
  spec: Spectrogram,
  width: number,
  height: number,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width));
  canvas.height = Math.max(1, Math.round(height));
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(canvas.width, canvas.height);
  const px = img.data;
  const rows = canvas.height;
  const logMin = Math.log(FMIN);
  const logMax = Math.log(FMAX);
  for (let y = 0; y < rows; y++) {
    // y=0 (top) -> FMAX, y=rows-1 (bottom) -> FMIN
    const f = Math.exp(logMax - ((logMax - logMin) * y) / Math.max(1, rows - 1));
    const binF = (f * spec.fftSize) / spec.sampleRate;
    const b0 = Math.max(0, Math.min(spec.bins - 2, Math.floor(binF)));
    const frac = Math.min(1, Math.max(0, binF - b0));
    for (let x = 0; x < canvas.width; x++) {
      const col = Math.min(spec.cols - 1, Math.floor((x / canvas.width) * spec.cols));
      const off = col * spec.bins;
      const v = spec.data[off + b0] * (1 - frac) + spec.data[off + b0 + 1] * frac;
      const [r, g, b] = colormap(v);
      const i = (y * canvas.width + x) * 4;
      px[i] = r;
      px[i + 1] = g;
      px[i + 2] = b;
      px[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/** Render a peak waveform to an offscreen canvas. */
export function renderWaveformBase(
  peaks: Peaks,
  width: number,
  height: number,
  color: string,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width));
  canvas.height = Math.max(1, Math.round(height));
  const ctx = canvas.getContext('2d')!;
  const mid = canvas.height / 2;
  const amp = canvas.height / 2 - 2;
  ctx.fillStyle = 'rgba(0,0,0,0)';
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = color;
  const n = peaks.min.length;
  for (let x = 0; x < canvas.width; x++) {
    const c = Math.min(n - 1, Math.floor((x / canvas.width) * n));
    const y1 = mid - peaks.max[c] * amp;
    const y2 = mid - peaks.min[c] * amp;
    ctx.fillRect(x, Math.min(y1, y2), 1, Math.max(1, Math.abs(y2 - y1)));
  }
  return canvas;
}

/** Paint a pre-rendered base plus a playhead line onto a visible canvas. */
export function paintWithPlayhead(
  canvas: HTMLCanvasElement,
  base: HTMLCanvasElement,
  playheadFrac: number | null,
  playheadColor = '#34d399',
): void {
  const ctx = canvas.getContext('2d')!;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (w === 0 || h === 0) return;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(base, 0, 0, w, h);
  if (playheadFrac !== null && Number.isFinite(playheadFrac)) {
    const x = Math.min(w, Math.max(0, playheadFrac * w));
    ctx.strokeStyle = playheadColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
}

export function formatTime(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds < 0) return '0:00';
  const m = Math.floor(totalSeconds / 60);
  const s = Math.floor(totalSeconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}
