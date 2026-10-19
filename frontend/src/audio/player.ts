/**
 * Loudness-matched A/B player with a single synced playhead.
 *
 * Fair comparison rule: clip A (original) plays at unity gain, clip B
 * (cleaned) is scaled by 10^((L_A - L_B)/20) where L_A / L_B are the
 * input/output integrated LUFS from the backend report. The quieter clip is
 * brought up to the louder one's level so the louder clip can never "win"
 * the A/B test just by being louder. When either LUFS value is missing,
 * both clips play at unity gain (no guessing).
 */

export type ClipId = 'A' | 'B';

export interface ABPlayerOptions {
  onStateChange?: () => void;
}

export class ABPlayer {
  private ctx: AudioContext;
  private buffers: Record<ClipId, AudioBuffer>;
  private gains: Record<ClipId, number>;
  private active: ClipId = 'A';
  private source: AudioBufferSourceNode | null = null;
  private gainNode: GainNode | null = null;
  private startedAtCtxTime = 0;
  private offsetAtStart = 0;
  private opts: ABPlayerOptions;
  private disposed = false;

  /** Current playhead position in seconds (shared across A and B). */
  position = 0;
  playing = false;

  constructor(
    ctx: AudioContext,
    bufferA: AudioBuffer,
    bufferB: AudioBuffer,
    matchGainB: number | null,
    opts: ABPlayerOptions = {},
  ) {
    this.ctx = ctx;
    this.buffers = { A: bufferA, B: bufferB };
    this.gains = { A: 1, B: matchGainB ?? 1 };
    this.opts = opts;
  }

  get activeClip(): ClipId {
    return this.active;
  }

  get duration(): number {
    return Math.max(this.buffers.A.duration, this.buffers.B.duration);
  }

  /** Loudness-match gain actually applied to clip B (1 = no matching). */
  get bGain(): number {
    return this.gains.B;
  }

  private currentPosition(): number {
    if (!this.playing || !this.source) return this.position;
    const elapsed = this.ctx.currentTime - this.startedAtCtxTime;
    return this.offsetAtStart + elapsed;
  }

  private stopSource(): void {
    if (this.source) {
      try {
        this.source.onended = null;
        this.source.stop();
      } catch {
        /* already stopped */
      }
      this.source.disconnect();
      this.source = null;
    }
    if (this.gainNode) {
      this.gainNode.disconnect();
      this.gainNode = null;
    }
  }

  private startAt(offset: number, clip: ClipId): void {
    const buffer = this.buffers[clip];
    const safeOffset = Math.min(Math.max(0, offset), Math.max(0, buffer.duration - 0.05));
    this.stopSource();
    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    const g = this.ctx.createGain();
    g.gain.value = this.gains[clip];
    src.connect(g);
    g.connect(this.ctx.destination);
    src.onended = () => {
      // Natural end of the clip.
      if (this.source === src) {
        this.playing = false;
        this.position = 0;
        this.stopSource();
        this.opts.onStateChange?.();
      }
    };
    src.start(0, safeOffset);
    this.source = src;
    this.gainNode = g;
    this.startedAtCtxTime = this.ctx.currentTime;
    this.offsetAtStart = safeOffset;
    this.position = safeOffset;
    this.playing = true;
  }

  play(): void {
    if (this.disposed) return;
    void this.ctx.resume();
    if (this.playing) return;
    const pos = Math.min(this.position, this.buffers[this.active].duration - 0.05);
    this.startAt(Math.max(0, pos), this.active);
    this.opts.onStateChange?.();
  }

  pause(): void {
    if (this.disposed) return;
    if (!this.playing) return;
    this.position = Math.min(this.currentPosition(), this.buffers[this.active].duration);
    this.playing = false;
    this.stopSource();
    this.opts.onStateChange?.();
  }

  toggle(): void {
    if (this.playing) this.pause();
    else this.play();
  }

  seek(seconds: number): void {
    if (this.disposed) return;
    const t = Math.min(Math.max(0, seconds), this.duration);
    if (this.playing) {
      this.startAt(t, this.active);
    } else {
      this.position = t;
    }
    this.opts.onStateChange?.();
  }

  /** Switch clips while keeping the exact same playhead position. */
  switchTo(clip: ClipId): void {
    if (this.disposed || clip === this.active) return;
    const pos = this.playing
      ? Math.min(this.currentPosition(), this.buffers[clip].duration - 0.05)
      : Math.min(this.position, this.buffers[clip].duration - 0.05);
    this.active = clip;
    if (this.playing) {
      this.startAt(Math.max(0, pos), clip);
    } else {
      this.position = Math.max(0, pos);
    }
    this.opts.onStateChange?.();
  }

  /** Called every animation frame by the UI to advance the playhead. */
  tick(): void {
    if (this.disposed) return;
    if (this.playing) {
      const pos = this.currentPosition();
      const dur = this.buffers[this.active].duration;
      this.position = Math.min(pos, dur);
      if (pos >= dur) {
        // Safety net in case onended didn't fire.
        this.playing = false;
        this.position = 0;
        this.stopSource();
        this.opts.onStateChange?.();
      }
    }
  }

  dispose(): void {
    this.disposed = true;
    this.playing = false;
    this.stopSource();
  }
}

/**
 * Compute the loudness-match gain for clip B from measured LUFS values.
 * Returns null when matching is not possible (missing values) — the UI then
 * plays both clips at unity gain and says so honestly.
 */
export function loudnessMatchGain(
  inputLufs: number | undefined,
  outputLufs: number | undefined,
): number | null {
  if (
    typeof inputLufs !== 'number' ||
    typeof outputLufs !== 'number' ||
    !Number.isFinite(inputLufs) ||
    !Number.isFinite(outputLufs)
  ) {
    return null;
  }
  const g = Math.pow(10, (inputLufs - outputLufs) / 20);
  if (!Number.isFinite(g) || g <= 0) return null;
  // Clamp to a sane range so a broken measurement can't deafen anyone.
  return Math.min(10, Math.max(0.1, g));
}
