import { useEffect, useRef, useState } from 'react';
import { formatTime } from '../audio/dsp';

/**
 * Microphone recorder. Requests the raw signal — echo cancellation, noise
 * suppression and auto gain are all disabled so the browser does not
 * pre-damage the audio before VoiceClean processes it.
 */
const MIC_CONSTRAINTS: MediaTrackConstraints = {
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: false,
  channelCount: 1,
  sampleRate: 48000,
};

interface RecorderProps {
  onRecording: (file: File) => void;
  disabled?: boolean;
}

export default function Recorder({ onRecording, disabled }: RecorderProps) {
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [level, setLevel] = useState(0);
  const mediaRef = useRef<{
    stream: MediaStream;
    recorder: MediaRecorder;
    chunks: Blob[];
    raf: number;
    timer: number;
    analyser: AnalyserNode;
    ctx: AudioContext;
    data: Uint8Array;
  } | null>(null);

  const cleanup = () => {
    const m = mediaRef.current;
    if (!m) return;
    cancelAnimationFrame(m.raf);
    window.clearInterval(m.timer);
    m.stream.getTracks().forEach((t) => t.stop());
    void m.ctx.close().catch(() => undefined);
    mediaRef.current = null;
  };

  useEffect(() => cleanup, []);

  const start = async () => {
    setError(null);
    if (recording || disabled) return;
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: MIC_CONSTRAINTS });
    } catch (e) {
      setError(
        e instanceof DOMException && e.name === 'NotAllowedError'
          ? 'Microphone access was denied. Allow it in your browser to record.'
          : `Could not open the microphone: ${e instanceof Error ? e.message : 'unknown error'}`,
      );
      return;
    }

    const mimeCandidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/mp4',
    ];
    const mimeType = mimeCandidates.find((m) => MediaRecorder.isTypeSupported(m));
    let recorder: MediaRecorder;
    try {
      recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    } catch (e) {
      stream.getTracks().forEach((t) => t.stop());
      setError(`This browser can't record audio: ${e instanceof Error ? e.message : 'unknown error'}`);
      return;
    }

    const chunks: Blob[] = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };
    recorder.onstop = () => {
      const type = recorder.mimeType || 'audio/webm';
      const blob = new Blob(chunks, { type });
      const ext = type.includes('mp4') ? 'm4a' : 'webm';
      const file = new File(
        [blob],
        `voiceclean-recording-${new Date().toISOString().replace(/[:.]/g, '-')}.${ext}`,
        { type },
      );
      cleanup();
      setRecording(false);
      setElapsed(0);
      setLevel(0);
      onRecording(file);
    };

    // Live level meter (display only — does not touch the recorded signal).
    const ctx = new AudioContext();
    const src = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    src.connect(analyser);
    const data = new Uint8Array(analyser.fftSize);

    mediaRef.current = { stream, recorder, chunks, raf: 0, timer: 0, analyser, ctx, data };

    const tickMeter = () => {
      const m = mediaRef.current;
      if (!m) return;
      m.analyser.getByteTimeDomainData(m.data);
      let peak = 0;
      for (let i = 0; i < m.data.length; i++) {
        const v = Math.abs(m.data[i] - 128) / 128;
        if (v > peak) peak = v;
      }
      setLevel(peak);
      m.raf = requestAnimationFrame(tickMeter);
    };
    const t0 = Date.now();
    mediaRef.current.timer = window.setInterval(() => {
      setElapsed((Date.now() - t0) / 1000);
    }, 250);
    mediaRef.current.raf = requestAnimationFrame(tickMeter);

    recorder.start(250);
    setRecording(true);
  };

  const stop = () => {
    const m = mediaRef.current;
    if (m && m.recorder.state !== 'inactive') {
      m.recorder.stop();
    }
  };

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="font-medium text-zinc-200">Record with your mic</p>
          <p className="text-xs text-zinc-500 mt-0.5">
            Raw capture — no browser noise suppression or auto-gain.
          </p>
        </div>
        {!recording ? (
          <button
            type="button"
            onClick={start}
            disabled={disabled}
            className="shrink-0 rounded-full bg-red-500/15 border border-red-500/40 text-red-300 px-4 py-2 text-sm font-medium hover:bg-red-500/25 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            <span className="w-2.5 h-2.5 rounded-full bg-red-500" aria-hidden="true" />
            Record
          </button>
        ) : (
          <button
            type="button"
            onClick={stop}
            className="shrink-0 rounded-full bg-zinc-100 text-zinc-900 px-4 py-2 text-sm font-medium hover:bg-white flex items-center gap-2"
            aria-label="Stop recording"
          >
            <span className="w-2.5 h-2.5 bg-zinc-900" aria-hidden="true" />
            Stop · {formatTime(elapsed)}
          </button>
        )}
      </div>

      {/* level meter */}
      <div
        className="mt-3 h-2 rounded-full bg-zinc-800 overflow-hidden"
        role="meter"
        aria-label="Microphone level"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(level * 100)}
      >
        <div
          className="h-full bg-emerald-400 transition-[width] duration-75"
          style={{ width: `${Math.min(100, level * 100)}%` }}
        />
      </div>

      {recording && (
        <p className="mt-2 text-xs text-red-300 flex items-center gap-1.5" role="status">
          <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" aria-hidden="true" />
          Recording… press Stop when you're done.
        </p>
      )}
      {error && (
        <p role="alert" className="mt-2 text-sm text-red-400">
          {error}
        </p>
      )}
    </div>
  );
}
