import { useEffect, useRef, useState, type Ref } from 'react';
import {
  createRNNoise,
  RNNOISE_FRAME,
  RNNOISE_SAMPLE_RATE,
  type RNNoiseInstance,
} from '../audio/rnnoise';
import {
  PREVIEW_PROCESSOR_NAME,
  PREVIEW_PROCESSOR_SOURCE,
} from '../audio/previewWorklet';

/**
 * Live preview tab: your microphone runs through the genuine RNNoise neural
 * network (Xiph.Org, Jitsi WebAssembly build) in real time.
 *
 * Signal path: mic (raw capture, no browser pre-processing) -> AudioWorklet
 * (courier only) -> main thread RNNoise inference, 480-sample frames ->
 * AudioWorklet output queue -> speakers. Nothing is faked: if the WASM
 * engine fails to load, the tab says so instead of pretending.
 */

const MIC_CONSTRAINTS: MediaTrackConstraints = {
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: false,
  channelCount: 1,
  sampleRate: 48000,
};

type LiveState = 'idle' | 'starting' | 'live' | 'error';

interface Session {
  stream: MediaStream;
  ctx: AudioContext;
  node: AudioWorkletNode;
  rnnoise: RNNoiseInstance;
  blobUrl: string;
  pending: Float32Array;
  pendingFill: number;
  raf: number;
  bypass: boolean;
}

export default function LivePreview() {
  const [state, setState] = useState<LiveState>('idle');
  const [status, setStatus] = useState(
    'Hear what RNNoise does to your mic, live — before you clean a file.',
  );
  const [bypass, setBypass] = useState(false);
  const [underruns, setUnderruns] = useState(0);
  const sessionRef = useRef<Session | null>(null);
  const bypassRef = useRef(false);

  const inBarRef = useRef<HTMLDivElement>(null);
  const outBarRef = useRef<HTMLDivElement>(null);
  const vadBarRef = useRef<HTMLDivElement>(null);
  const inAnalyserRef = useRef<AnalyserNode | null>(null);
  const outAnalyserRef = useRef<AnalyserNode | null>(null);

  useEffect(() => {
    bypassRef.current = bypass;
  }, [bypass]);

  useEffect(() => {
    // Safety: tear down if the component unmounts mid-session.
    return () => {
      void stopSession();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stopSession = async () => {
    const s = sessionRef.current;
    sessionRef.current = null;
    if (!s) return;
    cancelAnimationFrame(s.raf);
    try {
      s.node.port.postMessage({ type: 'clear' });
      s.node.disconnect();
    } catch { /* ignore */ }
    s.stream.getTracks().forEach((t) => t.stop());
    s.rnnoise.destroy();
    URL.revokeObjectURL(s.blobUrl);
    try {
      await s.ctx.close();
    } catch { /* ignore */ }
    inAnalyserRef.current = null;
    outAnalyserRef.current = null;
  };

  const start = async () => {
    if (sessionRef.current) return;
    setState('starting');
    setUnderruns(0);
    setStatus('Loading the RNNoise engine…');

    let rnnoise: RNNoiseInstance;
    try {
      rnnoise = await createRNNoise();
    } catch (e) {
      setState('error');
      setStatus(e instanceof Error ? e.message : 'Could not start the RNNoise engine.');
      return;
    }

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: MIC_CONSTRAINTS });
    } catch (e) {
      rnnoise.destroy();
      setState('error');
      setStatus(
        e instanceof DOMException && e.name === 'NotAllowedError'
          ? 'Microphone access was denied. Allow it in your browser to use live preview.'
          : `Could not open the microphone: ${e instanceof Error ? e.message : 'unknown error'}`,
      );
      return;
    }

    let ctx: AudioContext;
    try {
      ctx = new AudioContext({ sampleRate: RNNOISE_SAMPLE_RATE, latencyHint: 'interactive' });
    } catch (e) {
      stream.getTracks().forEach((t) => t.stop());
      rnnoise.destroy();
      setState('error');
      setStatus(`Could not create audio output: ${e instanceof Error ? e.message : 'unknown error'}`);
      return;
    }

    if (ctx.sampleRate !== RNNOISE_SAMPLE_RATE) {
      // RNNoise only accepts 48 kHz. Refuse rather than resample badly.
      stream.getTracks().forEach((t) => t.stop());
      rnnoise.destroy();
      void ctx.close().catch(() => undefined);
      setState('error');
      setStatus(
        `Live preview needs 48 kHz audio, but this device runs at ${ctx.sampleRate} Hz. ` +
          'Preview is disabled rather than run at the wrong rate.',
      );
      return;
    }

    const blobUrl = URL.createObjectURL(
      new Blob([PREVIEW_PROCESSOR_SOURCE], { type: 'application/javascript' }),
    );
    try {
      await ctx.audioWorklet.addModule(blobUrl);
    } catch (e) {
      URL.revokeObjectURL(blobUrl);
      stream.getTracks().forEach((t) => t.stop());
      rnnoise.destroy();
      void ctx.close().catch(() => undefined);
      setState('error');
      setStatus(`Could not start the audio worker: ${e instanceof Error ? e.message : 'unknown error'}`);
      return;
    }

    const node = new AudioWorkletNode(ctx, PREVIEW_PROCESSOR_NAME, {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    const source = ctx.createMediaStreamSource(stream);
    const inAnalyser = ctx.createAnalyser();
    inAnalyser.fftSize = 512;
    const outAnalyser = ctx.createAnalyser();
    outAnalyser.fftSize = 512;
    source.connect(inAnalyser);
    source.connect(node);
    node.connect(outAnalyser);
    outAnalyser.connect(ctx.destination);
    inAnalyserRef.current = inAnalyser;
    outAnalyserRef.current = outAnalyser;

    const session: Session = {
      stream,
      ctx,
      node,
      rnnoise,
      blobUrl,
      pending: new Float32Array(RNNOISE_FRAME),
      pendingFill: 0,
      raf: 0,
      bypass: false,
    };
    sessionRef.current = session;

    const inData = new Uint8Array(inAnalyser.fftSize);
    const outData = new Uint8Array(outAnalyser.fftSize);

    node.port.onmessage = (e: MessageEvent) => {
      const s = sessionRef.current;
      const m = e.data as { type?: string; samples?: Float32Array; count?: number };
      if (!s) return;
      if (m?.type === 'underrun' && typeof m.count === 'number') {
        setUnderruns(m.count);
        return;
      }
      if (m?.type !== 'input' || !m.samples) return;
      const samples = m.samples;
      let off = 0;
      while (off < samples.length) {
        const room = RNNOISE_FRAME - s.pendingFill;
        const take = Math.min(room, samples.length - off);
        s.pending.set(samples.subarray(off, off + take), s.pendingFill);
        s.pendingFill += take;
        off += take;
        if (s.pendingFill === RNNOISE_FRAME) {
          // Full 10 ms frame — run the real network (or bypass: pass through).
          const out = bypassRef.current
            ? new Float32Array(s.pending)
            : s.rnnoise.processFrame(s.pending);
          const buf = out.buffer.slice(0) as ArrayBuffer;
          s.node.port.postMessage({ type: 'processed', samples: new Float32Array(buf) }, [buf]);
          s.pendingFill = 0;
        }
      }
    };

    const meter = () => {
      const s = sessionRef.current;
      if (!s) return;
      const peak = (an: AnalyserNode | null, data: Uint8Array): number => {
        if (!an) return 0;
        an.getByteTimeDomainData(data);
        let p = 0;
        for (let i = 0; i < data.length; i++) {
          const v = Math.abs(data[i] - 128) / 128;
          if (v > p) p = v;
        }
        return p;
      };
      const pi = peak(inAnalyserRef.current, inData);
      const po = peak(outAnalyserRef.current, outData);
      if (inBarRef.current) inBarRef.current.style.width = `${Math.min(100, pi * 100)}%`;
      if (outBarRef.current) outBarRef.current.style.width = `${Math.min(100, po * 100)}%`;
      if (vadBarRef.current) {
        const v = bypassRef.current ? 0 : s.rnnoise.lastVad;
        vadBarRef.current.style.width = `${Math.min(100, v * 100)}%`;
      }
      s.raf = requestAnimationFrame(meter);
    };
    session.raf = requestAnimationFrame(meter);

    setState('live');
    setStatus(
      'Live. Speak — the lower meter is your voice after RNNoise. Toggle bypass to compare.',
    );
  };

  const stop = async () => {
    await stopSession();
    setState('idle');
    setStatus('Hear what RNNoise does to your mic, live — before you clean a file.');
    setBypass(false);
  };

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-6 space-y-5">
      <div>
        <h2 className="text-lg font-semibold text-zinc-100">Live preview</h2>
        <p className="mt-1 text-sm text-zinc-400" role="status">
          {status}
        </p>
      </div>

      <div className="space-y-3">
        <Meter label="Mic in (raw)" barRef={inBarRef} color="bg-zinc-400" />
        <Meter label="After RNNoise" barRef={outBarRef} color="bg-emerald-400" />
        <Meter label="Voice activity (network output)" barRef={vadBarRef} color="bg-sky-400" />
      </div>

      <div className="flex flex-wrap items-center gap-3">
        {state === 'live' ? (
          <button
            type="button"
            onClick={stop}
            className="rounded-xl bg-zinc-100 px-5 py-2.5 text-sm font-semibold text-zinc-900 hover:bg-white"
          >
            Stop preview
          </button>
        ) : (
          <button
            type="button"
            onClick={start}
            disabled={state === 'starting'}
            className="rounded-xl bg-emerald-500 px-5 py-2.5 text-sm font-semibold text-zinc-950 hover:bg-emerald-400 disabled:opacity-50"
          >
            {state === 'starting' ? 'Starting…' : 'Start live preview'}
          </button>
        )}
        <label className="flex items-center gap-2 text-sm text-zinc-300 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={bypass}
            disabled={state !== 'live'}
            onChange={(e) => setBypass(e.target.checked)}
            className="h-4 w-4 rounded accent-emerald-400 disabled:opacity-40"
          />
          Bypass (hear the raw mic)
        </label>
      </div>

      {underruns > 0 && state === 'live' && (
        <p className="text-xs text-amber-300/90" role="status">
          The audio worker ran dry {underruns} time{underruns === 1 ? '' : 's'} — you may have
          heard a brief gap. Close other heavy tabs if it keeps happening.
        </p>
      )}

      <div className="rounded-xl border border-zinc-800 bg-zinc-950/60 p-4 text-xs text-zinc-500 leading-relaxed">
        <p className="font-medium text-zinc-400 mb-1">How this works</p>
        <p>
          Your microphone is captured raw (no browser noise suppression) at 48 kHz mono and
          sliced into 10&nbsp;ms frames. Each frame runs through{' '}
          <span className="text-zinc-300">RNNoise</span>, the pretrained recurrent neural
          network by Xiph.Org, compiled to WebAssembly and running entirely on this device —
          the same engine family the full cleaning pipeline uses. Audio never leaves your
          browser in this tab. Headphones are recommended to avoid feedback.
        </p>
      </div>
    </div>
  );
}

function Meter({
  label,
  barRef,
  color,
}: {
  label: string;
  barRef: Ref<HTMLDivElement>;
  color: string;
}) {
  return (
    <div>
      <p className="text-xs text-zinc-500 mb-1">{label}</p>
      <div className="h-2.5 rounded-full bg-zinc-800 overflow-hidden">
        <div ref={barRef} className={`h-full ${color}`} style={{ width: '0%' }} />
      </div>
    </div>
  );
}
