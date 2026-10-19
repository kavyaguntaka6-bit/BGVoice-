import { useCallback, useEffect, useRef, useState, type Ref } from 'react';
import {
  downloadResult,
  downloadUrl,
  isVideoInput,
  type DownloadFormat,
  type JobReport,
} from '../api/client';
import {
  computePeaks,
  computeSpectrogram,
  formatTime,
  paintWithPlayhead,
  renderSpectrogramBase,
  renderWaveformBase,
} from '../audio/dsp';
import { ABPlayer, loudnessMatchGain, type ClipId } from '../audio/player';

interface ResultsViewProps {
  inputFile: File;
  jobId: string;
  report: JobReport | undefined;
  onReset: () => void;
}

type LoadState =
  | { kind: 'loading'; note: string }
  | { kind: 'error'; message: string }
  | { kind: 'ready' };

const WAVE_W = 960;
const WAVE_H = 110;
const SPEC_H = 150;

export default function ResultsView({ inputFile, jobId, report, onReset }: ResultsViewProps) {
  const [load, setLoad] = useState<LoadState>({ kind: 'loading', note: 'Loading audio…' });
  const [downloadMsg, setDownloadMsg] = useState<string | null>(null);
  const [busyFormat, setBusyFormat] = useState<DownloadFormat | null>(null);
  const [, setUiTick] = useState(0);

  const playerRef = useRef<ABPlayer | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const rafRef = useRef(0);
  const baseRef = useRef<{
    waveA: HTMLCanvasElement;
    waveB: HTMLCanvasElement;
    specA: HTMLCanvasElement;
    specB: HTMLCanvasElement;
  } | null>(null);

  const waveARef = useRef<HTMLCanvasElement>(null);
  const waveBRef = useRef<HTMLCanvasElement>(null);
  const specARef = useRef<HTMLCanvasElement>(null);
  const specBRef = useRef<HTMLCanvasElement>(null);
  const timeRef = useRef<HTMLSpanElement>(null);

  const matchGain = loudnessMatchGain(report?.input_lufs, report?.output_lufs);
  const showMp4 = isVideoInput(inputFile, report);

  const paintAll = useCallback(() => {
    const base = baseRef.current;
    const p = playerRef.current;
    if (!base || !p) return;
    const frac = p.duration > 0 ? p.position / p.duration : null;
    const pairs: Array<[HTMLCanvasElement | null, HTMLCanvasElement, string]> = [
      [waveARef.current, base.waveA, '#f87171'],
      [waveBRef.current, base.waveB, '#34d399'],
      [specARef.current, base.specA, '#f87171'],
      [specBRef.current, base.specB, '#34d399'],
    ];
    for (const [cv, b, color] of pairs) {
      if (cv) paintWithPlayhead(cv, b, frac, color);
    }
    if (timeRef.current) {
      timeRef.current.textContent = `${formatTime(p.position)} / ${formatTime(p.duration)}`;
    }
  }, []);

  // Decode both clips once, then build the player.
  useEffect(() => {
    let cancelled = false;
    const ctx = new AudioContext();
    ctxRef.current = ctx;

    const decode = async (src: ArrayBuffer) => {
      // Copy: decodeAudioData detaches the buffer in some browsers.
      const copy = src.slice(0);
      return ctx.decodeAudioData(copy);
    };

    (async () => {
      try {
        setLoad({ kind: 'loading', note: 'Decoding your original…' });
        const aBuf = await decode(await inputFile.arrayBuffer());
        if (cancelled) return;
        setLoad({ kind: 'loading', note: 'Fetching the cleaned result…' });
        const res = await fetch(downloadUrl(jobId, 'wav'));
        if (!res.ok) {
          throw new Error(`Could not fetch the cleaned audio (HTTP ${res.status}).`);
        }
        const bBuf = await decode(await res.arrayBuffer());
        if (cancelled) return;

        setLoad({ kind: 'loading', note: 'Drawing waveforms…' });
        // Let the loading note paint before the heavy canvas work.
        await new Promise((r) => setTimeout(r, 30));
        if (cancelled) return;

        baseRef.current = {
          waveA: renderWaveformBase(computePeaks(aBuf.getChannelData(0), WAVE_W), WAVE_W, WAVE_H, '#f87171'),
          waveB: renderWaveformBase(computePeaks(bBuf.getChannelData(0), WAVE_W), WAVE_W, WAVE_H, '#34d399'),
          specA: renderSpectrogramBase(computeSpectrogram(aBuf), WAVE_W, SPEC_H),
          specB: renderSpectrogramBase(computeSpectrogram(bBuf), WAVE_W, SPEC_H),
        };

        playerRef.current = new ABPlayer(ctx, aBuf, bBuf, matchGain, {
          onStateChange: () => setUiTick((n) => n + 1),
        });
        setLoad({ kind: 'ready' });
      } catch (e) {
        if (!cancelled) {
          setLoad({
            kind: 'error',
            message:
              e instanceof Error ? e.message : 'Could not load the audio for comparison.',
          });
        }
      }
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inputFile, jobId]);

  // rAF loop: advance the player and repaint playheads.
  useEffect(() => {
    if (load.kind !== 'ready') return;
    const loop = () => {
      playerRef.current?.tick();
      paintAll();
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [load.kind, paintAll]);

  // Keyboard: A/B switch and space to play/pause.
  useEffect(() => {
    if (load.kind !== 'ready') return;
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      if (e.key === 'a' || e.key === 'A') playerRef.current?.switchTo('A');
      else if (e.key === 'b' || e.key === 'B') playerRef.current?.switchTo('B');
      else if (e.key === ' ') {
        e.preventDefault();
        playerRef.current?.toggle();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [load.kind]);

  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafRef.current);
      playerRef.current?.dispose();
      playerRef.current = null;
      void ctxRef.current?.close().catch(() => undefined);
      ctxRef.current = null;
    };
  }, []);

  const seekFromClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const p = playerRef.current;
    if (!p) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / Math.max(1, rect.width);
    p.seek(frac * p.duration);
    paintAll();
  };

  const handleDownload = async (format: DownloadFormat) => {
    setDownloadMsg(null);
    setBusyFormat(format);
    try {
      const base = inputFile.name.replace(/\.[^.]+$/, '') || 'voiceclean';
      await downloadResult(jobId, format, `${base}.cleaned.${format}`);
    } catch (e) {
      setDownloadMsg(e instanceof Error ? e.message : 'Download failed.');
    } finally {
      setBusyFormat(null);
    }
  };

  const player = playerRef.current;
  const active: ClipId = player?.activeClip ?? 'A';
  const playing = player?.playing ?? false;

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-zinc-100">Result</h2>
        <button
          type="button"
          onClick={onReset}
          className="text-sm text-zinc-400 hover:text-zinc-100 underline underline-offset-2"
        >
          Clean another file
        </button>
      </div>

      {load.kind === 'loading' && (
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-8 text-center" role="status">
          <div className="mx-auto w-8 h-8 rounded-full border-2 border-zinc-700 border-t-emerald-400 animate-spin" aria-hidden="true" />
          <p className="mt-3 text-sm text-zinc-400">{load.note}</p>
        </div>
      )}

      {load.kind === 'error' && (
        <div className="rounded-2xl border border-red-900/60 bg-red-950/30 p-6" role="alert">
          <p className="font-medium text-red-300">Couldn't load the comparison</p>
          <p className="mt-1 text-sm text-red-200/80">{load.message}</p>
          <p className="mt-2 text-sm text-zinc-400">
            Your cleaned file may still be fine — try downloading it below.
          </p>
        </div>
      )}

      {load.kind === 'ready' && (
        <>
          {/* Transport */}
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-4">
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => playerRef.current?.toggle()}
                className="rounded-xl bg-emerald-500 px-5 py-2.5 text-sm font-semibold text-zinc-950 hover:bg-emerald-400"
                aria-label={playing ? 'Pause' : 'Play'}
              >
                {playing ? '❚❚ Pause' : '▶ Play'}
              </button>
              <div className="flex rounded-xl overflow-hidden border border-zinc-700" role="group" aria-label="Compare original and cleaned">
                {(['A', 'B'] as ClipId[]).map((c) => (
                  <button
                    key={c}
                    type="button"
                    onClick={() => playerRef.current?.switchTo(c)}
                    aria-pressed={active === c}
                    className={[
                      'px-4 py-2.5 text-sm font-medium transition-colors',
                      active === c
                        ? c === 'A'
                          ? 'bg-red-400/20 text-red-200'
                          : 'bg-emerald-400/20 text-emerald-200'
                        : 'bg-zinc-900 text-zinc-400 hover:bg-zinc-800',
                    ].join(' ')}
                  >
                    {c === 'A' ? 'A · Original' : 'B · Cleaned'}
                  </button>
                ))}
              </div>
              <span ref={timeRef} className="ml-auto text-sm text-zinc-400 tabular-nums">
                0:00 / 0:00
              </span>
            </div>
            <p className="mt-2 text-xs text-zinc-500">
              {matchGain !== null ? (
                <>
                  A/B is <span className="text-zinc-300">level-matched</span> — the cleaned clip
                  is brought to the original's loudness so you judge the cleanup, not the volume.
                </>
              ) : (
                <>
                  Loudness measurements weren't available, so both clips play at their natural
                  levels — the volume difference you hear is real, not compensated.
                </>
              )}{' '}
              <span className="text-zinc-600">Keys: A / B to switch, Space to play/pause. Click a waveform to seek.</span>
            </p>
          </div>

          {/* Waveforms + spectrograms */}
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-4 space-y-4">
            <ClipBlock
              label="A · Original"
              color="text-red-300"
              waveRef={waveARef}
              specRef={specARef}
              onSeek={seekFromClick}
            />
            <ClipBlock
              label="B · Cleaned"
              color="text-emerald-300"
              waveRef={waveBRef}
              specRef={specBRef}
              onSeek={seekFromClick}
            />
            <p className="text-xs text-zinc-600">
              Spectrograms use a logarithmic frequency axis (low frequencies at the bottom).
            </p>
          </div>

          <div className="grid gap-5 md:grid-cols-2">
            <ReportPanel report={report} />
            <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5">
              <h3 className="font-medium text-zinc-100 mb-3">Download</h3>
              <div className="flex flex-wrap gap-2">
                {(['wav', 'mp3'] as DownloadFormat[]).map((f) => (
                  <button
                    key={f}
                    type="button"
                    disabled={busyFormat !== null}
                    onClick={() => handleDownload(f)}
                    className="rounded-xl border border-zinc-700 bg-zinc-800 px-4 py-2.5 text-sm font-medium text-zinc-100 hover:bg-zinc-700 disabled:opacity-50"
                  >
                    {busyFormat === f ? 'Preparing…' : f.toUpperCase()}
                  </button>
                ))}
                {showMp4 && (
                  <button
                    type="button"
                    disabled={busyFormat !== null}
                    onClick={() => handleDownload('mp4')}
                    className="rounded-xl border border-zinc-700 bg-zinc-800 px-4 py-2.5 text-sm font-medium text-zinc-100 hover:bg-zinc-700 disabled:opacity-50"
                  >
                    {busyFormat === 'mp4' ? 'Preparing…' : 'MP4'}
                  </button>
                )}
              </div>
              {!showMp4 && (
                <p className="mt-2 text-xs text-zinc-600">
                  MP4 download appears only for video inputs.
                </p>
              )}
              {downloadMsg && (
                <p role="alert" className="mt-3 text-sm text-red-400">
                  {downloadMsg}
                </p>
              )}
            </div>
          </div>
        </>
      )}

      {/* Downloads are also offered when the comparison view failed to load. */}
      {load.kind === 'error' && (
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5">
          <h3 className="font-medium text-zinc-100 mb-3">Download anyway</h3>
          <div className="flex flex-wrap gap-2">
            {(['wav', 'mp3'] as DownloadFormat[]).map((f) => (
              <button
                key={f}
                type="button"
                disabled={busyFormat !== null}
                onClick={() => handleDownload(f)}
                className="rounded-xl border border-zinc-700 bg-zinc-800 px-4 py-2.5 text-sm font-medium text-zinc-100 hover:bg-zinc-700 disabled:opacity-50"
              >
                {busyFormat === f ? 'Preparing…' : f.toUpperCase()}
              </button>
            ))}
            {showMp4 && (
              <button
                type="button"
                disabled={busyFormat !== null}
                onClick={() => handleDownload('mp4')}
                className="rounded-xl border border-zinc-700 bg-zinc-800 px-4 py-2.5 text-sm font-medium text-zinc-100 hover:bg-zinc-700 disabled:opacity-50"
              >
                {busyFormat === 'mp4' ? 'Preparing…' : 'MP4'}
              </button>
            )}
          </div>
          {downloadMsg && (
            <p role="alert" className="mt-3 text-sm text-red-400">
              {downloadMsg}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function ClipBlock({
  label,
  color,
  waveRef,
  specRef,
  onSeek,
}: {
  label: string;
  color: string;
  waveRef: Ref<HTMLCanvasElement>;
  specRef: Ref<HTMLCanvasElement>;
  onSeek: (e: React.MouseEvent<HTMLCanvasElement>) => void;
}) {
  return (
    <div>
      <p className={`text-xs font-medium uppercase tracking-wide ${color} mb-1.5`}>{label}</p>
      <canvas
        ref={waveRef}
        onClick={onSeek}
        className="w-full h-[110px] rounded-lg bg-zinc-950 cursor-pointer"
        role="img"
        aria-label={`${label} waveform — click to seek`}
      />
      <canvas
        ref={specRef}
        className="w-full h-[150px] rounded-lg bg-zinc-950 mt-1.5"
        role="img"
        aria-label={`${label} spectrogram, log frequency axis`}
      />
    </div>
  );
}

/** Report panel: renders ONLY values the backend actually measured. */
function ReportPanel({ report }: { report: JobReport | undefined }) {
  const rows: Array<[string, string]> = [];
  if (report) {
    if (typeof report.duration_s === 'number') {
      rows.push(['Duration', formatTime(report.duration_s)]);
    }
    if (typeof report.input_lufs === 'number') {
      rows.push(['Input loudness', `${report.input_lufs.toFixed(1)} LUFS`]);
    }
    if (typeof report.output_lufs === 'number') {
      rows.push(['Output loudness', `${report.output_lufs.toFixed(1)} LUFS`]);
    }
    if (typeof report.noise_floor_before_db === 'number') {
      rows.push(['Noise floor, before', `${report.noise_floor_before_db.toFixed(1)} dB`]);
    }
    if (typeof report.noise_floor_after_db === 'number') {
      rows.push(['Noise floor, after', `${report.noise_floor_after_db.toFixed(1)} dB`]);
    }
    if (typeof report.processing_time_s === 'number') {
      rows.push(['Processing time', `${report.processing_time_s.toFixed(1)} s`]);
    }
  }

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5">
      <h3 className="font-medium text-zinc-100 mb-3">Measured report</h3>
      {rows.length === 0 ? (
        <p className="text-sm text-zinc-500">
          The backend didn't return measurements for this job, so there's nothing to show here —
          no numbers are invented.
        </p>
      ) : (
        <dl className="space-y-2">
          {rows.map(([k, v]) => (
            <div key={k} className="flex justify-between gap-4 text-sm">
              <dt className="text-zinc-500">{k}</dt>
              <dd className="text-zinc-100 tabular-nums font-medium">{v}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}
