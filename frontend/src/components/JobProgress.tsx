import { useCallback, useEffect, useRef, useState } from 'react';
import { getJob, humanizeStage, type JobReport } from '../api/client';

interface JobProgressProps {
  jobId: string;
  onDone: (report: JobReport | undefined) => void;
  onCancel: () => void;
}

const POLL_MS = 1000;
const MAX_NET_FAILURES = 5;

/** Polls the real job status. Shows the backend's own stage name and
 *  progress, surfaces the backend's error text, and never gets stuck:
 *  repeated network failures surface a clear error with a retry button. */
export default function JobProgress({ jobId, onDone, onCancel }: JobProgressProps) {
  const [stage, setStage] = useState<string | undefined>(undefined);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<'queued' | 'processing'>('queued');
  const [fatal, setFatal] = useState<string | null>(null);
  const failures = useRef(0);
  const cancelled = useRef(false);
  const doneRef = useRef(onDone);
  doneRef.current = onDone;

  const poll = useCallback(async () => {
    if (cancelled.current) return;
    try {
      const job = await getJob(jobId);
      failures.current = 0;
      if (cancelled.current) return;
      if (job.status === 'done') {
        cancelled.current = true;
        doneRef.current(job.report);
        return;
      }
      if (job.status === 'error') {
        cancelled.current = true;
        setFatal(job.error || 'The backend reported a failure without details.');
        return;
      }
      setStatus(job.status === 'processing' ? 'processing' : 'queued');
      setStage(job.stage);
      setProgress(job.progress);
      window.setTimeout(poll, POLL_MS);
    } catch (e) {
      if (cancelled.current) return;
      if (e instanceof DOMException && e.name === 'AbortError') return;
      failures.current += 1;
      if (failures.current >= MAX_NET_FAILURES) {
        cancelled.current = true;
        setFatal(
          e instanceof Error
            ? e.message
            : 'Could not reach the backend after several tries.',
        );
        return;
      }
      window.setTimeout(poll, POLL_MS * 2);
    }
  }, [jobId]);

  useEffect(() => {
    cancelled.current = false;
    failures.current = 0;
    setFatal(null);
    setProgress(0);
    setStage(undefined);
    setStatus('queued');
    const t = window.setTimeout(poll, 300);
    return () => {
      cancelled.current = true;
      window.clearTimeout(t);
    };
  }, [poll]);

  const retry = () => {
    cancelled.current = false;
    failures.current = 0;
    setFatal(null);
    void poll();
  };

  if (fatal) {
    return (
      <div className="rounded-2xl border border-red-900/60 bg-red-950/30 p-6" role="alert">
        <p className="font-medium text-red-300">Something went wrong</p>
        <p className="mt-1 text-sm text-red-200/80 break-words">{fatal}</p>
        <div className="mt-4 flex gap-2">
          <button
            type="button"
            onClick={retry}
            className="rounded-xl bg-zinc-100 px-4 py-2 text-sm font-medium text-zinc-900 hover:bg-white"
          >
            Try again
          </button>
          <button
            type="button"
            onClick={onCancel}
            className="rounded-xl border border-zinc-700 px-4 py-2 text-sm text-zinc-300 hover:bg-zinc-800"
          >
            Start over
          </button>
        </div>
      </div>
    );
  }

  const pct = Math.round(progress * 100);
  return (
    <div
      className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-6"
      role="status"
      aria-live="polite"
      aria-label={`Processing: ${humanizeStage(stage)}, ${pct} percent`}
    >
      <div className="flex items-center justify-between gap-3">
        <p className="font-medium text-zinc-100">
          {status === 'queued' && !stage ? 'In the queue…' : humanizeStage(stage)}
          <span className="ml-2 inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse" aria-hidden="true" />
        </p>
        <p className="text-sm text-zinc-400 tabular-nums">{pct}%</p>
      </div>
      <div
        className="mt-3 h-3 rounded-full bg-zinc-800 overflow-hidden"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
        aria-label="Cleaning progress"
      >
        <div
          className="h-full rounded-full bg-emerald-400 transition-[width] duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="mt-2 text-xs text-zinc-500">
        {status === 'queued'
          ? 'Your job is waiting for a free worker.'
          : 'Working through the pipeline — this can take a while for long files.'}
      </p>
      <button
        type="button"
        onClick={onCancel}
        className="mt-4 text-xs text-zinc-500 hover:text-zinc-300 underline underline-offset-2"
      >
        Cancel and start over
      </button>
    </div>
  );
}
