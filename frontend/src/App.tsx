import { useState } from 'react';
import DropZone from './components/DropZone';
import Recorder from './components/Recorder';
import Controls, { DEFAULT_SETTINGS, type CleanSettings } from './components/Controls';
import JobProgress from './components/JobProgress';
import ResultsView from './components/ResultsView';
import LivePreview from './components/LivePreview';
import { createJob, deleteJob, type JobReport } from './api/client';

type Tab = 'clean' | 'live';
type Phase = 'idle' | 'processing' | 'results';

export default function App() {
  const [tab, setTab] = useState<Tab>('clean');
  const [phase, setPhase] = useState<Phase>('idle');
  const [file, setFile] = useState<File | null>(null);
  const [settings, setSettings] = useState<CleanSettings>(DEFAULT_SETTINGS);
  const [jobId, setJobId] = useState<string | null>(null);
  const [report, setReport] = useState<JobReport | undefined>(undefined);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const pickFile = (f: File) => {
    setFile(f);
    setSubmitError(null);
    // A new file means a fresh job.
    if (phase !== 'idle') {
      setPhase('idle');
      setJobId(null);
      setReport(undefined);
    }
  };

  const clearFile = () => {
    setFile(null);
    setSubmitError(null);
  };

  const submit = async () => {
    if (!file || submitting) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const id = await createJob({
        file,
        mode: settings.mode,
        strength: settings.strength,
        loudnessTarget: settings.loudnessTarget,
        roomTone: settings.roomTone,
        dereverb: settings.dereverb,
        enrollFile: settings.mode === 'isolate' ? settings.enrollFile : null,
      });
      setJobId(id);
      setReport(undefined);
      setPhase('processing');
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : 'Could not start the cleaning job.');
    } finally {
      setSubmitting(false);
    }
  };

  const cancelJob = () => {
    if (jobId) void deleteJob(jobId);
    setJobId(null);
    setReport(undefined);
    setPhase('idle');
  };

  const handleDone = (r: JobReport | undefined) => {
    setReport(r);
    setPhase('results');
  };

  const resetAll = () => {
    if (jobId) void deleteJob(jobId);
    setJobId(null);
    setReport(undefined);
    setFile(null);
    setPhase('idle');
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <header className="border-b border-zinc-900">
        <div className="mx-auto max-w-5xl px-4 py-5 flex items-center justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold tracking-tight">
              VoiceClean <span className="text-emerald-400">v2</span>
            </h1>
            <p className="text-xs text-zinc-500 mt-0.5">
              Clean voice recordings — noise, music, echo and room tone handled.
            </p>
          </div>
          <nav className="flex rounded-xl border border-zinc-800 overflow-hidden" aria-label="Main">
            {(
              [
                ['clean', 'Clean audio'],
                ['live', 'Live preview'],
              ] as Array<[Tab, string]>
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                onClick={() => setTab(id)}
                aria-current={tab === id ? 'page' : undefined}
                className={[
                  'px-4 py-2 text-sm font-medium transition-colors',
                  tab === id
                    ? 'bg-zinc-800 text-zinc-100'
                    : 'bg-transparent text-zinc-500 hover:text-zinc-300',
                ].join(' ')}
              >
                {label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-8">
        {tab === 'live' ? (
          <LivePreview />
        ) : phase === 'results' && jobId && file ? (
          <ResultsView
            inputFile={file}
            jobId={jobId}
            report={report}
            onReset={resetAll}
          />
        ) : phase === 'processing' && jobId ? (
          <div className="max-w-2xl mx-auto">
            <JobProgress jobId={jobId} onDone={handleDone} onCancel={cancelJob} />
          </div>
        ) : (
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="space-y-6">
              <DropZone onFile={pickFile} />
              <div className="flex items-center gap-3 text-xs text-zinc-600" aria-hidden="true">
                <span className="flex-1 h-px bg-zinc-900" />
                or
                <span className="flex-1 h-px bg-zinc-900" />
              </div>
              <Recorder onRecording={pickFile} />
            </div>
            <div>
              <Controls
                settings={settings}
                onChange={setSettings}
                fileName={file?.name ?? null}
                submitting={submitting}
                onSubmit={submit}
                onClearFile={clearFile}
              />
              {submitError && (
                <p role="alert" className="mt-3 text-sm text-red-400">
                  {submitError}
                </p>
              )}
            </div>
          </div>
        )}
      </main>

      <footer className="border-t border-zinc-900 mt-8">
        <div className="mx-auto max-w-5xl px-4 py-5 text-xs text-zinc-600 leading-relaxed">
          <p>
            Cleaning runs on your VoiceClean v2 backend. The live preview tab runs the genuine
            RNNoise neural network (Xiph.Org) as WebAssembly, entirely on this device.
          </p>
        </div>
      </footer>
    </div>
  );
}
