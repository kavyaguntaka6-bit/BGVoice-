export type ModeId = 'standard' | 'music' | 'isolate' | 'studio';
export type RoomTone = 'off' | 'subtle' | 'natural';

export interface CleanSettings {
  mode: ModeId;
  strength: number;
  dereverb: boolean;
  roomTone: RoomTone;
  loudnessTarget: number;
  enrollFile: File | null;
}

export const MODES: Array<{ id: ModeId; label: string; blurb: string }> = [
  {
    id: 'standard',
    label: 'Standard',
    blurb: 'General cleanup: steady background noise, hum and hiss.',
  },
  {
    id: 'music',
    label: 'Remove music & background track',
    blurb: 'Separates the voice from music or a background track.',
  },
  {
    id: 'isolate',
    label: 'Isolate my voice',
    blurb: 'Keeps one target voice and quiets everyone else. Optionally upload a short sample of your voice to guide it.',
  },
  {
    id: 'studio',
    label: 'Studio max quality',
    blurb: 'The full chain with stronger noise removal and studio finishing. Takes longer.',
  },
];

export const DEFAULT_SETTINGS: CleanSettings = {
  mode: 'standard',
  strength: 0.6,
  dereverb: false,
  roomTone: 'subtle',
  loudnessTarget: -16,
  enrollFile: null,
};

interface ControlsProps {
  settings: CleanSettings;
  onChange: (s: CleanSettings) => void;
  fileName: string | null;
  submitting: boolean;
  onSubmit: () => void;
  onClearFile: () => void;
}

const fieldLabel = 'block text-sm font-medium text-zinc-300 mb-1.5';
const selectCls =
  'w-full rounded-xl bg-zinc-900 border border-zinc-700 px-3 py-2.5 text-sm text-zinc-100 focus:border-emerald-400';

export default function Controls({
  settings,
  onChange,
  fileName,
  submitting,
  onSubmit,
  onClearFile,
}: ControlsProps) {
  const set = <K extends keyof CleanSettings>(key: K, value: CleanSettings[K]) =>
    onChange({ ...settings, [key]: value });

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5 space-y-5">
      {fileName && (
        <div className="flex items-center justify-between gap-3 rounded-xl bg-zinc-900 border border-zinc-800 px-3 py-2.5">
          <p className="text-sm text-zinc-300 truncate" title={fileName}>
            <span className="text-zinc-500">File:</span> {fileName}
          </p>
          <button
            type="button"
            onClick={onClearFile}
            className="shrink-0 text-xs text-zinc-400 hover:text-zinc-100 underline underline-offset-2"
          >
            Change
          </button>
        </div>
      )}

      <div>
        <label htmlFor="mode" className={fieldLabel}>
          Cleaning mode
        </label>
        <select
          id="mode"
          className={selectCls}
          value={settings.mode}
          onChange={(e) => set('mode', e.target.value as ModeId)}
        >
          {MODES.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
        <p className="mt-1.5 text-xs text-zinc-500">
          {MODES.find((m) => m.id === settings.mode)?.blurb}
        </p>
      </div>

      {settings.mode === 'isolate' && (
        <div>
          <label htmlFor="enroll" className={fieldLabel}>
            My voice sample <span className="text-zinc-500 font-normal">(optional)</span>
          </label>
          <input
            id="enroll"
            type="file"
            accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac,.webm"
            onChange={(e) => set('enrollFile', e.target.files?.[0] ?? null)}
            className="block w-full text-sm text-zinc-400 file:mr-3 file:rounded-lg file:border file:border-zinc-700 file:bg-zinc-800 file:px-3 file:py-2 file:text-sm file:text-zinc-200 hover:file:bg-zinc-700"
          />
          <p className="mt-1.5 text-xs text-zinc-500">
            {settings.enrollFile
              ? `Using "${settings.enrollFile.name}" as the reference voice.`
              : 'A few seconds of just your voice helps the isolation. Without it, the loudest clear voice is used.'}
          </p>
        </div>
      )}

      <div>
        <label htmlFor="strength" className={fieldLabel}>
          Cleanup strength{' '}
          <span className="text-zinc-500 font-normal tabular-nums">
            {settings.strength.toFixed(2)}
          </span>
        </label>
        <input
          id="strength"
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={settings.strength}
          onChange={(e) => set('strength', Number(e.target.value))}
          className="w-full"
          aria-valuetext={`${Math.round(settings.strength * 100)} percent`}
        />
        <div className="flex justify-between text-xs text-zinc-600 mt-1">
          <span>Gentle</span>
          <span>Aggressive</span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label htmlFor="roomtone" className={fieldLabel}>
            Room tone
          </label>
          <select
            id="roomtone"
            className={selectCls}
            value={settings.roomTone}
            onChange={(e) => set('roomTone', e.target.value as RoomTone)}
          >
            <option value="off">Off</option>
            <option value="subtle">Subtle</option>
            <option value="natural">Natural</option>
          </select>
        </div>
        <div>
          <label htmlFor="loudness" className={fieldLabel}>
            Loudness target
          </label>
          <select
            id="loudness"
            className={selectCls}
            value={String(settings.loudnessTarget)}
            onChange={(e) => set('loudnessTarget', Number(e.target.value))}
          >
            <option value="-14">−14 LUFS (loud, social)</option>
            <option value="-16">−16 LUFS (balanced)</option>
            <option value="-23">−23 LUFS (broadcast)</option>
          </select>
        </div>
      </div>

      <div className="flex items-start gap-3">
        <input
          id="dereverb"
          type="checkbox"
          checked={settings.dereverb}
          onChange={(e) => set('dereverb', e.target.checked)}
          className="mt-1 h-4 w-4 rounded accent-emerald-400"
        />
        <div>
          <label htmlFor="dereverb" className="text-sm font-medium text-zinc-200 cursor-pointer">
            Reduce echo
          </label>
          <p className="text-xs text-zinc-500">
            Tames room echo and reverb. Useful for recordings made in empty rooms.
          </p>
        </div>
      </div>

      <button
        type="button"
        onClick={onSubmit}
        disabled={!fileName || submitting}
        className="w-full rounded-xl bg-emerald-500 px-4 py-3 text-sm font-semibold text-zinc-950 hover:bg-emerald-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        {submitting ? 'Starting…' : fileName ? 'Clean my audio' : 'Pick a file or record first'}
      </button>
    </div>
  );
}
