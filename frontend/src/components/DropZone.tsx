import { useRef, useState } from 'react';

export const ACCEPTED_EXTENSIONS = [
  'wav',
  'mp3',
  'm4a',
  'aac',
  'ogg',
  'flac',
  'mp4',
  'mov',
  'mkv',
  'webm',
];

const ACCEPT_ATTR = ACCEPTED_EXTENSIONS.map((e) => `.${e}`).join(',');

function isAccepted(file: File): boolean {
  const name = file.name.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((e) => name.endsWith(`.${e}`));
}

interface DropZoneProps {
  onFile: (file: File) => void;
  disabled?: boolean;
  compact?: boolean;
}

export default function DropZone({ onFile, disabled, compact }: DropZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [rejected, setRejected] = useState<string | null>(null);

  const handleFile = (file: File | undefined | null) => {
    setRejected(null);
    if (!file) return;
    if (!isAccepted(file)) {
      setRejected(
        `"${file.name}" isn't a supported format. Use: ${ACCEPTED_EXTENSIONS.join(', ')}.`,
      );
      return;
    }
    onFile(file);
  };

  return (
    <div>
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label="Drop an audio or video file here, or press Enter to browse"
        aria-disabled={disabled}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => {
          if (disabled) return;
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          if (disabled) return;
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          if (disabled) return;
          e.preventDefault();
          setDragging(false);
          handleFile(e.dataTransfer.files?.[0]);
        }}
        className={[
          'rounded-2xl border-2 border-dashed transition-colors cursor-pointer',
          compact ? 'p-6' : 'p-10',
          dragging
            ? 'border-emerald-400 bg-emerald-400/10'
            : 'border-zinc-700 bg-zinc-900/60 hover:border-zinc-500 hover:bg-zinc-900',
          disabled ? 'opacity-50 cursor-not-allowed' : '',
        ].join(' ')}
      >
        <div className="flex flex-col items-center text-center gap-2">
          <svg
            className="w-10 h-10 text-zinc-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
            />
          </svg>
          <p className="text-zinc-200 font-medium">
            Drop your audio or video here
          </p>
          <p className="text-sm text-zinc-500">
            or <span className="text-emerald-400 underline">browse files</span>
          </p>
          <p className="text-xs text-zinc-600 mt-1">
            {ACCEPTED_EXTENSIONS.join(' · ')}
          </p>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_ATTR}
          className="hidden"
          disabled={disabled}
          onChange={(e) => {
            handleFile(e.target.files?.[0]);
            e.target.value = '';
          }}
        />
      </div>
      {rejected && (
        <p role="alert" className="mt-2 text-sm text-red-400">
          {rejected}
        </p>
      )}
    </div>
  );
}
