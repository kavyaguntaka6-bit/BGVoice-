/**
 * Thin client for the VoiceClean v2 backend API (FastAPI).
 * Base URL is configurable via the VITE_API_URL env var, default http://localhost:8000.
 * No mock responses anywhere — every function here hits the real HTTP API.
 */

export const API_URL: string =
  (
    (typeof window !== 'undefined' &&
      (window as unknown as { __VOICECLEAN_API_URL?: string }).__VOICECLEAN_API_URL) ||
    (import.meta.env.VITE_API_URL as string | undefined)
  )?.replace(/\/+$/, '') || 'http://localhost:8000';

export type JobStatus = 'queued' | 'processing' | 'done' | 'error';

/** Measured values returned by the backend pipeline. All fields optional —
 *  the UI renders only the ones actually present. */
export interface JobReport {
  mode?: string;
  strength?: number;
  loudness_target?: number;
  room_tone?: string;
  dereverb?: boolean;
  stages?: string[];
  duration_s?: number;
  input_lufs?: number;
  output_lufs?: number;
  noise_floor_before_db?: number;
  noise_floor_after_db?: number;
  processing_time_s?: number;
  hum_notches_hz?: number[];
  // The backend may flag video inputs under one of these names.
  has_video?: boolean;
  is_video?: boolean;
  video?: boolean;
  input_is_video?: boolean;
}

export interface JobState {
  status: JobStatus;
  progress: number; // 0..1
  stage?: string;
  report?: JobReport;
  error?: string;
}

export type DownloadFormat = 'wav' | 'mp3' | 'mp4';

export interface CreateJobOptions {
  file: File;
  mode: string;
  strength: number;
  loudnessTarget: number;
  roomTone: string;
  dereverb: boolean;
  enrollFile?: File | null;
}

export async function createJob(opts: CreateJobOptions): Promise<string> {
  const form = new FormData();
  form.append('file', opts.file, opts.file.name);
  form.append('mode', opts.mode);
  form.append('strength', String(opts.strength));
  form.append('loudness_target', String(opts.loudnessTarget));
  form.append('room_tone', opts.roomTone);
  form.append('dereverb', opts.dereverb ? 'true' : 'false');
  if (opts.enrollFile) {
    form.append('enroll_file', opts.enrollFile, opts.enrollFile.name);
  }

  let res: Response;
  try {
    res = await fetch(`${API_URL}/api/jobs`, { method: 'POST', body: form });
  } catch (e) {
    throw new Error(
      `Could not reach the VoiceClean API at ${API_URL}. Is the backend running?`,
    );
  }
  if (!res.ok) {
    const text = await safeErrorText(res);
    throw new Error(`Job creation failed (${res.status}): ${text}`);
  }
  const body = (await res.json()) as { job_id?: string; id?: string };
  const jobId = body.job_id ?? body.id;
  if (!jobId) throw new Error('The API did not return a job id.');
  return jobId;
}

export async function getJob(jobId: string, signal?: AbortSignal): Promise<JobState> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/api/jobs/${encodeURIComponent(jobId)}`, {
      signal,
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e;
    throw new Error(
      `Lost connection to the VoiceClean API at ${API_URL} while checking job status.`,
    );
  }
  if (res.status === 404) {
    throw new Error('Job not found — it may have expired on the server.');
  }
  if (!res.ok) {
    throw new Error(`Status check failed (${res.status}): ${await safeErrorText(res)}`);
  }
  const body = (await res.json()) as JobState;
  return {
    status: body.status,
    progress: clamp01(Number(body.progress ?? 0)),
    stage: body.stage,
    report: body.report,
    error: body.error,
  };
}

export async function deleteJob(jobId: string): Promise<void> {
  try {
    await fetch(`${API_URL}/api/jobs/${encodeURIComponent(jobId)}`, {
      method: 'DELETE',
    });
  } catch {
    // Best-effort cleanup; the server expires jobs on its own.
  }
}

export function downloadUrl(jobId: string, format: DownloadFormat): string {
  return `${API_URL}/api/jobs/${encodeURIComponent(jobId)}/download?format=${format}`;
}

/** Download a result file and trigger a browser save. Surfaces real API
 *  errors (e.g. 400 when MP4 is requested for an audio-only job). */
export async function downloadResult(
  jobId: string,
  format: DownloadFormat,
  filename: string,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(downloadUrl(jobId, format));
  } catch {
    throw new Error(
      `Could not reach the VoiceClean API at ${API_URL} to download the file.`,
    );
  }
  if (!res.ok) {
    throw new Error(`Download failed (${res.status}): ${await safeErrorText(res)}`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
  }
}

async function safeErrorText(res: Response): Promise<string> {
  try {
    const text = await res.text();
    if (!text) return res.statusText || 'unknown error';
    // The API may return JSON {detail: ...} or plain text.
    try {
      const parsed = JSON.parse(text) as { detail?: unknown; error?: unknown };
      if (typeof parsed.detail === 'string') return parsed.detail;
      if (typeof parsed.error === 'string') return parsed.error;
    } catch {
      /* not JSON */
    }
    return text.slice(0, 300);
  } catch {
    return res.statusText || 'unknown error';
  }
}

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.min(1, Math.max(0, v));
}

/** Human-readable label for a backend pipeline stage name. Unknown stages
 *  are shown as-is (never hidden, never stuck). */
export function humanizeStage(stage?: string): string {
  if (!stage) return 'Working';
  const map: Record<string, string> = {
    ingest: 'Reading your audio',
    linear: 'Removing hum & hiss',
    separate_music: 'Separating music',
    vad: 'Finding speech',
    separate_speaker: 'Isolating your voice',
    denoise: 'Removing noise',
    dereverb: 'Reducing echo',
    postfilter: 'Polishing speech',
    roomtone: 'Balancing room tone',
    tone: 'Studio finishing',
    loudness: 'Setting loudness',
    export: 'Writing the file',
  };
  return map[stage] ?? stage.replace(/_/g, ' ');
}

/** Best-effort client-side check for video input (used to decide whether the
 *  MP4 download makes sense). The report's own flags take precedence. */
export function isVideoInput(file: File, report?: JobReport): boolean {
  if (report) {
    if (report.has_video === true || report.is_video === true || report.video === true || report.input_is_video === true) return true;
    if (report.has_video === false || report.is_video === false) return false;
  }
  const name = file.name.toLowerCase();
  const videoExts = ['.mp4', '.mov', '.mkv', '.m4v'];
  if (videoExts.some((e) => name.endsWith(e))) return true;
  if (file.type.startsWith('video/')) return true;
  return false;
}
