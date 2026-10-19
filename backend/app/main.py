"""VoiceClean v2 API — FastAPI layer over pipeline.pipeline.

Run from ~/workspace/voiceclean-v2/backend with::

    PYTHONPATH=. ../.venv/bin/uvicorn app.main:app --port 8000

Endpoints:
    POST   /api/jobs                  create a job (multipart upload)
    GET    /api/jobs/{job_id}         job status / progress / report
    GET    /api/jobs/{job_id}/download?format=wav|mp3|mp4
    DELETE /api/jobs/{job_id}         delete job record + all its files

Jobs persist in SQLite (jobs.sqlite3) and their files are auto-deleted
JOB_TTL_S (1 h) after creation by a background sweep thread.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import config as C
from pipeline.pipeline import run_pipeline

log = logging.getLogger("voiceclean.api")

# ----------------------------------------------------------------------------
# Paths — config values are relative to backend/ (the cwd the server runs from)
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / C.JOB_DB_PATH
UPLOAD_DIR = BASE_DIR / C.UPLOAD_DIR
RESULT_DIR = BASE_DIR / C.RESULT_DIR
MAX_FILE_BYTES = C.MAX_FILE_MB * 1024 * 1024
TTL_S = C.JOB_TTL_S

_STATUSES = ("queued", "processing", "done", "error")
_FORMATS = ("wav", "mp3", "mp4")

# Uploaded file parts are spooled to tempfile.gettempdir() by the multipart
# parser; /tmp can be a small tmpfs, so keep them on a disk-backed dir under
# the backend tree instead.
_TEMP_DIR = BASE_DIR / ".tmp"
_TEMP_DIR.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(_TEMP_DIR)

_db_lock = threading.Lock()       # serializes all SQLite access
_pipeline_lock = threading.Lock()  # serializes pipeline runs (model caches)

# ----------------------------------------------------------------------------
# SQLite helpers
# ----------------------------------------------------------------------------


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db_lock, _connect() as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0.0,
                stage TEXT NOT NULL DEFAULT '',
                report_json TEXT,
                error TEXT,
                input_path TEXT,
                output_wav TEXT,
                output_mp3 TEXT,
                output_mp4 TEXT,
                is_video INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            )"""
        )


def get_job_row(job_id: str) -> Optional[sqlite3.Row]:
    with _db_lock, _connect() as con:
        return con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    with _db_lock, _connect() as con:
        con.execute(f"UPDATE jobs SET {cols} WHERE id = ?",
                    (*fields.values(), job_id))


def delete_job_row(job_id: str) -> None:
    with _db_lock, _connect() as con:
        con.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


# ----------------------------------------------------------------------------
# JSON safety: pipeline reports may carry numpy scalars/arrays
# ----------------------------------------------------------------------------


def _json_safe(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o


# ----------------------------------------------------------------------------
# Filesystem helpers
# ----------------------------------------------------------------------------


def _remove_path(path: Optional[str]) -> None:
    if not path:
        return
    try:
        p = Path(path)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists() or p.is_symlink():
            p.unlink(missing_ok=True)
    except OSError:
        pass


def _delete_job_files(row: sqlite3.Row) -> None:
    for key in ("input_path", "output_wav", "output_mp3", "output_mp4"):
        _remove_path(row[key])
    # Also drop the per-job working dirs (uploads/<id>, results/<id>).
    for base in (UPLOAD_DIR, RESULT_DIR):
        _remove_path(str(base / row["id"]))


def _expired(created_at: float) -> bool:
    return (time.time() - float(created_at)) > TTL_S


def _drop_expired_row(row: sqlite3.Row) -> None:
    _delete_job_files(row)
    delete_job_row(row["id"])


def sweep_expired() -> int:
    """Delete all jobs older than TTL. Returns number removed."""
    removed = 0
    now = time.time()
    with _db_lock, _connect() as con:
        rows = con.execute("SELECT * FROM jobs").fetchall()
    for row in rows:
        if (now - float(row["created_at"])) > TTL_S:
            _delete_job_files(row)
            delete_job_row(row["id"])
            removed += 1
    return removed


def _sweep_loop() -> None:
    while True:
        time.sleep(60)
        try:
            n = sweep_expired()
            if n:
                log.info("ttl sweep removed %d job(s)", n)
        except Exception:  # keep the thread alive no matter what
            log.exception("ttl sweep failed")


# ----------------------------------------------------------------------------
# Media helpers
# ----------------------------------------------------------------------------


def _has_video_stream(path: str) -> bool:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30,
        )
        return out.returncode == 0 and out.stdout.strip().splitlines()[:1] == ["video"]
    except (OSError, subprocess.SubprocessError):
        return False


def _run_ffmpeg(args: list[str]) -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
            capture_output=True, text=True, timeout=1800, check=True,
        )
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError("ffmpeg conversion failed: " + " | ".join(tail)) from e


def _ensure_mp3(job_id: str, row: sqlite3.Row) -> str:
    if row["output_mp3"] and Path(row["output_mp3"]).exists():
        return row["output_mp3"]
    out = str(RESULT_DIR / job_id / "output.mp3")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(["-i", row["output_wav"], "-codec:a", "libmp3lame",
                 "-b:a", C.MP3_BITRATE, out])
    update_job(job_id, output_mp3=out)
    return out


def _ensure_mp4(job_id: str, row: sqlite3.Row) -> str:
    if not row["is_video"]:
        raise HTTPException(400, "mp4 download is only available for video inputs")
    if row["output_mp4"] and Path(row["output_mp4"]).exists():
        return row["output_mp4"]
    out = str(RESULT_DIR / job_id / "output.mp4")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(["-i", row["input_path"], "-i", row["output_wav"],
                 "-map", "0:v:0", "-map", "1:a:0",
                 "-c:v", "copy", "-c:a", "aac", "-b:a", C.AAC_BITRATE,
                 "-shortest", "-movflags", "+faststart", out])
    update_job(job_id, output_mp4=out)
    return out


# ----------------------------------------------------------------------------
# Background worker
# ----------------------------------------------------------------------------


def process_job(job_id: str, input_path: str, output_wav: str,
                *, mode: str, strength: float, loudness_target: float,
                room_tone: str, dereverb_on: Optional[bool],
                enroll_path: Optional[str]) -> None:
    update_job(job_id, status="processing", progress=0.02, stage="starting")
    last = {"stage": "", "progress": 0.0, "t": 0.0}

    def progress(stage: str, frac: float) -> None:
        prog = 0.05 + 0.90 * max(0.0, min(1.0, float(frac)))
        now = time.monotonic()
        if stage != last["stage"] or abs(prog - last["progress"]) >= 0.02 \
                or now - last["t"] > 2.0:
            last.update(stage=stage, progress=prog, t=now)
            update_job(job_id, stage=stage, progress=round(prog, 4))

    try:
        with _pipeline_lock:
            report = run_pipeline(
                input_path, output_wav,
                mode=mode, strength=strength,
                loudness_target=loudness_target, room_tone=room_tone,
                dereverb_on=dereverb_on, enroll_path=enroll_path,
                progress=progress,
            )
        update_job(job_id, status="done", progress=1.0, stage="done",
                   report_json=json.dumps(_json_safe(report)), error=None)
        log.info("job %s done (%s, %.1fs)", job_id, mode,
                 report.get("processing_time_s", 0.0))
    except Exception as e:  # noqa: BLE001 — store, never re-raise in worker
        msg = str(e).splitlines()[0] if str(e).strip() else type(e).__name__
        msg = msg.replace(str(BASE_DIR), "<app>")[:500]
        update_job(job_id, status="error", stage="error", error=msg)
        log.warning("job %s failed: %s", job_id, msg)


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------

app = FastAPI(title=C.API_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    # Jobs stranded in queued/processing by a previous run will never finish.
    with _db_lock, _connect() as con:
        con.execute(
            "UPDATE jobs SET status='error', error='server restarted before "
            "processing finished' WHERE status IN ('queued','processing')"
        )
    t = threading.Thread(target=_sweep_loop, name="ttl-sweep", daemon=True)
    t.start()


def _get_live_job(job_id: str) -> sqlite3.Row:
    row = get_job_row(job_id)
    if row is None:
        raise HTTPException(404, "unknown job id")
    if _expired(row["created_at"]):
        _drop_expired_row(row)
        raise HTTPException(404, "unknown job id")
    return row


def _job_response(row: sqlite3.Row) -> dict:
    report = json.loads(row["report_json"]) if row["report_json"] else None
    return {
        "status": row["status"],
        "progress": float(row["progress"]),
        "stage": row["stage"],
        "report": report,
        "error": row["error"],
    }


def _parse_dereverb(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    v = value.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise HTTPException(400, "dereverb must be a boolean (true/false)")


def _save_upload(up: UploadFile, dest: Path, limit: int = MAX_FILE_BYTES) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with dest.open("wb") as f:
        while True:
            chunk = up.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    413, f"file exceeds {C.MAX_FILE_MB} MB limit")
            f.write(chunk)
    return size


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    enroll_file: Optional[UploadFile] = File(None),
    mode: str = Form(C.MODES[0]),
    strength: float = Form(C.DEFAULT_STRENGTH),
    loudness_target: float = Form(C.LOUDNESS_TARGET),
    room_tone: str = Form(C.DEFAULT_ROOM_TONE),
    dereverb: Optional[str] = Form(None),
) -> dict:
    # ---- validation (audio content is never logged) ----
    if file is None or not file.filename:
        raise HTTPException(400, "file is required")
    if mode not in C.MODES:
        raise HTTPException(400, f"mode must be one of {list(C.MODES)}")
    if not 0.0 <= strength <= 1.0:
        raise HTTPException(400, "strength must be between 0 and 1")
    if loudness_target not in C.LOUDNESS_OPTIONS:
        raise HTTPException(
            400, f"loudness_target must be one of {list(C.LOUDNESS_OPTIONS)}")
    if room_tone not in ("off", "subtle", "natural"):
        raise HTTPException(400, "room_tone must be off, subtle or natural")
    dereverb_on = _parse_dereverb(dereverb)

    job_id = uuid.uuid4().hex
    up_dir = UPLOAD_DIR / job_id
    res_dir = RESULT_DIR / job_id

    # ---- persist uploads (413 on oversize) ----
    in_name = Path(file.filename).name or "input"
    in_ext = "".join(Path(in_name).suffixes)[:16]
    input_path = str(up_dir / f"input{in_ext}")
    try:
        size = _save_upload(file, Path(input_path))
    finally:
        await file.close()
    enroll_path: Optional[str] = None
    if enroll_file is not None and enroll_file.filename:
        en_name = Path(enroll_file.filename).name or "enroll"
        en_ext = "".join(Path(en_name).suffixes)[:16]
        enroll_path = str(up_dir / f"enroll{en_ext}")
        try:
            _save_upload(enroll_file, Path(enroll_path))
        finally:
            await enroll_file.close()

    output_wav = str(res_dir / "output.wav")
    is_video = _has_video_stream(input_path)

    with _db_lock, _connect() as con:
        con.execute(
            """INSERT INTO jobs (id, status, progress, stage, report_json, error,
                                 input_path, output_wav, output_mp3, output_mp4,
                                 is_video, created_at)
               VALUES (?, 'queued', 0.0, 'queued', NULL, NULL, ?, ?, NULL, NULL,
                       ?, ?)""",
            (job_id, input_path, output_wav, int(is_video), time.time()),
        )

    log.info("job %s queued: mode=%s strength=%.2f lufs=%.0f room=%s "
             "dereverb=%s video=%s bytes=%d", job_id, mode, strength,
             loudness_target, room_tone, dereverb_on, is_video, size)

    background_tasks.add_task(
        process_job, job_id, input_path, output_wav,
        mode=mode, strength=strength, loudness_target=loudness_target,
        room_tone=room_tone, dereverb_on=dereverb_on, enroll_path=enroll_path,
    )
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return _job_response(_get_live_job(job_id))


@app.get("/api/jobs/{job_id}/download")
def download_job(job_id: str, format: str = "wav") -> FileResponse:  # noqa: A002
    if format not in _FORMATS:
        raise HTTPException(400, f"format must be one of {list(_FORMATS)}")
    row = _get_live_job(job_id)
    if row["status"] != "done":
        raise HTTPException(409, f"job is {row['status']}; no output yet")
    if format == "wav":
        path, media, name = row["output_wav"], "audio/wav", f"{job_id}.wav"
    elif format == "mp3":
        try:
            path = _ensure_mp3(job_id, row)
        except RuntimeError as e:
            raise HTTPException(500, str(e)) from e
        media, name = "audio/mpeg", f"{job_id}.mp3"
    else:  # mp4 — remux original video with cleaned audio
        try:
            path = _ensure_mp4(job_id, row)
        except RuntimeError as e:
            raise HTTPException(500, str(e)) from e
        media, name = "video/mp4", f"{job_id}.mp4"
    if not path or not Path(path).exists():
        raise HTTPException(500, "output file missing")
    return FileResponse(path, media_type=media, filename=name)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    row = get_job_row(job_id)
    if row is None:
        raise HTTPException(404, "unknown job id")
    _delete_job_files(row)
    delete_job_row(job_id)
    log.info("job %s deleted", job_id)
    return {"deleted": True}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "title": C.API_TITLE}
