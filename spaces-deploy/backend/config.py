"""VoiceClean v2 — every tunable constant lives here.

Nothing in pipeline/, app/ or eval/ may hard-code a DSP/ML constant;
import it from this module instead.
"""

# ----------------------------------------------------------------------------
# Global audio format
# ----------------------------------------------------------------------------
SAMPLE_RATE = 48000          # Hz — the one true internal rate
CHANNELS = 1                 # mono internally

# ----------------------------------------------------------------------------
# Stage 0 — ingest
# ----------------------------------------------------------------------------
INGEST_CODEC = "pcm_f32le"
STEREO_IMBALANCE_DB = 20.0   # if one stereo ch is this much quieter, use louder
NOISE_FLOOR_FRAME = 2048     # samples, for noise-floor percentile
NOISE_FLOOR_HOP = 1024
NOISE_FLOOR_PERCENTILE = 10  # 10th-percentile frame RMS, dBFS
MAX_FILE_MB = 500
MAX_DURATION_MIN = 60

# ----------------------------------------------------------------------------
# Stage 1 — linear cleanup (zero-phase)
# ----------------------------------------------------------------------------
HP_CUTOFF_HZ = 80.0
HP_ORDER = 4
HUM_FREQS = (50.0, 60.0)     # base hum frequencies to hunt
HUM_MAX_HARM_HZ = 500.0
HUM_PEAK_ABOVE_NEIGHBOUR_DB = 12.0
HUM_NOTCH_Q = 30.0
HUM_SPECTRUM_NFFT = 8192

# ----------------------------------------------------------------------------
# Stage 2 — VAD (Silero)
# ----------------------------------------------------------------------------
VAD_INTERNAL_SR = 16000
VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_MS = 250
VAD_MIN_SILENCE_MS = 100
VAD_PAD_MS = 30
VAD_CHUNK_SAMPLES = 512      # silero 16 kHz chunk size

# ----------------------------------------------------------------------------
# Stage 3 — pre-separation
# ----------------------------------------------------------------------------
# Demucs
DEMUCS_MODEL = "htdemucs"
DEMUCS_SR = 44100            # demucs native rate; we resample to/from 48 k
# Target-speaker isolation
SPK_WINDOW_S = 1.0
SPK_HOP_S = 0.25
SPK_EMBED_WINDOW_S = 1.5
SPK_THETA = 0.5              # cosine-similarity decision point
SPK_TAU = 0.08               # sigmoid temperature
SPK_MEDIAN_WIN = 5           # median filter over windows
SPK_GAIN_FLOOR_DB = -40.0
SPK_RAMP_MS = 30.0           # minimum gain ramp to avoid clicks
SPK_AUTOREF_SPAN_S = 10.0    # auto reference: highest-SNR 10 s span

# ----------------------------------------------------------------------------
# Stage 4 — neural denoise (DeepFilterNet3)
# ----------------------------------------------------------------------------
DF_FFT = 960                 # 20 ms @ 48 kHz
DF_HOP = 480                 # 10 ms @ 48 kHz
DF_MODEL_NAME = "DeepFilterNet3"


def atten_cap_db(strength: float) -> float:
    """Attenuation cap A(s) = 10 + 50*s dB, s in [0,1]."""
    s = min(1.0, max(0.0, strength))
    return 10.0 + 50.0 * s


DEFAULT_STRENGTH = 0.6
STUDIO_STRENGTH = 0.8

# Chunked inference for long files
DF_CHUNK_S = 60.0
DF_OVERLAP_S = 2.0
DF_WARMUP_S = 1.0

# ----------------------------------------------------------------------------
# Stage 5 — dereverb (WPE)
# ----------------------------------------------------------------------------
WPE_DELAY = 3                # frames
WPE_TAPS = 12
WPE_ITERATIONS = 3
WPE_NFFT = 1024
WPE_HOP = 256

# ----------------------------------------------------------------------------
# Stage 6 — classical post-filter (MMSE-LSA)
# ----------------------------------------------------------------------------
PF_NFFT = 1024
PF_HOP = 256                 # 75% overlap, Hann
PF_A_D = 0.95                # noise PSD smoothing when speech present
PF_A_DD = 0.98               # decision-directed xi smoothing
PF_XI_FLOOR_DB = -25.0
PF_GAIN_FLOOR_DB = -12.0
PF_TAU_ATT_MS = 5.0
PF_TAU_REL_MS = 100.0
PF_FREQ_KERNEL = (0.25, 0.5, 0.25)
PF_MEDIAN_SIZE = 3           # 3x3 median on gains
PF_HARM_FLOOR = 0.7          # harmonic protection gain floor
PF_SIB_FLOOR = 0.8           # sibilant protection gain floor
PF_SIB_RATIO = 0.35          # E(4-10k)/E(0-10k) threshold
PF_SIB_FLATNESS = 0.4        # spectral flatness of 4-10 kHz
PF_F0_MIN = 60.0
PF_F0_MAX = 400.0
PF_HARM_MAX_HZ = 8000.0
PF_SIB_BAND = (4000.0, 10000.0)
PF_FULL_BAND = (0.0, 10000.0)

# ----------------------------------------------------------------------------
# Stage 7 — room tone
# ----------------------------------------------------------------------------
ROOM_TONE_OFF = "off"
ROOM_TONE_SUBTLE_DBFS = -68.0
ROOM_TONE_NATURAL_DBFS = -60.0
DEFAULT_ROOM_TONE = "subtle"  # off | subtle | natural

# ----------------------------------------------------------------------------
# Stage 8 — tone & dynamics (Studio mode)
# ----------------------------------------------------------------------------
PRESENCE_FREQ = 3500.0
PRESENCE_GAIN_DB = 2.0
PRESENCE_Q = 0.9
DEESS_BAND = (5000.0, 9000.0)
DEESS_ENV_MS = 5.0
DEESS_THRESH_ABOVE_MEDIAN_DB = 6.0
DEESS_MAX_DB = 6.0
COMP_THRESH_DBFS = -20.0
COMP_RATIO = 2.5
COMP_KNEE_DB = 6.0
COMP_ATTACK_MS = 8.0
COMP_RELEASE_MS = 120.0

# ----------------------------------------------------------------------------
# Stage 9 — loudness & export
# ----------------------------------------------------------------------------
LOUDNESS_TARGET = -16.0      # LUFS; options -14 / -23
LOUDNESS_OPTIONS = (-14.0, -16.0, -23.0)
TRUE_PEAK_DBTP = -1.0
LIMITER_LOOKAHEAD_MS = 5.0
LIMITER_RELEASE_MS = 50.0
LIMITER_OVERSAMPLE = 4
EXPORT_WAV_SR = 48000
MP3_BITRATE = "320k"
AAC_BITRATE = "256k"

# Long-file chunking (pipeline level not needed; DF handles it)
LONG_FILE_MIN_S = 600.0

# ----------------------------------------------------------------------------
# Modes -> stage lists
# ----------------------------------------------------------------------------
MODES = ("standard", "music", "isolate", "studio")

# ----------------------------------------------------------------------------
# API / jobs
# ----------------------------------------------------------------------------
JOB_DB_PATH = "jobs.sqlite3"
JOB_TTL_S = 3600             # auto-delete files after 1 hour
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
API_TITLE = "VoiceClean v2"
