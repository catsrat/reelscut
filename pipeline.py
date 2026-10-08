"""
Clip Reels — turn a long video into short vertical reels.

Pipeline:
  YouTube URL
    -> download (yt-dlp)
    -> extract 16kHz audio (ffmpeg)
    -> transcribe with timestamps (whisper.cpp)
    -> pick the best moments (Claude, or a heuristic fallback if no API key)
    -> cut + crop to vertical 9:16 + burn captions (ffmpeg)
    -> downloadable .mp4 reels

Everything here is plain function calls so app.py can drive it from a
background thread and report progress.
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import functools
import glob
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
# Fast English-only model, and a larger multilingual model for other languages.
# "medium" is much better than "small" for Indian languages (Telugu/Hindi/Tamil);
# it falls back to "small" if medium hasn't been downloaded.
MODEL_PATH = os.path.join(ROOT, "models", "ggml-base.en.bin")
_MEDIUM = os.path.join(ROOT, "models", "ggml-medium.bin")
_SMALL = os.path.join(ROOT, "models", "ggml-small.bin")
# Prefer "small" for non-English: ~3x faster than medium on this GPU, and local
# Telugu quality is poor at any size anyway (real fix = an Indian-language STT API).
MULTILINGUAL_MODEL = _SMALL if os.path.exists(_SMALL) else _MEDIUM

# whisper.cpp speed: use most cores + flash attention (big speedup, ~same accuracy).
WHISPER_THREADS = max(4, (os.cpu_count() or 4) - 2)

# Sarvam AI — accurate Indian-language transcription (Telugu/Hindi/Tamil).
# REST limit is <30s/call and timing is chunk-level, so we split the audio into
# short chunks and transcribe each; English keeps the precise local word timing.
SARVAM_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_CHUNK_SECS = 20      # bigger chunks = fewer API calls (REST limit is <30s)
SARVAM_WORKERS = 2          # low concurrency to stay under rate limits
_SARVAM_LANG = {"te": "te-IN", "hi": "hi-IN", "ta": "ta-IN", "en": "en-IN"}

# Target each clip at this length window (seconds). Short = better for
# Shorts/Reels/TikTok. Claude aims for the sweet spot; fallback uses these.
# Let content decide length: short when punchy, longer when a thought needs it.
# (Jump-cut dead-space removal tightens the final clip afterward.)
MIN_CLIP = 15
MAX_CLIP = 90
TARGET_CLIP = 40

# Moods Claude can tag a clip with; each maps to a music/<mood>/ folder.
MOODS = ["upbeat", "calm", "dramatic", "inspirational", "funny", "neutral"]


def _proxy_args():
    """Route yt-dlp through a residential proxy if YTDLP_PROXY is set.

    Residential/mobile proxies make YouTube see a home IP, which gets past the
    data-center block. Format: http://user:pass@host:port (or socks5://...).
    """
    proxy = os.environ.get("YTDLP_PROXY", "").strip()
    return ["--proxy", proxy] if proxy else []


def _is_gdrive(url):
    """True for Google Drive share links. Drive doesn't block servers the way
    YouTube does, so these download for free with no proxy and no bot checks."""
    u = (url or "").lower()
    return "drive.google.com" in u or "docs.google.com" in u


def _tool(name):
    """Resolve an external tool: PATH first (Docker/Linux/Mac, or a Windows
    install), then the bundled Windows binaries in bin/."""
    found = shutil.which(name)
    if found:
        return found
    local = os.path.join(ROOT, "bin", name + (".exe" if os.name == "nt" else ""))
    return local if os.path.exists(local) else name


def ytdlp_cmd():
    """yt-dlp as `python -m yt_dlp` from this interpreter when installed — works
    the same on every OS and avoids Windows blocking the yt-dlp.exe shim."""
    if importlib.util.find_spec("yt_dlp"):
        return [sys.executable, "-m", "yt_dlp"]
    return ["yt-dlp"]


def ytdlp_available():
    return bool(importlib.util.find_spec("yt_dlp") or shutil.which("yt-dlp"))


def _redact(text):
    """Hide credentials embedded in URLs (e.g. the YTDLP_PROXY user:pass) so
    they never reach the logs or the UI's error message."""
    return re.sub(r"://[^/\s:@]+:[^/\s@]+@", "://***@", text)


def _run(cmd, cwd=None, timeout=None):
    """Run a command, raising with captured output if it fails or times out.

    Always an argument list, never shell=True: URLs and filenames come from
    users, and a shell would let a crafted link run commands on the server."""
    cmd = [str(c) for c in cmd]
    cmd[0] = _tool(cmd[0])
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Timed out after {timeout}s: {' '.join(cmd[:2])}")
    if proc.returncode != 0:
        raise RuntimeError(_redact(
            f"Command failed: {' '.join(cmd)}\n{proc.stderr[-2000:]}"
        ))
    return proc.stdout


# A slow machine still advances ffmpeg's output clock; a hung ffmpeg doesn't.
FFMPEG_STALL_SECS = 600


class FFmpegStalled(RuntimeError):
    pass


def _run_ffmpeg(cmd, cwd=None, expected=None, on_progress=None):
    """Run ffmpeg, reporting progress as a 0..1 fraction of `expected` output
    seconds, and kill it if its output clock stops moving for
    FFMPEG_STALL_SECS (stuck, as opposed to merely slow)."""
    import tempfile
    import threading

    cmd = [str(c) for c in cmd]
    cmd = [_tool(cmd[0]), "-progress", "pipe:1", "-nostats", *cmd[1:]]
    state = {"moved": time.time(), "secs": -1.0}
    stalled = threading.Event()
    with tempfile.TemporaryFile() as errf:  # a file, so a chatty stderr can't block the pipe
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=errf,
                                text=True, errors="replace")

        def watchdog():
            while proc.poll() is None:
                if time.time() - state["moved"] > FFMPEG_STALL_SECS:
                    stalled.set()
                    proc.kill()
                    return
                time.sleep(5)

        threading.Thread(target=watchdog, daemon=True).start()
        for line in proc.stdout:
            if not line.startswith("out_time_us="):
                continue
            try:
                secs = int(line.split("=", 1)[1]) / 1e6
            except ValueError:  # "N/A" before the first frame
                continue
            if secs > state["secs"]:
                state["secs"], state["moved"] = secs, time.time()
                if expected and on_progress:
                    on_progress(max(0.0, min(1.0, secs / expected)))
        proc.wait()
        if stalled.is_set():
            raise FFmpegStalled(
                f"Rendering stalled (no progress for {FFMPEG_STALL_SECS // 60} minutes).")
        if proc.returncode != 0:
            errf.seek(0)
            err = errf.read().decode("utf-8", "replace")
            raise RuntimeError(_redact(f"Command failed: {' '.join(cmd)}\n{err[-2000:]}"))


# ---------------------------------------------------------------- download

# Cap how long a video we'll process. Long videos are slow to transcribe on
# CPU, but we allow up to ~3h. ~1 clip per CLIP_EVERY_MINUTES of video.
MAX_VIDEO_MINUTES = 100
CLIP_EVERY_MINUTES = 4
MAX_CLIPS_CAP = 30
MUSIC_DIR = os.path.join(ROOT, "music")
BROLL_DIR = os.path.join(ROOT, "broll")  # gameplay/b-roll for split-screen mode


def get_title(url):
    # Drive needs no proxy; skip it so a flaky proxy can't slow the metadata call.
    proxy = [] if _is_gdrive(url) else _proxy_args()
    try:
        bin_to_use = ytdlp_cmd()
        out = _run(bin_to_use + ["--no-warnings", *proxy, "--print", "title", "--", url], timeout=70)
        return out.strip() or "Untitled video"
    except Exception:
        return "Untitled video"


def get_duration(url):
    """Return video length in seconds, or None if it can't be determined."""
    proxy = [] if _is_gdrive(url) else _proxy_args()
    try:
        bin_to_use = ytdlp_cmd()
        out = _run(bin_to_use + ["--no-warnings", *proxy, "--print", "duration", "--", url], timeout=70)
        return float(out.strip())
    except Exception:
        return None


def _probe_duration(path):
    """Duration of a local media file in seconds, or None."""
    try:
        out = _run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", path
        ]).strip()
        return float(out)
    except Exception:
        return None


def download_video(url, workdir):
    """Download the video as an mp4 into workdir, return its path.

    Caps at 1080p: a vertical 9:16 crop only keeps the centre ~600px of a
    landscape frame, so 720p sources looked soft after upscaling to 1080-wide.
    Retries and parallel fragments make flaky connections less fatal.
    """
    out_tmpl = os.path.join(workdir, "source.%(ext)s")
    bin_to_use = ytdlp_cmd()

    if _is_gdrive(url):
        # Google Drive: a single shared file, no proxy and no bot checks needed.
        # `best` just grabs the file as-is. Works for "Anyone with the link" files.
        cmd = bin_to_use + [
            "--no-warnings",
            "-f", "best",
            "--retries", "10",
            "--socket-timeout", "60",
            "-o", out_tmpl,
            "--", url,  # "--": a link can never be read as a yt-dlp option
        ]
        try:
            _run(cmd, timeout=900)
        except RuntimeError as e:
            msg = str(e).lower()
            if "permission" in msg or "not be downloaded" in msg or "quota" in msg \
                    or "private" in msg or "access" in msg:
                raise RuntimeError(
                    "Couldn't download this Google Drive file. Make sure it's "
                    "shared as “Anyone with the link” (Share → General "
                    "access → Anyone with the link), then paste the link again."
                )
            raise
    else:
        cmd = bin_to_use + [
            "--no-warnings",
            *_proxy_args(),
            "-f", "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b/best",
            # Try phone/TV YouTube clients — they often return video formats when
            # the default web client is blocked (the "format not available" error).
            "--extractor-args", "youtube:player_client=ios,tv,android,web",
            "--merge-output-format", "mp4",
            "--retries", "15",
            "--fragment-retries", "20",
            "--extractor-retries", "5",
            "--concurrent-fragments", "2",
            "--socket-timeout", "60",
            "-o", out_tmpl,
        ]
        # YouTube increasingly blocks anonymous downloads ("confirm you're not a
        # bot"). Cookies get past it:
        #  - local dev: YT_COOKIES_BROWSER=chrome (borrow the logged-in browser)
        #  - server: YTDLP_COOKIES_FILE=/path/to/cookies.txt (exported cookies)
        cookies_file = os.environ.get("YTDLP_COOKIES_FILE", "").strip()
        browser = os.environ.get("YT_COOKIES_BROWSER", "").strip()
        if cookies_file and os.path.exists(cookies_file):
            # yt-dlp rewrites the cookie jar on exit, but Render Secret Files are
            # read-only — copy to a writable path first.
            import shutil as _sh
            writable = os.path.join(workdir, "cookies.txt")
            try:
                _sh.copyfile(cookies_file, writable)
                cmd += ["--cookies", writable]
            except Exception:
                cmd += ["--cookies", cookies_file]
        elif browser:
            cmd += ["--cookies-from-browser", browser]
        cmd += ["--", url]  # "--": a link can never be read as a yt-dlp option
        try:
            _run(cmd, timeout=900)
        except RuntimeError as e:
            msg = str(e).lower()
            if "confirm you" in msg or "bot" in msg or "format is not available" in msg:
                raise RuntimeError(
                    "YouTube blocked this download from the server. A residential "
                    "proxy is required — set YTDLP_PROXY (http://user:pass@host:port) "
                    "in the environment. (Or upload the file, or paste a Google "
                    "Drive link instead — Drive needs no proxy.)"
                )
            raise
    matches = glob.glob(os.path.join(workdir, "source.*"))
    # Prefer mp4 if multiple intermediate files exist.
    mp4 = [m for m in matches if m.endswith(".mp4")]
    if mp4:
        return mp4[0]
    if matches:
        return matches[0]
    raise RuntimeError("Download produced no file.")


# ---------------------------------------------------------------- transcribe

def extract_audio(video_path, workdir):
    audio = os.path.join(workdir, "audio.wav")
    _run([
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        audio,
    ])
    return audio


def transcribe(audio_path, workdir, model_path=MODEL_PATH, language="en"):
    """Run whisper.cpp at WORD level, return a list of {start, end, text} words.

    Word-level timing (-ml 1 --split-on-word) powers the animated captions;
    group_phrases() rebuilds sentence-ish phrases for moment selection.
    """
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Whisper model not found at {model_path}."
        )
    out_base = os.path.join(workdir, "transcript")
    cmd = [
        "whisper-cli",
        "-m", model_path,
        "-f", audio_path,
        "-oj",                 # JSON with timestamps
        "-of", out_base,
        "-pp",                 # no live progress printing noise
        "-ml", "1", "-sow",    # one word per segment, split on word boundaries
        "-t", str(WHISPER_THREADS),  # use most cores
        "-fa",                 # flash attention — faster, ~same accuracy
    ]
    # base.en is English-only; for the multilingual model pass the language
    # ("te", "hi", ...) or "auto" to detect it.
    if language and language != "en":
        cmd += ["-l", language]
    try:
        _run(cmd, timeout=1800)
    except (RuntimeError, FileNotFoundError, OSError) as e:
        # Fallback to openai-whisper (optional: `pip install openai-whisper`)
        # if the whisper.cpp binary is blocked or not found.
        try:
            import whisper
            # Use "base" as it matches the ggml-base.en.bin intent
            model = whisper.load_model("base")
            result = model.transcribe(audio_path, word_timestamps=True, language=None if language == "auto" else language)
        except Exception as fallback_e:
            raise RuntimeError(f"Primary transcription failed ({e}) and fallback also failed ({fallback_e})") from e
        words = []
        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                words.append({
                    "start": w.get("start", 0),
                    "end": w.get("end", 0),
                    "text": w.get("word", "").strip(),
                })
        words = [w for w in words if w["text"]]
        if not words:
            raise RuntimeError("Transcription returned no speech segments.")
        return words
    # whisper.cpp can split a multi-byte character across segments, producing
    # invalid UTF-8 in the JSON for non-Latin scripts — decode tolerantly.
    with open(out_base + ".json", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    words = []
    for seg in data.get("transcription", []):
        off = seg.get("offsets", {})
        text = seg.get("text", "").strip()
        if not text:
            continue
        words.append({
            "start": off.get("from", 0) / 1000.0,
            "end": off.get("to", 0) / 1000.0,
            "text": text,
        })
    if not words:
        raise RuntimeError("Transcription returned no speech segments.")
    return words


def transcribe_groq(audio_path, workdir, language="en"):
    """Transcribe via Groq (free tier, high speed). Returns word list.
    Asks for word-level timestamps; if a response only has segments, words are
    spread evenly across each segment instead.
    """
    from groq import Groq
    client = Groq(timeout=300)  # uses GROQ_API_KEY
    # Always send small speech-grade audio: 16 kHz mono MP3 at 32 kbps is
    # ~0.24 MB/min (a 23-min video ≈ 5.5 MB vs 44 MB of WAV), so the upload is
    # quick and even a 100-minute video stays under Groq's 25 MB limit.
    upload_path = os.path.join(workdir, "audio_groq.mp3")
    _run(["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000",
          "-codec:a", "libmp3lame", "-b:a", "32k", upload_path])

    with open(upload_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(upload_path), f),
            # Turbo is several times faster and near-identical on English;
            # keep the full model for other languages.
            model="whisper-large-v3-turbo" if language == "en" else "whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            language=language if language != "auto" else None,
        )

    # The SDK returns segments/words as plain dicts (not attribute objects).
    def get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    words = [
        {"start": float(get(w, "start")), "end": float(get(w, "end")),
         "text": (get(w, "word") or "").strip()}
        for w in (get(transcription, "words") or [])
    ]
    words = [w for w in words if w["text"]]
    if words:
        return words

    for segment in get(transcription, "segments") or []:
        start = float(get(segment, "start"))
        end = float(get(segment, "end"))
        toks = (get(segment, "text") or "").split()
        if not toks:
            continue
        # Distribute words evenly across the segment duration
        per_word = (end - start) / len(toks)
        for i, tok in enumerate(toks):
            words.append({
                "start": start + (i * per_word),
                "end": start + ((i + 1) * per_word),
                "text": tok,
            })
    return words

def group_phrases(words, max_gap=0.7, max_words=12):
    """Group word-level entries into sentence-ish phrases for moment selection."""
    phrases = []
    cur = []
    for w in words:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            ends_sentence = cur[-1]["text"][-1:] in ".?!।"
            if gap > max_gap or ends_sentence or len(cur) >= max_words:
                phrases.append(_phrase(cur))
                cur = []
        cur.append(w)
    if cur:
        phrases.append(_phrase(cur))
    return phrases


def _phrase(group):
    return {
        "start": group[0]["start"],
        "end": group[-1]["end"],
        "text": " ".join(w["text"] for w in group),
    }


def transcribe_sarvam(audio_path, workdir, language, api_key, progress=None):
    """Transcribe via Sarvam (accurate for Indian languages). Returns word list.

    Sarvam's REST API caps at <30s/call with chunk-level timing, so we split the
    audio into short chunks, transcribe each, and spread the words evenly across
    the chunk's time window (good enough to cut clips and roughly sync captions).
    """
    import glob as _glob
    import httpx
    import concurrent.futures

    seg_dir = os.path.join(workdir, "sarvam_segments")
    os.makedirs(seg_dir, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-i", audio_path,
        "-f", "segment", "-segment_time", str(SARVAM_CHUNK_SECS),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        os.path.join(seg_dir, "seg_%05d.wav"),
    ])
    segs = sorted(_glob.glob(os.path.join(seg_dir, "seg_*.wav")))
    if not segs:
        raise RuntimeError("Could not split audio for Sarvam transcription.")

    lang_code = _SARVAM_LANG.get(language, "unknown")
    headers = {"api-subscription-key": api_key}

    def do_seg(item):
        idx, seg = item
        last_err = None
        for attempt in range(7):
            try:
                with open(seg, "rb") as f:
                    r = httpx.post(
                        SARVAM_URL, headers=headers,
                        data={"model": "saarika:v2.5", "language_code": lang_code,
                              "with_timestamps": "true"},
                        files={"file": (os.path.basename(seg), f, "audio/wav")},
                        timeout=120,
                    )
                if r.status_code == 200:
                    return idx, (r.json().get("transcript") or "").strip()
                if r.status_code in (401, 403):
                    raise RuntimeError(
                        f"Sarvam rejected the API key (HTTP {r.status_code}). "
                        "Check SARVAM_API_KEY."
                    )
                if r.status_code == 429:  # rate limited — honour retry-after
                    wait = float(r.headers.get("retry-after") or 0) or min(30, 3 * (attempt + 1))
                    last_err = "rate limit (429)"
                    time.sleep(wait)
                    continue
                last_err = f"HTTP {r.status_code}: {r.text[:160]}"
            except RuntimeError:
                raise
            except Exception as e:
                last_err = str(e)
            time.sleep(min(20, 3 * (attempt + 1)))  # backoff on network/other
        raise RuntimeError(
            f"Sarvam API error after retries: {last_err}. The free tier rate "
            "limit may be too low — wait a minute and retry, or upgrade the plan."
        )

    texts = [""] * len(segs)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=SARVAM_WORKERS) as ex:
        futures = [ex.submit(do_seg, (i, s)) for i, s in enumerate(segs)]
        for fut in concurrent.futures.as_completed(futures):
            idx, txt = fut.result()
            texts[idx] = txt
            done += 1
            if progress and done % 3 == 0:
                progress(
                    45 + int(22 * done / len(segs)),
                    f"Transcribing with Sarvam ({done}/{len(segs)} chunks)...",
                )

    # Spread each chunk's words evenly across its time window.
    words = []
    for idx, txt in enumerate(texts):
        toks = txt.split()
        if not toks:
            continue
        offset = idx * SARVAM_CHUNK_SECS
        per = SARVAM_CHUNK_SECS / len(toks)
        for k, tok in enumerate(toks):
            words.append({
                "start": offset + k * per,
                "end": offset + (k + 1) * per,
                "text": tok,
            })
    if not words:
        raise RuntimeError(
            "Sarvam returned no speech — the audio may be mostly music/silence."
        )
    return words


# ---------------------------------------------------------------- pick moments

def _transcript_for_prompt(segments):
    lines = []
    for s in segments:
        lines.append(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}")
    return "\n".join(lines)


def pick_moments_ai(segments, api_key, max_clips=5, min_clip=None, max_clip=None):
    """Ask Claude to choose the most engaging moments. Returns list of clips.

    min_clip/max_clip (seconds) hard-bound the length when a clipping campaign
    requires it — the clips MUST then fall within those bounds to get paid."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    transcript = _transcript_for_prompt(segments)
    total = segments[-1]["end"]

    lo = int(min_clip) if min_clip else MIN_CLIP
    hi = int(max_clip) if max_clip else MAX_CLIP
    if min_clip or max_clip:
        length_rule = (
            f"LENGTH — STRICT CAMPAIGN REQUIREMENT (the clipper is NOT PAID if a "
            f"clip falls outside this): every clip MUST be at least {lo} seconds "
            f"and at most {hi} seconds. Choose start/end so each clip is within "
            f"{lo}-{hi}s while still running hook -> full payoff.\n\n"
        )
    else:
        length_rule = (
            f"LENGTH — let the content decide, do not force a number:\n"
            f"- Aim for {lo}-60 seconds; go up to {hi} only if needed to "
            "land the full payoff. Never pad, and never cut a thought short just to "
            "hit a target. A complete 55-second story beats a truncated 25-second one.\n"
            "- Choose start/end so the clip runs from the hook through the payoff.\n\n"
        )

    system = (
        "You are a world-class short-form video editor and viral content "
        "strategist. You are given a timestamped transcript of a long video. "
        "Plan the clips that will perform best as standalone vertical reels "
        "(TikTok / Reels / Shorts).\n\n"
        "WHAT MAKES A GREAT CLIP:\n"
        "- HOOK: it opens with a strong hook in the first ~3 seconds — a bold "
        "claim, a question, or an intriguing setup that makes people stop scrolling.\n"
        "- COMPLETE PAYOFF: the clip MUST contain the full payoff — the answer, "
        "punchline, reveal, lesson, or reason. NEVER cut off before the "
        "interesting part is delivered. If a speaker sets up 'the reason I quit "
        "was…' or 'and then the craziest thing happened…', the clip MUST include "
        "the actual reason / what happened. A clip that ends on the setup is a "
        "FAILED clip.\n"
        "- SELF-CONTAINED: it makes sense on its own, without the rest of the video.\n"
        "- WORTH SHARING: emotional, surprising, funny, controversial, or genuinely useful.\n\n"
        + length_rule +
        "FOR EACH CLIP:\n"
        "- start/end: real timestamps from the transcript, start < end.\n"
        "- title: punchy, scroll-stopping, max 60 chars, in the video's language.\n"
        f"- mood: one of {', '.join(MOODS)}.\n"
        "- reason: one line on why it will perform.\n\n"
        f"Pick up to {max_clips} clips, best first. Quality over quantity — only "
        "include genuinely strong moments."
    )
    user = (
        f"Video length: {total:.0f} seconds.\n\n"
        f"Transcript:\n{transcript}"
    )

    schema = {
        "type": "object",
        "properties": {
            "clips": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "title": {"type": "string"},
                        "reason": {"type": "string"},
                        "mood": {"type": "string", "enum": MOODS},
                    },
                    "required": ["start", "end", "title", "reason", "mood"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["clips"],
        "additionalProperties": False,
    }

    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    text = next(b.text for b in resp.content if b.type == "text")
    clips = json.loads(text).get("clips", [])
    return _sanitize_clips(clips, segments, max_clip=max_clip)


# Words that tend to mark the exciting part of a stream / talk. Only used when
# there is no AI to read the transcript (no key, out of credits, outage).
HYPE_WORDS = {
    "wow", "omg", "crazy", "craziest", "huge", "massive", "holy", "haha",
    "hahaha", "lol", "million", "thousand", "hundred", "biggest", "let's", "lets",
    "secret", "never", "shocking",
}
# Payoff words: the moment something lands (a win, a reveal) — what a clip
# should end on. Weighted above general hype.
PAYOFF_WORDS = {
    "yes", "won", "win", "winning", "finally", "jackpot", "beautiful", "insane",
    "unbelievable", "incredible",
}
# Big numbers ("22 000", "200,000", "50x", "6k"): money, stats and multipliers
# are where stream and talk highlights tend to be.
_BIG_NUMBER = re.compile(r"^\$?\d+(x|k)$|^\$?\d{3,}$|^\$?\d{1,3}(,\d{3})+$")


def _loudness_per_second(audio_path, n):
    """RMS loudness in dB for each second of a 16-bit WAV, or None.
    Read a second at a time so a long video doesn't load into memory at once."""
    import wave

    import numpy as np
    try:
        with wave.open(audio_path, "rb") as w:
            if w.getsampwidth() != 2:
                return None
            rate, ch = w.getframerate(), w.getnchannels()
            out = np.zeros(n, dtype=np.float32)
            got = 0
            for i in range(n):
                buf = w.readframes(rate)
                if not buf:
                    break
                x = np.frombuffer(buf, dtype=np.int16)[::ch].astype(np.float32)
                out[i] = 20 * np.log10(np.sqrt((x * x).mean()) + 1e-6)
                got = i + 1
    except Exception:
        return None
    if got < 5:
        return None
    out[got:] = out[:got].min()
    return out


def _zscore(a):
    sd = a.std()
    return (a - a.mean()) / sd if sd > 1e-6 else a * 0


def _smooth(a, secs):
    import numpy as np
    return np.convolve(a, np.ones(secs) / secs, mode="same")


def _auto_title(segments, start, end, peak, lift):
    """Title from the most exciting line in the clip (most hype / payoff / big
    numbers, ties broken by closeness to the peak), repeats collapsed:
    'break break break we got it' -> 'Break we got it'."""
    inside = [s for s in segments if s["start"] >= start and s["end"] <= end] or segments
    n = len(lift)

    def excite(s):
        a, b = max(0, int(s["start"])), min(n, int(s["end"]) + 1)
        return float(lift[a:b].sum()) if b > a else 0.0

    best = max(inside, key=lambda s: (excite(s), -abs((s["start"] + s["end"]) / 2 - peak)))
    out = []
    for w in best["text"].split():
        if not out or w.lower().strip(".,?!") != out[-1].lower().strip(".,?!"):
            out.append(w)
    title = " ".join(out).strip() or "Clip"
    title = (title[:57] + "...") if len(title) > 60 else title
    return title[:1].upper() + title[1:]


def pick_moments_heuristic(segments, max_clips=5, words=None, audio_path=None,
                           min_clip=None, max_clip=None):
    """Pick moments without AI: score every second of the WHOLE video for
    excitement — loud reactions (vs the video's own average), fast talking,
    hype words, numbers, repeated shouts — then build a clip around each of
    the strongest peaks: ~20s of setup before it, the reaction after, cut on
    phrase boundaries. Not as smart as the AI picker, but finds the
    high-energy moments a stream clipper is usually after."""
    import bisect

    import numpy as np

    if not segments:
        return []
    total = segments[-1]["end"]
    n = int(total) + 1
    words = words or []

    score = np.zeros(n, dtype=np.float32)
    loud = _loudness_per_second(audio_path, n) if audio_path else None
    if loud is not None:
        score += np.clip(_zscore(loud), -1.5, 3.0)

    # Weights were tuned on a real 23-minute stream: these settings found its
    # three biggest moments (a 50x, a $22K win, a $200K near miss) with no AI.
    # Repeated chanting ("break break break") is deliberately NOT rewarded —
    # it happens on every spin; the payoff right after it is the highlight.
    rate = np.zeros(n, dtype=np.float32)
    lift = np.zeros(n, dtype=np.float32)  # hype + payoff + big numbers
    toks = [w["text"].strip().lower().strip(".,?!\"'") for w in words]
    for j, w in enumerate(words):
        i = min(n - 1, max(0, int(w["start"])))
        rate[i] += 1
        tok, raw = toks[j], w["text"]
        nxt = toks[j + 1] if j + 1 < len(toks) else ""
        if tok in HYPE_WORDS or "!" in raw:
            lift[i] += 1.0
        if tok in PAYOFF_WORDS:
            lift[i] += 0.75
        if _BIG_NUMBER.match(tok) or (tok.isdigit() and nxt in ("000", "thousand", "grand", "k")):
            lift[i] += 1.5
    if loud is not None:
        score *= 0.5
    if words:
        score += 0.3 * np.clip(_zscore(_smooth(rate, 3)), -1.5, 2.5)
        score += _smooth(lift, 3)
    score = _smooth(score, 5)

    lo = float(min_clip) if min_clip else MIN_CLIP
    hi = float(max_clip) if max_clip else 45.0
    starts = [s["start"] for s in segments]
    ends = [s["end"] for s in segments]

    def phrase_start_at_or_before(t):
        i = bisect.bisect_right(starts, t) - 1
        return starts[i] if i >= 0 else 0.0

    def phrase_end_at_or_after(t):
        i = bisect.bisect_left(ends, t)
        return ends[i] if i < len(ends) else total

    clips, taken = [], []
    for peak in np.argsort(-score):
        if len(clips) >= max_clips:
            break
        p = float(peak)
        if any(a - 5 <= p <= b + 5 for a, b in taken):
            continue
        start = phrase_start_at_or_before(max(0.0, p - 18))
        end = phrase_end_at_or_after(min(total, p + 12))
        if end - start < lo:
            end = phrase_end_at_or_after(min(total, start + lo))
        if end - start > hi:  # keep the climax, trim the setup
            i = bisect.bisect_left(starts, end - hi)
            start = starts[i] if i < len(starts) and starts[i] < end - lo else end - hi
        if end - start < min(lo, 5) or any(not (end <= a or start >= b) for a, b in taken):
            continue
        taken.append((start, end))
        title = _auto_title(segments, start, end, p, lift)
        clips.append({
            "start": start,
            "end": end,
            "title": title[:1].upper() + title[1:],
            "reason": "Auto-selected: high-energy moment.",
            "mood": "neutral",
        })
    return _sanitize_clips(clips, segments, max_clip=max_clip)


def parse_campaign_rules(text, api_key):
    """Read a clipping campaign's pasted rules / Google-Doc terms and return the
    structured requirements, so the tool can auto-build a compliant clip and give
    the clipper a 'payment-safe' checklist. Returns a dict (see schema)."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    system = (
        "You help social-media 'clippers' who get PAID through clipping campaigns "
        "(e.g. on Whop). You are given a campaign's rules / terms (pasted from a "
        "Google Doc). Extract the requirements precisely so a video tool can "
        "auto-build a COMPLIANT clip and the clipper never misses a payment.\n\n"
        "Return:\n"
        "- min_length_sec / max_length_sec: required clip length in seconds, or 0 "
        "if not specified.\n"
        "- language: spoken/caption language if specified — one of en, te, hi, ta "
        "— else \"\".\n"
        "- captions_required: true if captions/subtitles are required.\n"
        "- logo_required: true if a watermark / logo / handle overlay is required.\n"
        "- hashtags: every required hashtag, each starting with #.\n"
        "- mentions: every required @mention / handle to tag, each starting with @.\n"
        "- post_caption: a ready-to-paste caption/description for the post that "
        "obeys the caption rules and includes the required hashtags and mentions.\n"
        "- auto_handled: short bullets for each rule THIS VIDEO TOOL handles "
        "automatically (e.g. '9:16 vertical', 'clip length 30-60s', 'captions on', "
        "'logo overlay').\n"
        "- manual_todo: short bullets for rules only the clipper can do (where/when "
        "to post, which account/platform, tag in bio, follower minimums, deadlines, "
        "submitting the link on Whop, etc.).\n\n"
        "Be faithful — never invent requirements. If something is not stated, leave "
        "it 0 / \"\" / empty."
    )
    schema = {
        "type": "object",
        "properties": {
            "min_length_sec": {"type": "number"},
            "max_length_sec": {"type": "number"},
            "language": {"type": "string"},
            "captions_required": {"type": "boolean"},
            "logo_required": {"type": "boolean"},
            "hashtags": {"type": "array", "items": {"type": "string"}},
            "mentions": {"type": "array", "items": {"type": "string"}},
            "post_caption": {"type": "string"},
            "auto_handled": {"type": "array", "items": {"type": "string"}},
            "manual_todo": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "min_length_sec", "max_length_sec", "language", "captions_required",
            "logo_required", "hashtags", "mentions", "post_caption",
            "auto_handled", "manual_todo",
        ],
        "additionalProperties": False,
    }
    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": f"Campaign rules:\n\n{text}"}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    out = next(b.text for b in resp.content if b.type == "text")
    return json.loads(out)


def _sanitize_clips(clips, segments, max_clip=None):
    total = segments[-1]["end"]
    hard_max = float(max_clip) if max_clip else MAX_CLIP
    clean = []
    for c in clips:
        start = max(0.0, float(c.get("start", 0)))
        end = min(total, float(c.get("end", 0)))
        if end <= start:
            continue
        # Clamp absurd lengths (or the campaign's hard max).
        cap = hard_max if max_clip else MAX_CLIP * 1.5
        if end - start > cap:
            end = start + hard_max
        mood = (c.get("mood") or "neutral").strip().lower()
        if mood not in MOODS:
            mood = "neutral"
        clean.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "title": (c.get("title") or "Clip").strip()[:80],
            "reason": (c.get("reason") or "").strip(),
            "mood": mood,
        })
    return clean


# ---------------------------------------------------------------- render

# This Homebrew ffmpeg ships without libass/freetype, so there is no
# `subtitles` or `drawtext` filter. We render each caption to a transparent
# 1080x1920 PNG with Pillow and composite it with the `overlay` filter, which
# the build does include.

# 720x1280 (not 1080x1920): far faster to encode on a CPU server and looks
# basically identical on phones. Halves render + download time.
W, H = 720, 1280
# All fonts are bundled in fonts/ so the app renders identically on macOS and
# Linux (no dependency on OS system fonts — needed for the cloud/Docker worker).
_FONTS_DIR = os.path.join(ROOT, "fonts")


def _font(name):
    return os.path.join(_FONTS_DIR, name)


FONT_PATH = _font("NotoSans-Bold.ttf")     # Latin fallback / "classic" style
UNICODE_FONT = _font("NotoSans-Bold.ttf")  # broad fallback for unmapped scripts
FONT_SIZE = 60               # big, punchy reel-style captions (scaled for 720w)
CAPTION_CENTER_Y = 0.66      # vertical center of captions (fraction of height)
MAX_CHUNK_WORDS = 3          # words shown on screen at once
MAX_CHUNK_SECS = 1.6
CHUNK_GAP_BREAK = 0.45       # a pause longer than this starts a new chunk

# Viral caption styles (the 2026 short-form look). Each: Latin font + colour
# + whether to uppercase. Non-Latin scripts always use the per-script fonts.
_POPPINS = _font("Poppins-ExtraBold.ttf")
_ANTON = _font("Anton-Regular.ttf")
CAPTION_STYLES = {
    "bold_white":   {"font": _POPPINS, "color": (255, 255, 255), "upper": True},   # clean modern
    "yellow_punch": {"font": _POPPINS, "color": (255, 222, 0),   "upper": True},   # Hormozi-style
    "green_pop":    {"font": _POPPINS, "color": (60, 255, 90),   "upper": True},
    "tiktok_tall":  {"font": _ANTON,   "color": (255, 255, 255), "upper": True},   # tall condensed
    "mrbeast":      {"font": _ANTON,   "color": (255, 255, 255), "upper": True},   # Bold and bright
    "classic":      {"font": FONT_PATH, "color": (255, 255, 255), "upper": False}, # Arial, mixed case
}
POWER_WORDS = {"money", "secret", "viral", "ai", "hack", "free", "best", "shocking", "crazy"}
DEFAULT_STYLE = "bold_white"

# Indic scripts need a font with proper conjunct tables AND raqm (HarfBuzz)
# shaping, or consonant clusters render as broken, disjoint glyphs.
# (unicode_lo, unicode_hi, font_path, ttc_index)
_SCRIPT_FONTS = [
    (0x0C00, 0x0C7F, _font("NotoSansTelugu-Bold.ttf"), 0),       # Telugu
    (0x0900, 0x097F, _font("NotoSansDevanagari-Bold.ttf"), 0),   # Hindi/Devanagari
    (0x0B80, 0x0BFF, _font("NotoSansTamil-Bold.ttf"), 0),        # Tamil
    (0x0C80, 0x0CFF, _font("NotoSansKannada-Bold.ttf"), 0),      # Kannada
    (0x0D00, 0x0D7F, _font("NotoSansMalayalam-Bold.ttf"), 0),    # Malayalam
]


def _font_for(text, size, latin_font=FONT_PATH):
    from PIL import ImageFont
    raqm = ImageFont.Layout.RAQM

    def load(path, index=0):
        return ImageFont.truetype(path, size, index=index, layout_engine=raqm)

    # Match the first non-Latin script we see to its dedicated font.
    for ch in text:
        cp = ord(ch)
        for lo, hi, path, idx in _SCRIPT_FONTS:
            if lo <= cp <= hi:
                try:
                    return load(path, idx)
                except Exception:
                    break
    # Latin -> the chosen viral font; other non-Latin -> broad Unicode fallback.
    try:
        if all(ord(c) < 0x250 for c in text):
            return load(latin_font)
        return load(UNICODE_FONT)
    except Exception:
        return ImageFont.truetype(FONT_PATH, size)


def _caption_chunks(words, start, end, max_words=MAX_CHUNK_WORDS):
    """Group the clip's words into short on-screen chunks (times relative to clip).

    max_words=1 gives the punchy one-word-at-a-time viral caption style."""
    inside = []
    for w in words:
        if w["end"] <= start or w["start"] >= end:
            continue
        rs = max(0.0, w["start"] - start)
        re_ = min(end, w["end"]) - start
        if re_ > rs and w["text"].strip():
            inside.append({"start": rs, "end": re_, "text": w["text"].strip()})

    groups, cur = [], []
    for w in inside:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            dur = w["end"] - cur[0]["start"]
            if gap > CHUNK_GAP_BREAK or len(cur) >= max_words or dur > MAX_CHUNK_SECS:
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)

    chunks = []
    for i, g in enumerate(groups):
        ds, de = g[0]["start"], g[-1]["end"]
        # Hold each chunk until the next starts so captions don't flicker off.
        if i + 1 < len(groups):
            de = max(de, groups[i + 1][0]["start"])
        chunks.append((ds, de, " ".join(w["text"] for w in g)))
    return chunks


def _render_caption_png(text, png_path, style=None, font_size=FONT_SIZE,
                        center_y=CAPTION_CENTER_Y):
    """Render a caption onto a FULL WxH transparent frame, centred at center_y.

    Includes keyword highlighting for 'power words' to increase engagement.
    """
    from PIL import Image, ImageDraw

    style = CAPTION_STYLES.get(style or DEFAULT_STYLE, CAPTION_STYLES[DEFAULT_STYLE])
    if style["upper"] and all(ord(c) < 0x250 for c in text):
        text = text.upper()

    base_color = style["color"] + (255,)
    # Highlight color: Yellow for white style, Green for others
    highlight_color = (255, 222, 0, 255) if style == CAPTION_STYLES["bold_white"] else (60, 255, 90, 255)
    font = _font_for(text, font_size, style["font"])

    margin = 80
    max_w = W - 2 * margin
    words = text.split()

    lines = []
    cur_line = []
    cur_w = 0
    for w in words:
        w_len = font.getlength(w) + (10 if cur_line else 0)
        if cur_w + w_len <= max_w or not cur_line:
            cur_line.append(w)
            cur_w += w_len
        else:
            lines.append(cur_line)
            cur_line = [w]
            cur_w = w_len
    if cur_line:
        lines.append(cur_line)

    line_h = int(font_size * 1.2)
    pad = 22
    stroke = max(6, int(font_size * 0.11))
    strip_h = line_h * len(lines) + 2 * pad
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    y = max(0, min(H - strip_h, int(H * center_y - strip_h / 2))) + pad
    for line in lines:
        line_text = " ".join(line)
        line_w = font.getlength(line_text)
        x = (W - line_w) / 2
        for word in line:
            word_lower = word.lower().strip(".,!?")
            color = highlight_color if word_lower in POWER_WORDS else base_color
            draw.text(
                (x, y), word, font=font, fill=color,
                stroke_width=stroke, stroke_fill=(0, 0, 0, 255),
            )
            x += font.getlength(word) + 10
        y += line_h
    img.save(png_path)


# Aggressive dead-space removal (jump cuts): cut pauses longer than this.
SILENCE_DB = -30          # quieter than this counts as silence
SILENCE_MIN = 0.25        # only remove pauses longer than this (seconds)
SILENCE_PAD = 0.07        # leave a little air around speech so cuts aren't harsh


def _detect_silences(video_path, start, end):
    """Return [(s, e)] silence intervals within the clip, in clip-relative secs."""
    out = subprocess.run(
        [_tool("ffmpeg"), "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", video_path,
         "-af", f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN}",
         "-f", "null", "-"],
        capture_output=True, text=True, errors="replace",
    ).stderr
    sils, cur = [], None
    for line in out.splitlines():
        if "silence_start:" in line:
            try:
                cur = float(line.split("silence_start:")[1].strip())
            except Exception:
                cur = None
        elif "silence_end:" in line and cur is not None:
            try:
                e = float(line.split("silence_end:")[1].split("|")[0].strip())
                sils.append((max(0.0, cur), e))
            except Exception:
                pass
            cur = None
    return sils


def _keep_segments(duration, silences, pad=SILENCE_PAD):
    """Speech segments (complement of silences), keeping a little air around each."""
    keeps, prev = [], 0.0
    for s, e in silences:
        keep_end = min(duration, s + pad)
        if keep_end > prev:
            keeps.append((prev, keep_end))
        prev = max(keep_end, e - pad)
    if prev < duration:
        keeps.append((prev, duration))
    return [(a, b) for a, b in keeps if b - a > 0.05]


def _remap(t, keeps):
    """Map an original clip-relative time onto the tightened (gaps-removed) timeline."""
    new = 0.0
    for ks, ke in keeps:
        if t < ks:
            return new
        if t <= ke:
            return new + (t - ks)
        new += ke - ks
    return new


def _get_face_center(video_path, start, end):
    """Sample frames to find the average speaker's face center (X-coordinate).
    Returns a float (0.0 to 1.0) representing the horizontal center of the face.
    """
    import cv2  # lazy: only needed when face tracking is on (opencv-python-headless)

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    # Haar Cascade for face detection (built-in to OpenCV)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    centers = []
    # Sample one frame per second to keep it fast
    for t in range(int(start), int(end)):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            # Use the largest face found in the frame
            (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
            center_x = (x + w/2) / width
            centers.append(center_x)

    cap.release()
    return sum(centers) / len(centers) if centers else 0.5

CAM_SIZE = 0.30  # facecam box width as a fraction of the source width
CAM_PRESET_FALLBACK = 0.28  # "medium" — used when auto-detection isn't confident


@functools.lru_cache(maxsize=16)
def detect_facecam(video_path, corner=None, samples=12):
    """Find a streamer's webcam overlay in a recording.

    Returns (x, y, w, h, corner) in source pixels, or None if not confident.
    - Corner: where the streamer's face sits across sampled frames (unless given).
    - Box: an overlay has straight borders that stay put for the whole stream
      while the game behind it changes. The per-pixel MEDIAN edge strength over
      many frames keeps those persistent borders and washes out moving content;
      each box edge is the line that is strong along the other edge's span.
    Cached per video, so all reels of a job share one detection (~10 s).
    Tested: finds a 513x288 top-left cam within ~8 px (inside, never larger),
    the same cam moved bottom-right, and returns None for gameplay with no cam.
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    W0 = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H0 = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    if not (W0 and H0 and dur):
        cap.release()
        return None
    sw = 640
    scale = sw / W0
    sh = int(round(H0 * scale))
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gxs, gys, faces = [], [], []
    for t in np.linspace(dur * 0.05, dur * 0.95, samples):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        gf = g.astype(np.float32)
        gxs.append(np.abs(cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)))
        gys.append(np.abs(cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)))
        for (fx, fy, fw, fh) in cascade.detectMultiScale(g, 1.1, 5, minSize=(20, 20)):
            faces.append((fx + fw / 2, fy + fh / 2))
    cap.release()
    if len(gxs) < 6:
        return None

    if corner in (None, "", "auto"):
        if len(faces) < 3:
            return None
        cx = np.median([f[0] for f in faces])
        cy = np.median([f[1] for f in faces])
        corner = ("top" if cy < sh / 2 else "bottom") + "-" + ("left" if cx < sw / 2 else "right")
    right, bottom = corner.endswith("right"), corner.startswith("bottom")

    gx = np.median(np.stack(gxs), axis=0)
    gy = np.median(np.stack(gys), axis=0)
    if right:  # orient so the cam's corner is top-left
        gx, gy = gx[:, ::-1], gy[:, ::-1]
    if bottom:
        gx, gy = gx[::-1, :], gy[::-1, :]
    thr_x = max(25.0, float(np.percentile(gx, 92)))
    thr_y = max(25.0, float(np.percentile(gy, 92)))

    def best_col(row_end):
        # share of the cam's height with a strong vertical edge at x (±1 px)
        band = gx[:row_end] > thr_x
        cov = np.zeros(sw)
        for x in range(int(sw * 0.10), int(sw * 0.70)):
            cov[x] = band[:, x - 1:x + 2].any(axis=1).mean()
        x = int(cov.argmax())
        return x, float(cov[x])

    def best_row(col_end):
        band = gy[:, :col_end] > thr_y
        cov = np.zeros(sh)
        for y in range(int(sh * 0.08), int(sh * 0.70)):
            cov[y] = band[y - 1:y + 2, :].any(axis=0).mean()
        y = int(cov.argmax())
        return y, float(cov[y])

    y, _ = best_row(int(sw * 0.45))
    for _ in range(2):  # refine: each edge measured along the other's span
        x, cov_x = best_col(max(y, 8))
        y, cov_y = best_row(max(x, 8))
    if min(cov_x, cov_y) < 0.6:
        return None
    w, h = x / scale, y / scale
    if not (1.1 <= w / h <= 2.4):  # webcams are 4:3 .. ultra-wide, not slivers
        return None
    w, h = int(w) // 2 * 2, int(h) // 2 * 2
    return (int(W0 - w) if right else 0, int(H0 - h) if bottom else 0, w, h, corner)

# Facecam layout: webcam overlays are 16:9 boxes, so the cam is cropped 16:9 and
# shown full-width on top (720x405); the game fills the rest below. Cam on top
# is the streamer-clip norm — TikTok/Reels UI covers the bottom of the screen.
CAM_PANEL_H = (W * 9 // 16) // 2 * 2  # even: yuv420p needs even heights (404)


def _fill(w, h, x=None):
    """Filter that fills a w x h box: crop the largest region with the box's
    aspect ratio FIRST (centred, or at `x`), then scale only that region.
    Same framing as scale-to-cover + crop, but the expensive Lanczos scaler
    touches just the kept pixels — cropping a 1920x1080 frame to 9:16 before
    scaling is ~4x less scaling work than enlarging the whole frame first."""
    crop = f"crop=w='min(iw,ih*{w}/{h})':h='min(ih,iw*{h}/{w})'"
    if x is not None:
        crop += f":x='{x}'"
    # setsar=1: the crop is rounded to even pixels, so the scale is a hair
    # off-aspect; keep pixels square instead of tagging an odd aspect ratio.
    return f"{crop},scale={w}:{h}:flags=lanczos,setsar=1"


@functools.lru_cache(maxsize=32)
def _video_fps(path):
    """Average frame rate of the first video stream, or 0 if unknown."""
    try:
        out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=avg_frame_rate",
                    "-of", "default=nw=1:nk=1", path]).strip()
        num, _, den = out.partition("/")
        return float(num) / float(den or 1)
    except (RuntimeError, ValueError, ZeroDivisionError, OSError):
        return 0.0
_CAM_CORNER = {
    # corner -> (x_expr, y_expr) for cropping the facecam box (cw=iw*F, ch=cw*9/16)
    "top-left":     ("0", "0"),
    "top-right":    ("iw-iw*{F}", "0"),
    "bottom-left":  ("0", "ih-iw*{F}*9/16"),
    "bottom-right": ("iw-iw*{F}", "ih-iw*{F}*9/16"),
}


def make_clip(video_path, words, clip, out_dir, index, music_path=None,
              caption_style=DEFAULT_STYLE, music_volume=0.35, broll_path=None,
              split_mode="off", cam_corner="bottom-right", cam_size=CAM_SIZE,
              logo_path=None, logo_scale=0.16, logo_corner="top-right",
              captions=True, jump_cut=True, punchy=False, punch_zoom=False,
              color_pop=False, auto_crop=False, on_progress=None):
    """Cut, remove dead space, crop to 9:16, overlay captions, optional music.

    on_progress(fraction) is called while ffmpeg renders (0..1).


    split_mode:
      "off"      — normal full-frame vertical.
      "facecam"  — single source split into the 16:9 corner facecam (top strip)
                   + the game (rest of the frame below).
      "broll"    — speaker on top, looping b-roll (broll_path) on the bottom.
    Captions sit on the seam in split modes.

    logo_path: optional watermark image overlaid in a corner on every clip,
    scaled to `logo_scale` of the frame width and kept clear of the captions.
    """
    start, end = clip["start"], clip["end"]
    out_name = f"clip_{index:02d}.mp4"
    video_path = os.path.abspath(video_path)
    duration = end - start
    if split_mode == "broll" and not broll_path:
        split_mode = "off"  # no b-roll available
    split = split_mode in ("facecam", "broll")

    # Streamer layout: find the actual webcam box when corner/size are "auto"
    # (the default), so the top strip is exactly the cam — no gameplay or site
    # toolbar around it — and sized to the cam's real shape (16:9, 4:3, ...).
    cam_box = None
    cam_panel_h = CAM_PANEL_H
    if split_mode == "facecam":
        if cam_size == "auto" or cam_corner == "auto":
            try:
                cam_box = detect_facecam(video_path, None if cam_corner == "auto" else cam_corner)
            except Exception as e:  # never fail a reel over detection
                print(f"[facecam] detection error: {e}", flush=True)
            print(f"[facecam] auto-detect -> {cam_box or 'not found, using preset'}", flush=True)
        if cam_box:
            cam_corner = cam_box[4]
            bw, bh = cam_box[2], cam_box[3]
            cam_panel_h = max(300, min(640, int(round(W * bh / bw)) // 2 * 2))
        if cam_corner == "auto":
            cam_corner = "bottom-right"
        if cam_size == "auto":
            cam_size = CAM_PRESET_FALLBACK

    cap_words = 1 if punchy else MAX_CHUNK_WORDS
    cap_font = int(FONT_SIZE * 1.55) if punchy else FONT_SIZE
    caps = _caption_chunks(words, start, end, cap_words) if captions else []

    # Aggressive jump-cut: drop pauses so the clip is fast-paced. Skipped in
    # full-video mode so an already-edited reel keeps its original timing.
    if jump_cut:
        silences = _detect_silences(video_path, start, end)
        keeps = _keep_segments(duration, silences)
        kept = sum(ke - ks for ks, ke in keeps)
        tighten = bool(keeps) and kept < duration - 0.3  # only if it removes real time
    else:
        keeps = []
        tighten = False
    if tighten:
        caps = [(_remap(ds, keeps), _remap(de, keeps), t) for ds, de, t in caps]
        caps = [(a, b, t) for a, b, t in caps if b - a > 0.04]

    if split_mode == "facecam":
        cap_center = (cam_panel_h + 90) / H  # just under the cam, clear of the face
    elif split:
        cap_center = 0.5  # captions on the seam
    else:
        cap_center = CAPTION_CENTER_Y

    inputs = ["-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", video_path]

    # Captions -> ONE transparent track via the concat demuxer (full-frame PNGs
    # with transparent filler for gaps), overlaid ONCE. This scales to any number
    # of captions; one image-input per caption hits ffmpeg's input/decoder limit
    # ("Resource temporarily unavailable") on long or punchy clips.
    cap_idx = None
    timeline_end = kept if tighten else duration
    if caps:
        blank = os.path.abspath(os.path.join(out_dir, f"capblank_{index:02d}.png"))
        from PIL import Image as _Img
        _Img.new("RGBA", (W, H), (0, 0, 0, 0)).save(blank)
        lines = ["ffconcat version 1.0"]
        cursor = 0.0
        for i, (rs, re_, text) in enumerate(caps):
            if rs - cursor > 0.001:
                lines.append(f"file '{blank}'")
                lines.append(f"duration {rs - cursor:.3f}")
            png = os.path.abspath(os.path.join(out_dir, f"cap_{index:02d}_{i}.png"))
            _render_caption_png(text, png, caption_style, cap_font, cap_center)
            lines.append(f"file '{png}'")
            lines.append(f"duration {max(0.04, re_ - rs):.3f}")
            cursor = re_
        if timeline_end - cursor > 0.001:
            lines.append(f"file '{blank}'")
            lines.append(f"duration {timeline_end - cursor:.3f}")
        lines.append(f"file '{blank}'")  # concat drops the last duration; repeat
        concat_file = os.path.join(out_dir, f"caps_{index:02d}.txt")
        with open(concat_file, "w") as f:
            f.write("\n".join(lines) + "\n")
        inputs += ["-f", "concat", "-safe", "0", "-i", concat_file]
        cap_idx = 1  # 0=video, 1=caption track

    nxt_idx = 1 + (1 if cap_idx is not None else 0)
    broll_idx = None
    if split_mode == "broll":
        inputs += ["-stream_loop", "-1", "-i", os.path.abspath(broll_path)]
        broll_idx = nxt_idx
        nxt_idx += 1
    music_idx = None
    if music_path:
        inputs += ["-stream_loop", "-1", "-i", os.path.abspath(music_path)]
        music_idx = nxt_idx
        nxt_idx += 1
    logo_idx = None
    if logo_path and os.path.exists(logo_path):
        inputs += ["-i", os.path.abspath(logo_path)]
        logo_idx = nxt_idx
        nxt_idx += 1

    keep_expr = "+".join(f"between(t,{ks:.3f},{ke:.3f})" for ks, ke in keeps)
    # Source video chain (with jump cuts applied if tightening).
    vchain = "[0:v]"
    # Reels are 30fps. A 60fps source (common for streams) would otherwise run
    # every filter AND the encoder on twice the frames for no visible gain.
    # Done first, so everything after (incl. jump cuts) sees 30fps.
    if _video_fps(video_path) > 30.5:
        vchain += "fps=30,"
    if tighten:
        vchain += f"select='{keep_expr}',setpts=N/FRAME_RATE/TB,"

    half = _fill(W, H // 2)
    if split_mode == "facecam":
        # One source: 16:9 corner cam -> top strip, game -> everything below.
        x_expr, y_expr = _CAM_CORNER.get(cam_corner, _CAM_CORNER["bottom-right"])
        x_expr = x_expr.format(F=cam_size)
        y_expr = y_expr.format(F=cam_size)
        cam_crop = f"crop=iw*{cam_size}:iw*{cam_size}*9/16:{x_expr}:{y_expr}"
        if cam_box:  # exact detected webcam box, in source pixels
            bx, by, bw, bh = cam_box[:4]
            cam_crop = f"crop={bw}:{bh}:{bx}:{by}"
        game_h = H - cam_panel_h
        # Game: the centre of the frame (where the action is). The tall crop
        # is narrow enough that a corner cam stays out of it.
        game_fill = _fill(W, game_h)
        cam_fill = _fill(W, cam_panel_h)
        parts = [
            f"{vchain}split=2[g][c]",
            f"[c]{cam_crop},{cam_fill}[top]",
            f"[g]{game_fill}[bot]",
            "[top][bot]vstack=inputs=2,setsar=1[base]",
        ]
    elif split_mode == "broll":
        # Speaker fills the top half, looping b-roll fills the bottom half.
        parts = [
            f"{vchain}{half}[top]",
            f"[{broll_idx}:v]{half},setpts=PTS-STARTPTS[bot]",
            "[top][bot]vstack=inputs=2:shortest=1[base]",
        ]
    else:
        if auto_crop:
            center_x = _get_face_center(video_path, start, end)
            # 9:16 window centred on the detected face (clamped to the frame).
            crop = _fill(W, H, x=f"max(0,min(iw-ow,iw*{center_x}-ow/2))")
        else:
            crop = _fill(W, H)
        parts = [f"{vchain}{crop}[base]"]

    # Viral polish applied to the footage only (captions/logo stay crisp on top):
    #  - color_pop: punchier saturation/contrast for a scroll-stopping look.
    #  - punch_zoom: a slow continuous zoom so static footage feels alive.
    fx = []
    if color_pop:
        fx.append("eq=contrast=1.06:saturation=1.28:brightness=0.02")
    if punch_zoom:
        # Force a true 30fps FIRST (resamples by timestamp, preserving duration),
        # then zoompan d=1 outputs 1 frame per frame at 30fps -> duration is kept
        # and audio stays in sync. (Without the fps step, a non-30fps source —
        # e.g. 60fps gameplay — drifts because zoompan re-times by frame count.)
        fx.append("fps=30")
        fx.append(
            f"zoompan=z='min(pzoom+0.0006,1.12)':d=1:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps=30"
        )
    base_label = "base"
    if fx:
        parts.append(f"[base]{','.join(fx)}[basefx]")
        base_label = "basefx"

    last = base_label
    if cap_idx is not None:
        # Single overlay of the whole caption track (already positioned per-frame).
        parts.append(f"[{last}][{cap_idx}:v]overlay=0:0[capped]")
        last = "capped"

    # Logo/watermark: scale to a fraction of the width, sit in a padded corner
    # on top of everything (always visible), clear of the centred captions.
    if logo_idx is not None:
        lw = max(2, int(W * float(logo_scale)))
        pad = int(W * 0.045)
        corner = {
            "top-left":     f"{pad}:{pad}",
            "top-right":    f"W-w-{pad}:{pad}",
            "bottom-left":  f"{pad}:H-h-{pad}",
            "bottom-right": f"W-w-{pad}:H-h-{pad}",
        }.get(logo_corner, f"W-w-{pad}:{pad}")
        parts.append(f"[{logo_idx}:v]scale={lw}:-2[lg]")
        parts.append(f"[{last}][lg]overlay={corner}[vlogo]")
        last = "vlogo"

    # Audio: tightened to match the cuts, plus an optional quiet music bed.
    if tighten:
        parts.append(f"[0:a]aselect='{keep_expr}',asetpts=N/SR/TB[sa]")
        speech = "[sa]"
    else:
        speech = "[0:a]"
    if music_idx is not None:
        parts.append(
            f"[{music_idx}:a]volume={music_volume:.2f}[bg];"
            f"{speech}[bg]amix=inputs=2:duration=first:normalize=0[aout]"
        )
        audio_map = "[aout]"
    else:
        audio_map = speech if tighten else "0:a?"

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(parts),
        "-map", f"[{last}]", "-map", audio_map,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-threads", "2",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        out_name,
    ]
    expected = kept if tighten else duration  # output length after jump cuts
    try:
        _run_ffmpeg(cmd, cwd=out_dir, expected=expected, on_progress=on_progress)
    except FFmpegStalled:
        # A hang is usually a one-off (I/O hiccup, starved CPU) — retry once.
        print(f"[render] {out_name} stalled; retrying once", flush=True)
        _run_ffmpeg(cmd, cwd=out_dir, expected=expected, on_progress=on_progress)
    return out_name


# ---------------------------------------------------------------- orchestrate

_AUDIO_EXTS = (".mp3", ".m4a", ".wav", ".aac", ".ogg")


def _tracks_in(folder):
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(_AUDIO_EXTS)
    ]


_VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".m4v")


def pick_broll():
    """Return a gameplay/b-roll video from broll/, or None if none exist."""
    import random
    if not os.path.isdir(BROLL_DIR):
        return None
    vids = [
        os.path.join(BROLL_DIR, f) for f in os.listdir(BROLL_DIR)
        if f.lower().endswith(_VIDEO_EXTS)
    ]
    return random.choice(vids) if vids else None


def music_for_mood(mood):
    """Pick a track matching the clip's mood; fall back to any available track.

    Looks in music/<mood>/ first, then any track anywhere under music/.
    """
    import random
    if not os.path.isdir(MUSIC_DIR):
        return None
    candidates = _tracks_in(os.path.join(MUSIC_DIR, mood or "")) if mood else []
    if not candidates:
        # Fall back to any track anywhere under music/ (incl. the sample).
        for root, _dirs, files in os.walk(MUSIC_DIR):
            for f in files:
                if f.lower().endswith(_AUDIO_EXTS):
                    candidates.append(os.path.join(root, f))
    return random.choice(candidates) if candidates else None


def run_pipeline(url, workdir, api_key, progress, language="en", music=False,
                 caption_style=DEFAULT_STYLE, sarvam_key=None, music_volume=0.35,
                 split_mode="off", cam_corner="bottom-right", cam_size=CAM_SIZE,
                 source_file=None, logo_file=None, logo_scale=0.16,
                 logo_corner="top-right", clip_mode="moments", captions=True,
                 punchy=False, punch_zoom=False, color_pop=False,
                 auto_crop=False,
                 min_clip=None, max_clip=None, on_duration=None):
    """
    Full run. `progress(pct, message)` is called to report status.
    `language` is "en" (fast local English model), or a code like "te"/"hi"/"ta"
    /"auto". For non-English, Sarvam is used when `sarvam_key` is set (accurate
    Indian languages); otherwise it falls back to the local multilingual model.
    `music` mixes a quiet bed from music/ under the speech, if a track exists.

    `clip_mode`:
      "moments" — AI finds the best moments and cuts several reels (default).
      "full"    — keep the whole video as a single reel, no cutting and no
                  dead-space removal (for already-edited reels you just want
                  captioned / branded).
    `captions`: burn animated captions (set False to only crop + add a logo,
    e.g. for a reel that already has captions baked in).
    `on_duration(seconds)` is called once the video's length is known (before
    the slow steps) so the caller can enforce a quota by raising.
    Returns list of {file, title, reason, start, end, length}.
    """
    full_mode = clip_mode == "full"
    progress(2, "Checking video...")
    # Source is either an uploaded file (preferred for the product) or a URL.
    if source_file:
        video = os.path.abspath(source_file)
        dur = _probe_duration(video)
    else:
        dur = get_duration(url)
    if dur and dur > MAX_VIDEO_MINUTES * 60:
        mins = int(dur // 60)
        raise RuntimeError(
            f"This video is {mins} minutes long. For now Clip Reels handles "
            f"videos up to {MAX_VIDEO_MINUTES} minutes — try a shorter video, "
            "or a single segment of this one."
        )
    if dur and on_duration:
        on_duration(dur)

    if not source_file:
        progress(5, "Downloading video...")
        video = download_video(url, workdir)
        # Drive (and some sources) don't report duration up front — probe the
        # downloaded file so the length cap and clip count still work.
        if not dur:
            dur = _probe_duration(video)
            if dur and on_duration:
                on_duration(dur)

    # We need a transcript for caption text, and for AI moment selection.
    # If the user keeps the whole video AND wants no captions, skip it entirely
    # (much faster — handy for just slapping a logo on a finished reel).
    need_transcript = (not full_mode) or captions
    words = []
    audio = None
    if need_transcript:
        progress(30, "Extracting audio...")
        audio = extract_audio(video, workdir)

        prefer_sarvam = os.environ.get("PREFER_SARVAM", "").strip().lower() in ("1", "true", "yes")
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        # Fastest engine first, each failure falls through to the next:
        #  - Groq transcribes a whole video in one call (seconds) — best for English.
        #  - Sarvam is the accurate engine for Telugu / Hindi / Tamil.
        #  - Local whisper.cpp is the slow last resort.
        engines = []
        if language == "en" and groq_key:
            engines.append("groq")
        if sarvam_key and (language != "en" or prefer_sarvam):
            engines.append("sarvam")
        for extra in (("groq" if groq_key else None), ("sarvam" if sarvam_key else None), "local"):
            if extra and extra not in engines:
                engines.append(extra)

        errors = []
        for engine in engines:
            try:
                if engine == "groq":
                    progress(45, "Transcribing with Groq (fast cloud engine)...")
                    words = transcribe_groq(audio, workdir, language)
                elif engine == "sarvam":
                    progress(45, "Transcribing with Sarvam (Indian-language engine)...")
                    words = transcribe_sarvam(audio, workdir, language, sarvam_key, progress)
                else:
                    model_path = MODEL_PATH if language == "en" else MULTILINGUAL_MODEL
                    lang = "en" if language == "en" else language
                    mins = int((dur or 0) / 60)
                    note = "" if lang == "en" else " — non-English local model is slower"
                    est = f"~{mins} min of audio" if mins else "the audio"
                    progress(45, f"Transcribing {est}{note}. This can take a few minutes...")
                    words = transcribe(audio, workdir, model_path, lang)
            except Exception as e:
                print(f"[transcribe] {engine} failed: {e}", flush=True)
                errors.append(f"{engine}: {e}")
                continue
            if words:
                break
            errors.append(f"{engine}: no speech found")
        else:
            msg = "Transcription failed — " + " | ".join(errors)
            if "Device Guard" in msg or "Application Control" in msg or "WinError 4551" in msg:
                msg += ("\n\nWindows Application Control is blocking the local transcription "
                        "binaries. Set GROQ_API_KEY or SARVAM_API_KEY to use cloud transcription.")
            raise RuntimeError(msg)
    phrases = group_phrases(words)

    total = dur or (words[-1]["end"] if words else 0)

    if full_mode:
        # Keep the whole video as one reel — no cutting, no moment selection.
        progress(70, "Editing the whole video...")
        if not total:
            total = _probe_duration(video) or 0
        clips = [{
            "start": 0.0, "end": round(total, 2),
            "title": "Full video", "reason": "", "mood": "neutral",
        }]
    else:
        # Scale the number of clips to the video length (~1 per few minutes).
        max_clips = max(3, min(MAX_CLIPS_CAP, round(total / 60 / CLIP_EVERY_MINUTES)))

        def auto_pick():
            return pick_moments_heuristic(phrases, max_clips, words=words, audio_path=audio,
                                          min_clip=min_clip, max_clip=max_clip)
        if api_key:
            import anthropic
            progress(68, f"Claude is picking the best moments (up to {max_clips})...")
            try:
                clips = pick_moments_ai(phrases, api_key, max_clips, min_clip, max_clip)
            except (anthropic.APIError, ValueError, StopIteration) as e:
                # The AI step being unavailable (no credits, outage, unusable
                # reply) must not fail the whole job: log the real reason for
                # the owner, keep it out of the user's UI, and fall back.
                print(f"[ai] moment picker failed — using automatic selection: {e}", flush=True)
                progress(70, "Using automatic moment selection...")
                clips = auto_pick()
            if not clips:
                progress(70, "Little speech found — using automatic selection...")
                clips = auto_pick()
        else:
            progress(70, "Selecting moments (no AI key — using fallback)...")
            clips = auto_pick()

        if not clips:
            raise RuntimeError(
                "This video has almost no spoken content. Auto-clipping finds "
                "the best *spoken* moments, so it's built for talking videos — "
                "podcasts, talks, interviews, vlogs. (Tip: choose “Keep the "
                "whole video” to just caption/brand this one instead.)"
            )

    broll_path = pick_broll() if split_mode == "broll" else None

    results = []
    for i, clip in enumerate(clips, 1):
        n_clips = len(clips)
        progress(70 + int(28 * (i - 1) / n_clips),
                 f"Rendering reel {i} of {n_clips}: {clip['title']}")
        shown = [0]

        def on_render(frac, i=i, n=n_clips, title=clip["title"], shown=shown):
            # Live % inside a reel, so a slow render never looks frozen.
            # Throttled: every 5% is plenty and keeps database writes low.
            pct = int(frac * 100)
            if pct >= shown[0] + 5:
                shown[0] = pct
                progress(70 + int(28 * (i - 1 + frac) / n),
                         f"Rendering reel {i} of {n} — {pct}%: {title}")
        # Per-clip, mood-matched music.
        music_path = music_for_mood(clip.get("mood")) if music else None
        fname = make_clip(
            video, words, clip, workdir, i, music_path, caption_style,
            music_volume, broll_path, split_mode, cam_corner, cam_size,
            logo_path=logo_file, logo_scale=logo_scale, logo_corner=logo_corner,
            captions=captions, jump_cut=(not full_mode),
            punchy=punchy, punch_zoom=punch_zoom, color_pop=color_pop,
            auto_crop=auto_crop, on_progress=on_render,
        )
        # Actual length after dead-space removal (falls back to the cut range).
        actual = _probe_duration(os.path.join(workdir, fname))
        length = round(actual if actual else clip["end"] - clip["start"], 1)
        results.append({
            "file": fname,
            "title": clip["title"],
            "reason": clip["reason"],
            "mood": clip.get("mood", "neutral"),
            "start": clip["start"],
            "end": clip["end"],
            "length": length,
        })

    progress(100, "Done!")
    return results
