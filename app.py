"""
Clip Reels — web app.

Run:  ./venv/bin/python app.py
Then open http://localhost:5050

Set ANTHROPIC_API_KEY (AI moment-selection) and SARVAM_API_KEY (Indian-language
transcription) in the environment. Without ANTHROPIC_API_KEY it still works,
using a simple fallback selector.
"""

import os
import re
import shutil
import threading
import uuid

from flask import (
    Flask, render_template, request, jsonify, send_from_directory, abort
)
from werkzeug.utils import secure_filename

import pipeline
import storage

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4 GB uploads

ROOT = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(ROOT, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

# In-memory job registry. Fine for a single-user local MVP.
JOBS = {}


def _collect_opts(get):
    """Read the shared editing options from a dict-like getter (JSON or form)."""
    return {
        "language": (get("language") or "en").strip(),
        "caption_style": (get("caption_style") or "bold_white").strip(),
        "music": str(get("music")).lower() in ("true", "1", "on", "yes"),
        "music_volume": {"soft": 0.2, "medium": 0.4, "loud": 0.65}.get(
            (get("music_volume") or "medium").strip(), 0.4),
        "split_mode": (get("split_mode") or "off").strip(),
        "cam_corner": (get("cam_corner") or "bottom-right").strip(),
        "cam_size": {"small": 0.18, "medium": 0.28, "large": 0.4}.get(
            (get("cam_size") or "medium").strip(), 0.28),
        "logo_scale": {"small": 0.12, "medium": 0.20, "large": 0.32,
                       "xlarge": 0.45}.get(
            (get("logo_size") or "medium").strip(), 0.20),
        "logo_corner": (get("logo_corner") or "top-right").strip(),
        "logo_file": None,  # filled in from the uploaded logo, if any
        # "moments" = AI auto-clips best moments; "full" = keep the whole video.
        "clip_mode": (get("clip_mode") or "moments").strip(),
        # captions default ON; the form sends "true"/"false".
        "captions": (str(get("captions")).lower() in ("true", "1", "on", "yes"))
        if get("captions") is not None else True,
        # Viral effects (all default OFF unless toggled on).
        "punchy": str(get("punchy")).lower() in ("true", "1", "on", "yes"),
        "punch_zoom": str(get("punch_zoom")).lower() in ("true", "1", "on", "yes"),
        "color_pop": str(get("color_pop")).lower() in ("true", "1", "on", "yes"),
        # Optional pasted clipping-campaign rules (Whop / Google Doc).
        "rules": (get("rules") or "").strip(),
    }


def _apply_rules(req, opts):
    """Auto-apply parsed campaign requirements onto the editing options, and
    return the 'payment-safe' report (checklist + ready-to-paste caption)."""
    if req.get("captions_required"):
        opts["captions"] = True
    lang = (req.get("language") or "").strip().lower()
    if lang in ("en", "te", "hi", "ta"):
        opts["language"] = lang
    mn = req.get("min_length_sec") or 0
    mx = req.get("max_length_sec") or 0
    if mn:
        opts["min_clip"] = float(mn)
    if mx:
        opts["max_clip"] = float(mx)

    auto = list(req.get("auto_handled") or [])
    manual = list(req.get("manual_todo") or [])
    # If the campaign needs a watermark but none was uploaded, it's on the user.
    if req.get("logo_required") and not opts.get("logo_file"):
        manual.insert(0, "Upload your logo/watermark — this campaign requires it "
                         "(add it in “Your logo / watermark”).")
    return {
        "post_caption": req.get("post_caption") or "",
        "hashtags": req.get("hashtags") or [],
        "mentions": req.get("mentions") or [],
        "auto_handled": auto,
        "manual_todo": manual,
    }


def _attach_logo(workdir, opts):
    """Save an optional uploaded logo/watermark into the job dir."""
    logo = request.files.get("logo")
    if logo and logo.filename:
        name = secure_filename(logo.filename) or "logo.png"
        ext = os.path.splitext(name)[1].lower() or ".png"
        path = os.path.join(workdir, "logo" + ext)
        logo.save(path)
        opts["logo_file"] = path


def _worker(job_id, opts):
    job = JOBS[job_id]
    workdir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(workdir, exist_ok=True)

    def progress(pct, message):
        job["progress"] = pct
        job["message"] = message

    try:
        source_file = opts.get("source_file")
        url = opts.get("url")
        # Skip the slow yt-dlp title probe (cosmetic) so the download starts
        # right away — each clip gets its own AI title anyway.
        job["title"] = opts.get("title") or "Your video"
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        job["ai"] = bool(api_key)
        sarvam_key = os.environ.get("SARVAM_API_KEY", "").strip()

        # Read clipping-campaign rules first and auto-apply them, so the clip is
        # built compliant and we can hand back a payment-safe checklist.
        rules_text = opts.get("rules") or ""
        if rules_text and api_key:
            progress(3, "Reading the campaign rules...")
            try:
                req = pipeline.parse_campaign_rules(rules_text, api_key)
                job["compliance"] = _apply_rules(req, opts)
            except Exception:
                pass  # rules parsing is best-effort; never block the clip

        results = pipeline.run_pipeline(
            url, workdir, api_key, progress,
            language=opts["language"], music=opts["music"],
            caption_style=opts["caption_style"], sarvam_key=sarvam_key,
            music_volume=opts["music_volume"], split_mode=opts["split_mode"],
            cam_corner=opts["cam_corner"], cam_size=opts["cam_size"],
            source_file=source_file,
            logo_file=opts.get("logo_file"),
            logo_scale=opts.get("logo_scale", 0.16),
            logo_corner=opts.get("logo_corner", "top-right"),
            clip_mode=opts.get("clip_mode", "moments"),
            captions=opts.get("captions", True),
            punchy=opts.get("punchy", False),
            punch_zoom=opts.get("punch_zoom", False),
            color_pop=opts.get("color_pop", False),
            min_clip=opts.get("min_clip"),
            max_clip=opts.get("max_clip"),
        )
        # If R2 is configured, upload clips and attach durable URLs.
        if storage.enabled():
            for r in results:
                url = storage.upload_and_url(
                    os.path.join(workdir, r["file"]), f"{job_id}/{r['file']}"
                )
                if url:
                    r["url"] = url
        job["clips"] = results
        job["status"] = "done"
    except Exception as e:  # surface the real error to the UI
        job["status"] = "error"
        job["message"] = str(e)


def _busy():
    return any(j["status"] == "running" for j in JOBS.values())


@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/app")
def index():
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    # Show the link tab whenever yt-dlp is available: Google Drive links work
    # with no setup at all (Drive doesn't block servers). YouTube links also
    # work here, but only reliably when a proxy/cookies are configured —
    # `yt_enabled` tells the UI whether to advertise YouTube as ready.
    cookies_file = os.environ.get("YTDLP_COOKIES_FILE", "").strip()
    yt_enabled = bool(
        (cookies_file and os.path.exists(cookies_file))
        or os.environ.get("YT_COOKIES_BROWSER", "").strip()
        or os.environ.get("YTDLP_PROXY", "").strip()  # proxy makes links work in cloud
    )
    show_link = bool(shutil.which("yt-dlp"))
    return render_template(
        "index.html", has_key=has_key, show_link=show_link, yt_enabled=yt_enabled
    )


@app.route("/process", methods=["POST"])
def process():
    """Process a Google Drive / video link (multipart form, optional logo)."""
    url = (request.form.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Please paste a Google Drive link."}), 400
    if _busy():
        return jsonify({"error": "A video is already being processed. "
                                 "Please wait for it to finish."}), 409

    job_id = uuid.uuid4().hex[:12]
    workdir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(workdir, exist_ok=True)

    opts = _collect_opts(request.form.get)
    opts["url"] = url
    opts["source_file"] = None
    _attach_logo(workdir, opts)

    JOBS[job_id] = {
        "status": "running", "progress": 0, "message": "Starting...",
        "title": "", "clips": [], "ai": False,
    }
    threading.Thread(target=_worker, args=(job_id, opts), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/upload", methods=["POST"])
def upload():
    """Process an uploaded video file (multipart form)."""
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "Please choose a video file to upload."}), 400
    if _busy():
        return jsonify({"error": "A video is already being processed. "
                                 "Please wait for it to finish."}), 409

    job_id = uuid.uuid4().hex[:12]
    workdir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(workdir, exist_ok=True)
    name = secure_filename(f.filename) or "upload.mp4"
    ext = os.path.splitext(name)[1].lower() or ".mp4"
    saved = os.path.join(workdir, "source" + ext)
    f.save(saved)

    opts = _collect_opts(request.form.get)
    opts["url"] = None
    opts["source_file"] = saved
    opts["title"] = os.path.splitext(name)[0]
    _attach_logo(workdir, opts)

    # Reuse the job_id/workdir we already created for the saved file.
    JOBS[job_id] = {
        "status": "running", "progress": 0, "message": "Starting...",
        "title": opts["title"], "clips": [], "ai": False,
    }
    threading.Thread(target=_worker, args=(job_id, opts), daemon=True).start()
    return jsonify({"job_id": job_id})


# ---- Chunked upload: send the file in small pieces so big files get past the
# host's single-request size limit (a 210MB upload stalls as one request). ----

@app.route("/upload_start", methods=["POST"])
def upload_start():
    uid = uuid.uuid4().hex[:12]
    os.makedirs(os.path.join(JOBS_DIR, uid), exist_ok=True)
    return jsonify({"upload_id": uid})


def _safe_uid(uid):
    return uid if re.fullmatch(r"[a-f0-9]{12}", uid or "") else None


@app.route("/upload_chunk", methods=["POST"])
def upload_chunk():
    uid = _safe_uid(request.form.get("upload_id"))
    if not uid:
        return jsonify({"error": "bad upload id"}), 400
    workdir = os.path.join(JOBS_DIR, uid)
    if not os.path.isdir(workdir):
        return jsonify({"error": "unknown upload"}), 404
    chunk = request.files.get("chunk")
    if not chunk:
        return jsonify({"error": "no chunk"}), 400
    # Chunks arrive in order (client awaits each) -> append.
    with open(os.path.join(workdir, "upload.part"), "ab") as f:
        f.write(chunk.read())
    return jsonify({"ok": True})


@app.route("/upload_done", methods=["POST"])
def upload_done():
    uid = _safe_uid(request.form.get("upload_id"))
    if not uid:
        return jsonify({"error": "bad upload id"}), 400
    workdir = os.path.join(JOBS_DIR, uid)
    part = os.path.join(workdir, "upload.part")
    if not os.path.exists(part):
        return jsonify({"error": "Upload not found — please try again."}), 400
    if _busy():
        return jsonify({"error": "A video is already being processed. "
                                 "Please wait for it to finish."}), 409

    name = secure_filename(request.form.get("filename") or "") or "upload.mp4"
    ext = os.path.splitext(name)[1].lower() or ".mp4"
    saved = os.path.join(workdir, "source" + ext)
    os.replace(part, saved)

    opts = _collect_opts(request.form.get)
    opts["url"] = None
    opts["source_file"] = saved
    opts["title"] = os.path.splitext(name)[0]
    _attach_logo(workdir, opts)

    JOBS[uid] = {
        "status": "running", "progress": 0, "message": "Starting...",
        "title": opts["title"], "clips": [], "ai": False,
    }
    threading.Thread(target=_worker, args=(uid, opts), daemon=True).start()
    return jsonify({"job_id": uid})


@app.route("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)


@app.route("/clip/<job_id>/<path:filename>")
def clip(job_id, filename):
    if job_id not in JOBS:
        abort(404)
    return send_from_directory(os.path.join(JOBS_DIR, job_id), filename)


if __name__ == "__main__":
    # Cloud hosts (Render/Railway/Fly) inject $PORT; locally default to 5050
    # (5000 is hijacked by macOS AirPlay Receiver).
    port = int(os.environ.get("PORT", "5050"))
    print(f"\n  Clip Reels running at  http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
