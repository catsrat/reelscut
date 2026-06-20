"""
Clip Reels — web app.

Run:  ./venv/bin/python app.py
Then open http://localhost:5050

Set ANTHROPIC_API_KEY (AI moment-selection) and SARVAM_API_KEY (Indian-language
transcription) in the environment. Without ANTHROPIC_API_KEY it still works,
using a simple fallback selector.
"""

import os
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
    }


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
        job["title"] = opts.get("title") or (
            pipeline.get_title(url) if url else "Your video"
        )
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        job["ai"] = bool(api_key)
        sarvam_key = os.environ.get("SARVAM_API_KEY", "").strip()
        results = pipeline.run_pipeline(
            url, workdir, api_key, progress,
            language=opts["language"], music=opts["music"],
            caption_style=opts["caption_style"], sarvam_key=sarvam_key,
            music_volume=opts["music_volume"], split_mode=opts["split_mode"],
            cam_corner=opts["cam_corner"], cam_size=opts["cam_size"],
            source_file=source_file,
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


def _start_job(opts):
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "status": "running", "progress": 0, "message": "Starting...",
        "title": "", "clips": [], "ai": False,
    }
    threading.Thread(target=_worker, args=(job_id, opts), daemon=True).start()
    return job_id


@app.route("/")
def index():
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    # Upload-first: only show the YouTube-link tab where it can actually work —
    # i.e. yt-dlp is present AND cookies are configured to get past YouTube's
    # bot check. Otherwise show a clean upload-only interface.
    cookies_file = os.environ.get("YTDLP_COOKIES_FILE", "").strip()
    has_cookies = bool(
        (cookies_file and os.path.exists(cookies_file))
        or os.environ.get("YT_COOKIES_BROWSER", "").strip()
    )
    show_link = bool(shutil.which("yt-dlp")) and has_cookies
    return render_template("index.html", has_key=has_key, show_link=show_link)


@app.route("/process", methods=["POST"])
def process():
    """Process a YouTube link (JSON body)."""
    data = request.json or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Please paste a YouTube link."}), 400
    if _busy():
        return jsonify({"error": "A video is already being processed. "
                                 "Please wait for it to finish."}), 409
    opts = _collect_opts(data.get)
    opts["url"] = url
    opts["source_file"] = None
    return jsonify({"job_id": _start_job(opts)})


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

    # Reuse the job_id/workdir we already created for the saved file.
    JOBS[job_id] = {
        "status": "running", "progress": 0, "message": "Starting...",
        "title": opts["title"], "clips": [], "ai": False,
    }
    threading.Thread(target=_worker, args=(job_id, opts), daemon=True).start()
    return jsonify({"job_id": job_id})


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
