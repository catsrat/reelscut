"""
Clip Reels — web app.

Run:  ./venv/bin/python app.py      (also starts a job worker in-process)
Then open http://localhost:5051

In production run the web app and the worker as separate processes:
    gunicorn -w 1 --threads 8 app:app
    python worker.py

Set ANTHROPIC_API_KEY (AI moment-selection) and SARVAM_API_KEY (Indian-language
transcription) in the environment. Without ANTHROPIC_API_KEY it still works,
using a simple fallback selector. Accounts and plans: see auth.py / billing.py
(without WHOP_CLIENT_ID it runs in local mode — no login, no limits).
"""

import os
import re
import secrets
import shutil
import threading
import time
import uuid
from datetime import timedelta

from flask import (
    Flask, render_template, request, jsonify, send_from_directory, abort
)
from werkzeug.utils import secure_filename

ROOT = os.path.dirname(os.path.abspath(__file__))

# Load API keys from .env. Real environment variables win over the file, so a
# host's configured secrets are never overridden by a stray local .env.
_env_file = os.path.join(ROOT, ".env")
if os.path.exists(_env_file):
    with open(_env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().removeprefix("export ").strip()
            os.environ.setdefault(k, v.strip().strip("'\""))

import auth
import billing
import db
import pipeline

GB = 1024 ** 3
MB = 1024 ** 2

# ---- Limits (override via environment) ----
MAX_UPLOAD_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "2048")) * MB)
JOB_TTL_HOURS = float(os.environ.get("JOB_TTL_HOURS", "24"))  # 0 = keep forever
UPLOAD_TTL_SECS = 2 * 3600       # unfinished chunked uploads are dropped after this
MAX_PENDING_UPLOADS = int(os.environ.get("MAX_PENDING_UPLOADS", "3"))
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "20"))  # queued jobs, all users
MIN_FREE_BYTES = int(float(os.environ.get("MIN_FREE_GB", "5")) * GB)
MAX_LOGO_BYTES = 10 * MB
LOGO_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
CLEANUP_EVERY_SECS = 30 * 60
CLIP_TTL_HOURS = float(os.environ.get("CLIP_TTL_HOURS", "72"))  # saved reels; 0 = keep
CLIPS_MIN_FREE = 400 * MB  # prune oldest saved reels below this much free disk

app = Flask(__name__)
# Per-request cap: covers the one-shot /upload route (+ logo). Chunked uploads
# send 5MB pieces and are capped by total size in upload_chunk.
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + 50 * MB

db.init()


def _secret_key():
    """SECRET_KEY from the environment, else one generated once and kept in
    DATA_DIR so sign-ins survive restarts."""
    key = os.environ.get("SECRET_KEY", "").strip()
    if key:
        return key
    path = os.path.join(db.DATA_DIR, "secret_key")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
    with open(path) as f:
        return f.read().strip()


app.secret_key = _secret_key()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",   # also blocks cross-site POSTs (CSRF)
    SESSION_COOKIE_SECURE=os.environ.get("PUBLIC_URL", "").startswith("https://"),
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
)
app.register_blueprint(auth.bp)

JOBS_DIR = os.path.join(ROOT, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

UPLOADS_LOCK = threading.Lock()
PENDING_UPLOADS = {}  # upload_id -> {"user": user id, "touched": ts}


def _collect_opts(get):
    """Read the shared editing options from a dict-like getter (JSON or form)."""
    return {
        "language": (get("language") or "en").strip(),
        "caption_style": (get("caption_style") or "bold_white").strip(),
        "music": str(get("music")).lower() in ("true", "1", "on", "yes"),
        "music_volume": {"soft": 0.2, "medium": 0.4, "loud": 0.65}.get(
            (get("music_volume") or "medium").strip(), 0.4),
        "split_mode": (get("split_mode") or "off").strip(),
        # "auto" = detect the webcam box in the video (pipeline.detect_facecam)
        "cam_corner": (get("cam_corner") or "auto").strip(),
        "cam_size": {"small": 0.18, "medium": 0.28, "large": 0.4, "auto": "auto"}.get(
            (get("cam_size") or "auto").strip(), "auto"),
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
        "auto_crop": str(get("auto_crop")).lower() in ("true", "1", "on", "yes"),
        # Optional pasted clipping-campaign rules (Whop / Google Doc).
        "rules": (get("rules") or "").strip(),
    }


def _attach_logo(workdir, opts):
    """Save an optional uploaded logo/watermark into the job dir."""
    logo = request.files.get("logo")
    if logo and logo.filename:
        name = secure_filename(logo.filename) or "logo.png"
        ext = os.path.splitext(name)[1].lower() or ".png"
        if ext not in LOGO_EXTS:
            return  # not an image we can overlay; skip rather than fail the job
        logo.stream.seek(0, os.SEEK_END)
        too_big = logo.stream.tell() > MAX_LOGO_BYTES
        logo.stream.seek(0)
        if too_big:
            return
        path = os.path.join(workdir, "logo" + ext)
        logo.save(path)
        opts["logo_file"] = path


DISK_MSG = "The server is low on disk space right now — please try again in a bit."


def _size_label(n):
    """2147483648 -> '2 GB', 524288000 -> '500 MB' (non-breaking space)."""
    return f"{n / GB:g} GB" if n >= GB else f"{n // MB} MB"


def _too_big_msg():
    return (f"That file is too large (max {MAX_UPLOAD_BYTES // MB} MB). "
            "Trim it first, or paste a Google Drive link instead.")


def _out_of_minutes(q):
    msg = (f"You've used all {q['minutes']:.0f} minutes on your {q['plan']} plan "
           "this month.")
    msg += " Upgrade to keep clipping." if q["upgrade_url"] else ""
    return {"error": msg, "upgrade_url": q["upgrade_url"]}


def _disk_ok(extra=0):
    """True if there's room for `extra` more bytes and still MIN_FREE_GB spare."""
    try:
        return shutil.disk_usage(JOBS_DIR).free - extra >= MIN_FREE_BYTES
    except OSError:
        return True


def _refuse(user, extra_bytes=0):
    """Why this user can't submit a video right now, as a (json, status) reply,
    or None if they can. Checked before any file is saved."""
    if db.active_job_for_user(user["id"]):
        return {"error": "You already have a video processing — wait for it "
                         "to finish, then send the next one."}, 409
    if db.queue_length() >= MAX_QUEUE:
        return {"error": "We're very busy right now — please try again in a "
                         "few minutes."}, 503
    q = billing.quota(user)
    if not q["unlimited"] and q["left"] <= 0:
        return _out_of_minutes(q), 402
    if not _disk_ok(extra_bytes):
        return {"error": DISK_MSG}, 507
    return None


def _enqueue(user, job_id, opts, title=""):
    db.create_job(job_id, user["id"], opts, title or "Your video")
    return jsonify({"job_id": job_id})


# ---- Cleanup: delete finished jobs after JOB_TTL_HOURS and abandoned chunked
# uploads after UPLOAD_TTL_SECS, so jobs/ can't grow until the disk is full. ----

def _last_touched(path):
    """Newest mtime of a job dir or the files directly in it (appending to
    upload.part updates the file, not the directory)."""
    times = [os.path.getmtime(path)]
    for entry in os.scandir(path):
        try:
            times.append(entry.stat().st_mtime)
        except OSError:
            pass
    return max(times)


def cleanup_jobs():
    now = time.time()
    active = db.active_job_ids()
    with UPLOADS_LOCK:
        pending = set(PENDING_UPLOADS)
    removed = 0
    for name in os.listdir(JOBS_DIR):
        path = os.path.join(JOBS_DIR, name)
        if name in active or not os.path.isdir(path):
            continue
        try:
            age = now - _last_touched(path)
        except OSError:
            continue
        if name in pending or os.path.exists(os.path.join(path, "upload.part")):
            expired = age > UPLOAD_TTL_SECS
        else:
            expired = JOB_TTL_HOURS > 0 and age > JOB_TTL_HOURS * 3600
        if expired:
            shutil.rmtree(path, ignore_errors=True)
            with UPLOADS_LOCK:
                PENDING_UPLOADS.pop(name, None)
            removed += 1
    if removed:
        print(f"[cleanup] removed {removed} old job folder(s)", flush=True)
    cleanup_saved_clips(now)


def cleanup_saved_clips(now=None):
    """Saved reels on the persistent disk: delete after CLIP_TTL_HOURS, and
    oldest-first whenever free space drops below CLIPS_MIN_FREE — the disk also
    holds the database, which must never run out of room."""
    now = now or time.time()
    if not os.path.isdir(db.CLIPS_DIR):
        return
    dirs = []
    for name in os.listdir(db.CLIPS_DIR):
        path = os.path.join(db.CLIPS_DIR, name)
        if os.path.isdir(path):
            try:
                dirs.append((_last_touched(path), path))
            except OSError:
                pass
    dirs.sort()  # oldest first
    removed = 0
    for touched, path in dirs:
        too_old = CLIP_TTL_HOURS > 0 and now - touched > CLIP_TTL_HOURS * 3600
        low = shutil.disk_usage(db.CLIPS_DIR).free < CLIPS_MIN_FREE
        if too_old or low:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    if removed:
        print(f"[cleanup] removed {removed} saved reel folder(s)", flush=True)


def _janitor():
    while True:
        try:
            cleanup_jobs()
        except Exception as e:  # never let cleanup take the app down
            print(f"[cleanup] failed: {e}", flush=True)
        time.sleep(CLEANUP_EVERY_SECS)


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": _too_big_msg()}), 413


@app.route("/")
def landing():
    # Pricing section comes straight from the same settings billing uses, so
    # the page can never advertise different minutes than users actually get.
    plans = [{"name": p.name, "minutes": p.minutes, "price": p.price, "url": p.url}
             for p in sorted(billing.paid_plans(), key=lambda p: p.minutes)]
    return render_template(
        "landing.html", free_minutes=billing.FREE.minutes, plans=plans,
        store_url=os.environ.get("WHOP_STORE_URL", "").strip(),
    )


@app.route("/app")
@auth.login_required
def index(user):
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
    show_link = pipeline.ytdlp_available()
    return render_template(
        "index.html", has_key=has_key, show_link=show_link, yt_enabled=yt_enabled,
        user=user, quota=billing.quota_json(billing.quota(user)),
        accounts=auth.enabled(), max_upload=_size_label(MAX_UPLOAD_BYTES),
    )


@app.route("/me")
@auth.login_required
def me(user):
    """Current plan + minutes, so the UI can refresh the meter after a job."""
    return jsonify(billing.quota_json(billing.quota(user)))


@app.route("/process", methods=["POST"])
@auth.login_required
def process(user):
    """Process a Google Drive / video link (multipart form, optional logo)."""
    url = (request.form.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Please paste a Google Drive link."}), 400
    # Only real web links — anything else (e.g. "--exec=...") must never
    # reach yt-dlp, where it could be read as a command-line option.
    if not re.fullmatch(r"https?://\S+", url, re.I):
        return jsonify({"error": "Please paste a full link starting with https://"}), 400
    refusal = _refuse(user)
    if refusal:
        return jsonify(refusal[0]), refusal[1]

    job_id = uuid.uuid4().hex[:12]
    workdir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(workdir, exist_ok=True)

    opts = _collect_opts(request.form.get)
    opts["url"] = url
    opts["source_file"] = None
    _attach_logo(workdir, opts)
    return _enqueue(user, job_id, opts)


@app.route("/upload", methods=["POST"])
@auth.login_required
def upload(user):
    """Process an uploaded video file (multipart form). Size is capped by
    MAX_CONTENT_LENGTH (the 413 handler returns a friendly message)."""
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "Please choose a video file to upload."}), 400
    refusal = _refuse(user)
    if refusal:
        return jsonify(refusal[0]), refusal[1]

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
    return _enqueue(user, job_id, opts, opts["title"])


# ---- Chunked upload: send the file in small pieces so big files get past the
# host's single-request size limit (a 210MB upload stalls as one request). ----

@app.route("/upload_start", methods=["POST"])
@auth.login_required
def upload_start(user):
    try:
        size = max(0, int(request.form.get("size") or 0))
    except ValueError:
        size = 0
    # Refuse up front, before the user spends minutes uploading.
    if size > MAX_UPLOAD_BYTES:
        return jsonify({"error": _too_big_msg()}), 413
    refusal = _refuse(user, extra_bytes=size)
    if refusal:
        return jsonify(refusal[0]), refusal[1]

    now = time.time()
    with UPLOADS_LOCK:
        # Forget uploads that went quiet; the janitor deletes their folders.
        for k, v in list(PENDING_UPLOADS.items()):
            if now - v["touched"] > UPLOAD_TTL_SECS:
                PENDING_UPLOADS.pop(k)
        if len(PENDING_UPLOADS) >= MAX_PENDING_UPLOADS:
            return jsonify({"error": "Too many uploads in progress — please "
                                     "try again in a few minutes."}), 429
        uid = uuid.uuid4().hex[:12]
        # size: what the browser says it will send; chunks: offset -> bytes
        # received, so upload_done can tell a complete file from a gappy one.
        PENDING_UPLOADS[uid] = {"user": user["id"], "touched": now, "size": size, "chunks": {}}
    os.makedirs(os.path.join(JOBS_DIR, uid), exist_ok=True)
    # Create the file up front: parallel pieces then all open it "r+b" and
    # write at their own offset (two "wb" opens would truncate each other).
    open(os.path.join(JOBS_DIR, uid, "upload.part"), "wb").close()
    return jsonify({"upload_id": uid})


def _safe_uid(uid):
    return uid if re.fullmatch(r"[a-f0-9]{12}", uid or "") else None


def _drop_upload(uid):
    with UPLOADS_LOCK:
        PENDING_UPLOADS.pop(uid, None)
    shutil.rmtree(os.path.join(JOBS_DIR, uid), ignore_errors=True)


def _pending_for(uid, user):
    """The pending upload, only if it belongs to this user."""
    with UPLOADS_LOCK:
        pending = PENDING_UPLOADS.get(uid)
    return pending if pending and pending["user"] == user["id"] else None


@app.route("/upload_chunk", methods=["POST"])
@auth.login_required
def upload_chunk(user):
    uid = _safe_uid(request.form.get("upload_id"))
    if not uid:
        return jsonify({"error": "bad upload id"}), 400
    pending = _pending_for(uid, user)
    workdir = os.path.join(JOBS_DIR, uid)
    if not pending or not os.path.isdir(workdir):
        return jsonify({"error": "This upload expired — please try again."}), 404
    chunk = request.files.get("chunk")
    if not chunk:
        return jsonify({"error": "no chunk"}), 400
    data = chunk.read()
    try:
        offset = int(request.form.get("offset", ""))
    except ValueError:
        return jsonify({"error": "bad offset"}), 400
    if offset < 0:
        return jsonify({"error": "bad offset"}), 400
    if offset + len(data) > MAX_UPLOAD_BYTES:
        _drop_upload(uid)
        return jsonify({"error": _too_big_msg()}), 413
    size = pending.get("size") or 0
    if size and offset + len(data) > size:
        return jsonify({"error": "Upload got out of sync — please try again."}), 409
    # The browser sends several pieces at once, in any order: write each at
    # its own offset. Idempotent, so a retried piece just overwrites itself.
    part = os.path.join(workdir, "upload.part")
    with open(part, "r+b" if os.path.exists(part) else "w+b") as f:
        f.seek(offset)
        f.write(data)
    with UPLOADS_LOCK:
        pending.setdefault("chunks", {})[offset] = len(data)
        pending["touched"] = time.time()
    return jsonify({"ok": True})


@app.route("/upload_done", methods=["POST"])
@auth.login_required
def upload_done(user):
    uid = _safe_uid(request.form.get("upload_id"))
    if not uid:
        return jsonify({"error": "bad upload id"}), 400
    workdir = os.path.join(JOBS_DIR, uid)
    part = os.path.join(workdir, "upload.part")
    pending = _pending_for(uid, user)
    if not pending or not os.path.exists(part):
        return jsonify({"error": "Upload not found — please try again."}), 400
    size = pending.get("size") or 0
    with UPLOADS_LOCK:
        got = sum(pending.get("chunks", {}).values())
    if size and (got < size or os.path.getsize(part) < size):
        return jsonify({"error": "The upload didn't finish — some pieces are "
                                 "missing. Please try again."}), 400
    refusal = _refuse(user)
    if refusal:
        _drop_upload(uid)
        return jsonify(refusal[0]), refusal[1]

    name = secure_filename(request.form.get("filename") or "") or "upload.mp4"
    ext = os.path.splitext(name)[1].lower() or ".mp4"
    saved = os.path.join(workdir, "source" + ext)
    os.replace(part, saved)
    with UPLOADS_LOCK:
        PENDING_UPLOADS.pop(uid, None)

    # We know the length now: say "not enough minutes" before queueing, not
    # after the wait. (The worker re-checks with a fresh plan lookup.)
    q = billing.quota(user)
    dur = pipeline._probe_duration(saved)
    if dur and not q["unlimited"] and dur / 60 > q["left"] + 0.01:
        shutil.rmtree(workdir, ignore_errors=True)
        return jsonify({
            "error": f"This video is {dur / 60:.0f} min, but your {q['plan']} plan "
                     f"has {q['left']:.0f} min left this month. Trim it or upgrade.",
            "upgrade_url": q["upgrade_url"],
        }), 402

    opts = _collect_opts(request.form.get)
    opts["url"] = None
    opts["source_file"] = saved
    opts["title"] = os.path.splitext(name)[0]
    _attach_logo(workdir, opts)
    return _enqueue(user, uid, opts, opts["title"])


def _own_job(job_id, user):
    job = db.get_job(job_id) if _safe_uid(job_id) else None
    return job if job and job["user_id"] == user["id"] else None


@app.route("/status/<job_id>")
@auth.login_required
def status(user, job_id):
    job = _own_job(job_id, user)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(db.job_public(job))


@app.route("/clip/<job_id>/<path:filename>")
@auth.login_required
def clip(user, job_id, filename):
    if not _own_job(job_id, user):
        abort(404)
    # Saved reels on the persistent disk first; the temporary job folder only
    # if the disk was too full to move them there.
    for base in (os.path.join(db.CLIPS_DIR, job_id), os.path.join(JOBS_DIR, job_id)):
        if os.path.isfile(os.path.join(base, os.path.basename(filename))):
            # ?dl=1: send as a download (the Download button) instead of inline.
            return send_from_directory(base, os.path.basename(filename),
                                       as_attachment=request.args.get("dl") == "1")
    abort(404)


# Runs under gunicorn too (single worker), not just `python app.py`.
threading.Thread(target=_janitor, daemon=True).start()


if __name__ == "__main__":
    # Local convenience: run the job worker inside this process too.
    # (In production run `python worker.py` separately; RUN_WORKER=0 here.)
    if os.environ.get("RUN_WORKER", "1") != "0":
        import worker
        threading.Thread(target=worker.work_forever, daemon=True).start()
    if not auth.enabled():
        print("  Local mode: no sign-in, no limits (set WHOP_CLIENT_ID to enable accounts).")
    # Cloud hosts (Render/Railway/Fly) inject $PORT; locally default to 5051
    # (5000 is hijacked by macOS AirPlay Receiver).
    port = int(os.environ.get("PORT", "5051"))
    print(f"\n  Clip Reels running at  http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
