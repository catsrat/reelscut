"""
Job worker: takes queued jobs from the database and runs the pipeline, one at a
time. Run it next to the web app:

    python worker.py

Several workers can run at once (each claims jobs atomically), e.g. one per
CPU-heavy machine. `python app.py` also starts one in-process for local use.
"""

import json
import os
import threading
import time
import traceback

import billing
import db
import pipeline
import storage

ROOT = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(ROOT, "jobs")
POLL_SECS = 2
HEARTBEAT_SECS = 30
STALE_AFTER_SECS = HEARTBEAT_SECS * 10   # no heartbeat this long = worker died


class QuotaExceeded(RuntimeError):
    pass


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


def run_job(job):
    job_id = job["id"]
    opts = json.loads(job["opts"])
    workdir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(workdir, exist_ok=True)
    user = db.get_user(job["user_id"])
    charged = {"minutes": 0.0}

    def progress(pct, message):
        db.update_progress(job_id, pct, message)
        print(f"[{job_id}] {pct}% — {message}", flush=True)

    def on_duration(secs):
        # Re-check the plan with Whop now (not the cached one): the quota is
        # what decides whether we spend the compute on this video.
        mins = secs / 60
        q = billing.quota(user, force=True)
        if not q["unlimited"] and mins > q["left"] + 0.01:
            raise QuotaExceeded(
                f"This video is {mins:.0f} min, but your {q['plan']} plan has "
                f"{q['left']:.0f} min left this month. Upgrade your plan or "
                "trim the video."
            )
        charged["minutes"] = mins

    print(f"[{job_id}] worker started (user={job['user_id']}, "
          f"source={'upload' if opts.get('source_file') else 'link'})", flush=True)
    compliance = None
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        sarvam_key = os.environ.get("SARVAM_API_KEY", "").strip()

        # Read clipping-campaign rules first and auto-apply them, so the clip is
        # built compliant and we can hand back a payment-safe checklist.
        rules_text = opts.get("rules") or ""
        if rules_text and api_key:
            progress(3, "Reading the campaign rules...")
            try:
                req = pipeline.parse_campaign_rules(rules_text, api_key)
                compliance = _apply_rules(req, opts)
            except Exception as e:
                # best-effort: never block the clip, but tell the owner why
                print(f"[ai] campaign rules not read: {e}", flush=True)

        results = pipeline.run_pipeline(
            opts.get("url"), workdir, api_key, progress,
            language=opts["language"], music=opts["music"],
            caption_style=opts["caption_style"], sarvam_key=sarvam_key,
            music_volume=opts["music_volume"], split_mode=opts["split_mode"],
            cam_corner=opts["cam_corner"], cam_size=opts["cam_size"],
            source_file=opts.get("source_file"),
            logo_file=opts.get("logo_file"),
            logo_scale=opts.get("logo_scale", 0.16),
            logo_corner=opts.get("logo_corner", "top-right"),
            clip_mode=opts.get("clip_mode", "moments"),
            captions=opts.get("captions", True),
            punchy=opts.get("punchy", False),
            punch_zoom=opts.get("punch_zoom", False),
            color_pop=opts.get("color_pop", False),
            auto_crop=opts.get("auto_crop", False),
            min_clip=opts.get("min_clip"),
            max_clip=opts.get("max_clip"),
            on_duration=on_duration,
        )
        # If R2 is configured, upload clips and attach durable URLs.
        if storage.enabled():
            for r in results:
                url = storage.upload_and_url(
                    os.path.join(workdir, r["file"]), f"{job_id}/{r['file']}"
                )
                if url:
                    r["url"] = url
        db.finish(job_id, {"clips": results, "compliance": compliance,
                           "ai": bool(api_key)}, charged["minutes"])
    except Exception as e:  # surface the real error to the UI
        db.fail(job_id, str(e), {"compliance": compliance} if compliance else None)
        print(f"[{job_id}] ERROR: {e}", flush=True)
        if not isinstance(e, QuotaExceeded):
            traceback.print_exc()


def _heartbeat(job_id, stop):
    while not stop.wait(HEARTBEAT_SECS):
        try:
            db.heartbeat(job_id)
        except Exception:
            pass


def work_forever(stop=None):
    stop = stop or threading.Event()
    db.init()
    print("[worker] waiting for jobs", flush=True)
    while not stop.is_set():
        try:
            db.fail_stale_running(STALE_AFTER_SECS)
            job = db.claim_next()
        except Exception as e:
            print(f"[worker] queue error: {e}", flush=True)
            job = None
        if not job:
            stop.wait(POLL_SECS)
            continue
        # Long steps (transcription) don't report progress; the heartbeat
        # tells other workers this job is still alive.
        beat = threading.Event()
        threading.Thread(target=_heartbeat, args=(job["id"], beat), daemon=True).start()
        try:
            run_job(job)
        finally:
            beat.set()


if __name__ == "__main__":
    work_forever()
