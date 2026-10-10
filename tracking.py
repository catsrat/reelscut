"""Server-side TikTok events (Events API), alongside the browser pixel.

Many browsers and ad blockers stop the pixel in the page, so the two events
that matter most for ads — a sign-up and a plan purchase — are also sent from
here. Each one carries the same event_id as its pixel twin, so TikTok counts
it once.

Nothing is sent unless TIKTOK_ACCESS_TOKEN is set, and only for visitors whose
browser has tracking on: the page sets the rc_ads cookie when the pixel starts
(straight away outside Europe, only after "Accept" in Europe). No emails or
phone numbers are sent: TikTok matches on its own cookie / click id, the IP
and browser, and a hash of the Whop user id.

Env:
  TIKTOK_PIXEL_ID          the pixel (default: the REELSCUT pixel; "" = off)
  TIKTOK_ACCESS_TOKEN      Events Manager -> the pixel -> Settings -> Generate
                           access token. A secret: set it on the host only.
  TIKTOK_TEST_EVENT_CODE   optional: the code from the "Test events" tab, so
                           events show up there while you check the setup
"""
import hashlib
import os
import threading
import time

import httpx
from flask import request

# A pixel ID is public (it's in the page source), so the default lives here.
PIXEL_ID = os.environ.get("TIKTOK_PIXEL_ID", "DB5A7HBC77U6DDDLBL6G").strip()
ENDPOINT = os.environ.get(
    "TIKTOK_EVENTS_URL", "https://business-api.tiktok.com/open_api/v1.3/event/track/")


def browser_allows():
    """True when this visitor's browser has the pixel running (see _tiktok.html)."""
    return request.cookies.get("rc_ads") == "1"


def _client_ip():
    # Behind the host's proxy the visitor is the first X-Forwarded-For entry.
    return (request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or request.remote_addr or "")


def send(event, event_id, properties=None, user_id=None, page_path=None):
    """Send one event for the visitor behind the current request. Returns True
    if it was queued. Never raises and never slows the request down: the call
    to TikTok happens on a background thread."""
    token = os.environ.get("TIKTOK_ACCESS_TOKEN", "").strip()
    if not (PIXEL_ID and token and browser_allows()):
        return False
    user = {"ip": _client_ip(), "user_agent": request.headers.get("User-Agent", "")[:500]}
    for key, cookie in (("ttp", "_ttp"), ("ttclid", "rc_ttclid")):
        value = request.cookies.get(cookie, "")
        if value and len(value) <= 500:
            user[key] = value
    if user_id:
        user["external_id"] = hashlib.sha256(str(user_id).encode()).hexdigest()

    props = dict(properties or {})
    # The API wants the product under "contents"; the pixel takes it flat.
    content = {k: props.pop(k) for k in ("content_id", "content_name") if k in props}
    if content:
        if "value" in props:
            content.update(price=props["value"], quantity=1)
        props["contents"] = [content]
    payload = {
        "event_source": "web",
        "event_source_id": PIXEL_ID,
        "data": [{
            "event": event,
            "event_time": int(time.time()),
            "event_id": event_id,
            "user": user,
            # path only: a query string can hold sign-in codes
            "page": {"url": request.url_root.rstrip("/") + (page_path or request.path)},
            "properties": props,
        }],
    }
    test_code = os.environ.get("TIKTOK_TEST_EVENT_CODE", "").strip()
    if test_code:
        payload["test_event_code"] = test_code
    threading.Thread(target=_post, args=(payload, token), daemon=True).start()
    return True


def _post(payload, token):
    event = payload["data"][0]["event"]
    try:
        r = httpx.post(ENDPOINT, json=payload, headers={"Access-Token": token}, timeout=10)
        try:
            body = r.json()
        except ValueError:
            body = {}
        if r.status_code >= 400 or body.get("code", 0) != 0:
            print(f"[tiktok] {event} not accepted: HTTP {r.status_code} "
                  f"{str(body.get('message') or r.text)[:200]}", flush=True)
        else:
            print(f"[tiktok] {event} sent", flush=True)
    except Exception as e:  # ad reporting must never break the app
        print(f"[tiktok] {event} send failed: {e}", flush=True)
