"""
Sign in with Whop (OAuth 2.1 + PKCE).

  WHOP_CLIENT_ID   your Whop app's client id (Developer dashboard).
                   Unset = local mode: no login, a single unlimited "local" user
                   (how the app has always worked on your own machine).
  WHOP_CLIENT_SECRET  the app's client secret (Whop app → OAuth tab). Whop
                   rejects the sign-in code exchange without it.
  PUBLIC_URL       this site's address, e.g. https://reelscut.app. Register
                   PUBLIC_URL + "/auth/callback" as the app's redirect URI.
                   Defaults to the address the request came in on.

We only need to know who the user is; paid access is checked separately with
the app's API key (billing.py), so no Whop tokens are stored.
"""

import base64
import functools
import hashlib
import os
import secrets
from urllib.parse import urlencode

import httpx
from flask import (
    Blueprint, jsonify, redirect, render_template_string, request, session
)

import billing
import db

WHOP_OAUTH = "https://api.whop.com/oauth"
SCOPES = "openid profile email"

bp = Blueprint("auth", __name__)


def enabled():
    return bool(os.environ.get("WHOP_CLIENT_ID", "").strip())


def _client_id():
    return os.environ["WHOP_CLIENT_ID"].strip()


def _redirect_uri():
    base = os.environ.get("PUBLIC_URL", "").strip().rstrip("/") or request.url_root.rstrip("/")
    return base + "/auth/callback"


def current_user():
    if not enabled():
        return db.get_user(billing.LOCAL_USER) or db.upsert_user(billing.LOCAL_USER, name="You")
    uid = session.get("uid")
    return db.get_user(uid) if uid else None


def login_required(view):
    """Pages redirect to sign-in; API calls get a 401 the UI can act on."""
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        user = current_user()
        if not user:
            if request.method == "GET" and request.accept_mimetypes.accept_html \
                    and not request.path.startswith(("/status", "/clip", "/me")):
                return redirect("/login?" + urlencode({"next": request.path}))
            return jsonify({"error": "Please sign in again.", "login": "/login"}), 401
        return view(user, *args, **kwargs)
    return wrapped


def _b64url(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _safe_next(url):
    # Only local paths — never bounce a user to another site after login.
    return url if url and url.startswith("/") and not url.startswith("//") else "/app"


@bp.route("/login")
def login():
    if not enabled():
        return redirect("/app")
    verifier = _b64url(secrets.token_bytes(32))
    state = _b64url(secrets.token_bytes(16))
    session["oauth"] = {
        "verifier": verifier,
        "state": state,
        "next": _safe_next(request.args.get("next")),
    }
    params = {
        "response_type": "code",
        "client_id": _client_id(),
        "redirect_uri": _redirect_uri(),
        "scope": SCOPES,
        "state": state,
        "nonce": _b64url(secrets.token_bytes(16)),
        "code_challenge": _b64url(hashlib.sha256(verifier.encode()).digest()),
        "code_challenge_method": "S256",
    }
    return redirect(f"{WHOP_OAUTH}/authorize?{urlencode(params)}")


_ERROR_PAGE = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sign-in problem — Reelscut</title>
<body style="font-family:Inter,-apple-system,Segoe UI,Roboto,sans-serif;background:#0b0b0c;color:#f4f4f5;
display:grid;place-items:center;min-height:100vh;margin:0;padding:16px">
<div style="max-width:400px;width:100%;text-align:center;background:#131315;border:1px solid rgba(255,255,255,.07);
border-radius:20px;padding:36px 28px"><h2 style="margin:0 0 8px;font-size:20px;font-weight:600">Couldn't sign you in</h2>
<p style="color:#a1a1aa;margin:0 0 24px;font-size:15px;line-height:1.55">{{ msg }}</p>
<a href="/login" style="display:inline-block;color:#fff;background:#ff3b4e;padding:10px 18px;
border-radius:10px;text-decoration:none;font-weight:600;font-size:14px">Try again</a></div>"""


def _fail(msg):
    return render_template_string(_ERROR_PAGE, msg=msg), 400


@bp.route("/auth/callback")
def callback():
    oauth = session.pop("oauth", None)
    if request.args.get("error"):
        return _fail(request.args.get("error_description") or "Sign-in was cancelled.")
    if not oauth or not secrets.compare_digest(request.args.get("state", ""), oauth["state"]):
        return _fail("This sign-in link expired. Please start again.")
    code = request.args.get("code")
    if not code:
        return _fail("Whop didn't send a sign-in code.")
    try:
        tok = httpx.post(f"{WHOP_OAUTH}/token", json={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": _redirect_uri(),
            "client_id": _client_id(),
            "client_secret": os.environ.get("WHOP_CLIENT_SECRET", "").strip(),
            "code_verifier": oauth["verifier"],
        }, timeout=15)
        if tok.status_code >= 400:
            # Whop's error body says what's wrong (e.g. "client_secret is
            # required"); it never contains our secret.
            raise RuntimeError(f"token exchange {tok.status_code}: {tok.text[:300]}")
        access_token = tok.json()["access_token"]
        info = httpx.get(f"{WHOP_OAUTH}/userinfo",
                         headers={"Authorization": f"Bearer {access_token}"},
                         timeout=15)
        info.raise_for_status()
        profile = info.json()
    except Exception as e:
        print(f"[auth] Whop sign-in failed: {e}", flush=True)
        return _fail("Whop sign-in failed. Please try again in a minute.")

    user_id = profile.get("sub")
    if not user_id:
        return _fail("Whop didn't tell us who you are.")
    db.upsert_user(user_id, email=profile.get("email"),
                   name=profile.get("name") or profile.get("preferred_username"))
    session.clear()  # fresh session after login (no fixation)
    session.permanent = True
    session["uid"] = user_id
    return redirect(oauth["next"])


@bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect("/")
