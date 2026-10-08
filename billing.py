"""
Plans and monthly minute quotas, sold through Whop.

Users get FREE_MINUTES of video per calendar month. Buying one of your Whop
products unlocks a bigger monthly quota. We ask Whop which product the user has
access to (cached for a few minutes), so cancellations and upgrades apply on
their own — no webhooks needed.

  FREE_MINUTES     video minutes/month without a paid plan (default 30)
  WHOP_PLANS       paid plans as "prod_id:Name:minutes[:price[:link]]",
                   comma-separated, e.g.
                   "prod_abc:Creator:300:$19/mo:https://whop.com/you/products/creator/"
                   Price and link are optional and only used on the pricing
                   section (the link is where that plan's "Get" button goes).
  WHOP_API_KEY     Whop API key (Developer dashboard) used for the access check
  WHOP_STORE_URL   where the "Upgrade" button sends people (your Whop page)
"""

import os
import time
from dataclasses import dataclass

import httpx

import db

WHOP_API = "https://api.whop.com/api/v1"
PLAN_CACHE_SECS = 10 * 60
LOCAL_USER = "local"


@dataclass(frozen=True)
class Plan:
    key: str            # "free", "unlimited", or the Whop product id
    name: str
    minutes: float      # monthly quota; 0 with unlimited=True means no cap
    unlimited: bool = False
    price: str = ""     # display label for the pricing section, e.g. "$19/mo"
    url: str = ""       # this plan's Whop page (pricing section "Get" button)


FREE = Plan("free", "Free", float(os.environ.get("FREE_MINUTES", "30")))
UNLIMITED = Plan("unlimited", "Local", 0, unlimited=True)


def paid_plans():
    """Parse WHOP_PLANS, biggest quota first (so the best plan a user owns wins)."""
    plans = []
    for item in os.environ.get("WHOP_PLANS", "").split(","):
        # maxsplit=4: the optional link is last and keeps its own "https://".
        parts = [p.strip() for p in item.split(":", 4)]
        if len(parts) < 3 or not parts[0]:
            continue
        price = parts[3] if len(parts) > 3 else ""
        url = parts[4] if len(parts) > 4 and parts[4].startswith(("https://", "http://")) else ""
        try:
            plans.append(Plan(parts[0], parts[1] or parts[0], float(parts[2]),
                              price=price, url=url))
        except ValueError:
            print(f"[billing] ignoring bad WHOP_PLANS entry: {item!r}", flush=True)
    return sorted(plans, key=lambda p: p.minutes, reverse=True)


def _plan_by_key(key):
    if key == UNLIMITED.key:
        return UNLIMITED
    return next((p for p in paid_plans() if p.key == key), FREE)


def has_access(user_id, product_id):
    """True/False from Whop, or None if Whop couldn't be asked."""
    api_key = os.environ.get("WHOP_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        r = httpx.get(
            f"{WHOP_API}/users/{user_id}/access/{product_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        r.raise_for_status()
        return bool(r.json().get("has_access"))
    except Exception as e:
        print(f"[billing] Whop access check failed for {user_id}: {e}", flush=True)
        return None


def current_plan(user, force=False):
    """The user's plan, re-checked with Whop at most every PLAN_CACHE_SECS."""
    if user["id"] == LOCAL_USER:
        return UNLIMITED
    if not force and time.time() - user["plan_checked_at"] < PLAN_CACHE_SECS:
        return _plan_by_key(user["plan"])

    plan, unsure = FREE, False
    for p in paid_plans():
        ok = has_access(user["id"], p.key)
        if ok:
            plan = p
            break
        if ok is None:
            unsure = True
    if plan is FREE and unsure:
        # Whop unreachable: keep the last known plan rather than downgrade a
        # paying user during an outage. Retry on the next request.
        return _plan_by_key(user["plan"])
    db.set_plan(user["id"], plan.key)
    return plan


def quota(user, force=False):
    """Everything the UI and the worker need to know about the user's minutes."""
    plan = current_plan(user, force=force)
    used = db.minutes_used(user["id"])
    left = float("inf") if plan.unlimited else max(0.0, plan.minutes - used)
    return {
        "plan": plan.name,
        "paid": plan.key not in (FREE.key, UNLIMITED.key),
        "unlimited": plan.unlimited,
        "minutes": plan.minutes,
        "used": round(used, 1),
        "left": left,
        "upgrade_url": os.environ.get("WHOP_STORE_URL", "").strip(),
    }


def quota_json(q):
    """quota() made JSON-safe (no infinity)."""
    return {**q, "left": None if q["unlimited"] else round(q["left"], 1)}
