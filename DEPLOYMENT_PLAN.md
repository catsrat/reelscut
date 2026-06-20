# Clip Reels — Cloud Deployment Plan

How to take the local Mac app live as a real subscription product.

## The core idea: 3 pieces, not 1

The local app does everything on your Mac. In the cloud it splits into:

```
   USER'S BROWSER
        │  (uploads a video, watches progress, downloads clips)
        ▼
┌─────────────────────┐     ┌──────────────────────────┐     ┌────────────────┐
│  WEB APP (UI + API)  │     │   WORKER (the engine)     │     │   STORAGE      │
│  Vercel — Next.js    │ ──► │  Render / Railway / Fly   │ ──► │  Cloudflare R2 │
│  start job, status   │ que │  ffmpeg + clip + captions │     │  finished mp4s │
└─────────────────────┘  ue └────────────┬─────────────┘     └────────────────┘
        │                                 │ audio
        │                                 ▼
        │                   ┌──────────────────────────┐
        │                   │  TRANSCRIPTION (cloud STT)│
        │                   │  Sarvam (Indian langs) /  │
        │                   │  Deepgram / Whisper API   │
        │                   └──────────────────────────┘
        ▼
┌─────────────────────┐
│  ACCOUNTS + BILLING  │  Auth (Clerk/Supabase) + Payments (Razorpay/Stripe)
│  + a small database  │  Postgres (Supabase/Neon) = users, jobs, usage
└─────────────────────┘
```

**Why this split:** Vercel runs the website beautifully but cannot run minutes-long
video jobs (it times out, has no real disk, no GPU). The heavy work goes to a
"worker" host built for long jobs.

## Recommended services (small-scale, founder-friendly)

| Need | Pick | Why | Rough cost |
|---|---|---|---|
| Website + UI | **Vercel** (Next.js) | What it's made for; great free tier | $0 → $20/mo |
| Worker (ffmpeg/clipping) | **Render** or **Railway** background worker | Long jobs, simple deploy. CPU-only is fine (transcription moves to an API) | ~$7–25/mo |
| Job queue | **Inngest** (free tier) or Upstash Redis | Runs jobs in background, retries, status | $0 to start |
| Transcription | **Sarvam AI** (Indian langs) | Fixes Telugu/Hindi/Tamil quality AND removes GPU need | per-minute (see below) |
| Moment selection | **Claude API** (already built) | Picks clips + titles + mood | cents/video |
| File storage | **Cloudflare R2** | Cheap, no egress fees | ~$0–5/mo |
| Database | **Supabase** or **Neon** Postgres | Users, jobs, usage metering. Free tier ok early | $0 → $25/mo |
| Auth | **Clerk** or Supabase Auth | Login/signup done for you | $0 → $25/mo |
| Payments | **Razorpay** (India) or Stripe | Subscriptions, UPI/cards | % per transaction |

**Floor cost while validating:** roughly **$0–50/month** plus per-video API costs.
You only pay real money once people actually use it.

## Per-video cost (sets your pricing floor)

For a ~30-min source video:
- **Transcription (Sarvam):** the main variable cost — billed per audio-minute. Confirm current rate on their pricing page.
- **Claude (moment selection):** ~1–5 cents (only the transcript text is sent).
- **Compute + storage + bandwidth:** small.

➡️ Add these up per video, then price tiers (e.g. credits = "X minutes/month")
comfortably above cost. This is the unit economics of the whole business.

## What has to be BUILT/CHANGED (in order)

**Phase 1 — Product change: uploads, not YouTube links** *(required for cloud)*
- Replace "paste link" with file upload (drag & drop).
- Kills the YouTube bot-blocking + Terms-of-Service problem in one move.
- Refactor `pipeline.py` into a worker function (it's already cleanly separated).

**Phase 2 — Swap local whisper → Sarvam API**
- One function change: send audio to Sarvam, get word-level transcript back.
- Removes the 1.5GB models + GPU dependency; fixes Telugu quality.

**Phase 3 — Port the rendering to Linux**
- Bundle fonts in the worker image (macOS fonts won't exist on a server):
  - Captions: Poppins, Anton (already have them).
  - Indian scripts: **Noto Sans Telugu / Devanagari / Tamil** (free, OFL).
- Install `ffmpeg` + `libraqm` (HarfBuzz shaping) in the container.

**Phase 4 — Web app + storage + queue**
- Next.js UI on Vercel (upload form, progress, results gallery).
- Worker pulls jobs from a queue; writes clips to R2; updates job status in DB.

**Phase 5 — Accounts + billing**
- Sign up / log in.
- Usage metering (minutes processed per user).
- Razorpay subscription tiers (e.g. Free trial / Creator / Pro).

## Honest reality check

- This is a **real build (weeks, not a toggle)** and ongoing monthly costs. I do
  the building, but it's a serious step up from a local script.
- **Do Phase 1–2 only after validating** that creators want this and will pay.
  Otherwise you're paying server + API bills for an audience of zero.
- The wedge that justifies all of it: **Indian-language clipping done right**
  (Telugu/Hindi/Tamil), where Sarvam + this pipeline beats the English-first
  incumbents.

## Suggested sequence

1. Validate: make demo reels → show 10 creators → confirm they'd pay.
2. If yes → Phase 1 (uploads) + Phase 2 (Sarvam): the product becomes
   cloud-ready and Telugu-accurate.
3. Then Phase 3–5: deploy, accounts, payments → launch a paid beta.
