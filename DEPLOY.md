# Deploy guide — getting Clip Reels live

This covers **Phase 2: deploy the worker** (the engine) to Render. Later phases
(R2 storage, Vercel frontend, accounts, payments) build on top of this.

## What's already done (Phase 1) ✅
- App reads any video via **file upload** (no YouTube needed).
- Fonts are **bundled** (`fonts/`), so it renders the same on Linux.
- `Dockerfile` packages everything (ffmpeg, whisper.cpp, fonts, English model).
- Server reads `$PORT` and runs under gunicorn.

> ⚠️ The Dockerfile hasn't been test-built (no Docker on the Mac). Expect to fix
> 1–2 small things on the first deploy — that's normal. I'll help debug the logs.

## Step 1 — Put the code on GitHub
1. Create a free **GitHub** account if you don't have one.
2. Create a new **private** repo called `clip-reels`.
3. From `~/clip-reels`, push the code (ask me and I'll give you the exact commands).
   - `.dockerignore` already excludes `venv/`, `jobs/`, `models/`.

## Step 2 — Create the worker on Render
1. Sign up at **render.com** (free to start).
2. **New → Web Service → Build from a Git repo →** pick your `clip-reels` repo.
3. Render auto-detects the `Dockerfile`. Settings:
   - **Instance type:** start with **Standard** (the free/starter tier has too
     little RAM/CPU for video + whisper). ~$7–25/mo.
   - **Disk:** add a small persistent disk (e.g. 10 GB) mounted at `/app/jobs`
     so clips survive (until we move to R2 in Phase 3).
4. **Environment variables** (Settings → Environment):
   - `ANTHROPIC_API_KEY` = your key
   - `SARVAM_API_KEY` = your key
   - (`PORT` is set by Render automatically)
5. **Create Web Service.** First build takes a while (it compiles whisper.cpp +
   downloads the model). Watch the logs.

## Step 3 — Test it
- Open the Render URL → upload a short video → confirm you get clips.
- **Note:** the "YouTube link" tab will NOT work on the server (no browser
  cookies, and datacenter IPs are blocked). **Upload is the cloud path.** We'll
  hide the link tab for the cloud build.

## What needs you (accounts), what needs me (code)
| Task | You | Me |
|---|---|---|
| GitHub repo + push | create account | give commands |
| Render service | create + set env vars | fix Dockerfile/build issues |
| Phase 3: R2 storage | create Cloudflare R2 + keys | wire it into the app |
| Phase 4: Vercel frontend | create account | build the Next.js page |
| Phase 5: accounts + payments | create Clerk + Stripe/Razorpay | wire them in |

## Honest note
The free tiers won't run this (video + whisper need real RAM/CPU). Budget
**~$7–25/mo** for the worker once you're past testing. Don't pay for the bigger
phases until the worker is live and you've shown it to a few creators.
