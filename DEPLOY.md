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

## Step 2 — Create the app on Render (Blueprint)
> **Not Vercel.** Vercel runs short serverless functions with a read-only disk,
> no ffmpeg and no background worker, so this app crashes there
> (`FUNCTION_INVOCATION_FAILED`). Delete the Vercel project (or disconnect the
> repo) so pushes to `main` stop deploying there.

1. Sign up at **render.com** and connect your GitHub.
2. **New → Blueprint →** pick the `reelscut` repo. Render reads `render.yaml` and
   sets up everything: Docker build, the 1 CPU / 2 GB instance, Frankfurt region,
   and a 5 GB persistent disk at `/app/data`. Change the plan/region in
   `render.yaml` first if you want something else (the region can't change later).
   - The disk holds the SQLite database (accounts, job queue, minutes used).
     Rendered clips in `/app/jobs` are temporary (deleted after `JOB_TTL_HOURS`);
     set up R2 (Phase 3) so the clips themselves survive.
3. Render asks for the secret values:
   - `ANTHROPIC_API_KEY`, `SARVAM_API_KEY` — your keys.
   - `PUBLIC_URL` — the service URL, e.g. `https://reelscut.onrender.com`
     (you can fill it in after the first deploy).
   - `WHOP_*` — see "Accounts, plans & payments" below. ⚠️ Without
     `WHOP_CLIENT_ID` the app has **no login and no limits**: fine for a quick
     private test, but anyone with the URL could use your API keys.
4. **Apply.** The first build takes a while (it compiles whisper.cpp and downloads
   the model). Watch the logs. Pushes to `main` redeploy automatically.

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
| Phase 5: accounts + payments | create the Whop app + products | ✅ done (Whop) |

## Honest note
The free tiers won't run this (video + whisper need real RAM/CPU). Budget
**~$7–25/mo** for the worker once you're past testing. Don't pay for the bigger
phases until the worker is live and you've shown it to a few creators.

## Accounts, plans & payments (Whop)
Sign-in and payment both go through **Whop**. Without `WHOP_CLIENT_ID` the app
runs in **local mode**: no login and no limits, as it always has on your machine.

1. **Whop → Developer → create an app.** Add the redirect URI
   `https://<your-site>/auth/callback`. Copy the **client id** and **client secret** (OAuth tab), and create a **company**
   **API key** with permission to read members/access.
2. **Create one Whop product per paid plan**, e.g. *Creator* (300 min/month) and
   *Pro* (1200 min/month). Note each product id (`prod_...`).
3. **Environment variables:**

| Variable | Example | What it does |
|---|---|---|
| `WHOP_CLIENT_ID` | `app_xxx` | Turns on "Sign in with Whop" |
| `WHOP_CLIENT_SECRET` | from the app's OAuth tab | Required by Whop to finish sign-in |
| `PUBLIC_URL` | `https://reelscut.app` | Your site address (used for the redirect URI; enables secure cookies) |
| `WHOP_API_KEY` | `...` | Checks which product each user has bought |
| `WHOP_PLANS` | `prod_aaa:Creator:300:$19/mo,prod_bbb:Pro:1200:$39/mo` | Product → plan name → minutes per month → price label (optional, shown on the landing page's pricing section) |
| `FREE_MINUTES` | `30` | Free minutes per month for everyone else |
| `WHOP_STORE_URL` | `https://whop.com/your-store/` | Where the **Upgrade** button goes |
| `SECRET_KEY` | long random string | Signs login cookies (auto-generated into `data/` if unset) |

**How it works:**
- **Charging.** Usage is counted in source-video minutes, per calendar month (UTC). A video is only charged when it finishes successfully.
- **Plan checks.** The user's plan is checked with Whop at most every 10 minutes and again right before each video, so upgrades and cancellations take effect without webhooks. If Whop is unreachable, the last known plan is kept.
- **Queue.** Each user can have one video in the queue at a time. The queue holds at most `MAX_QUEUE` jobs (default 20).
- **Worker.** The job worker (`python worker.py`) runs beside the web server. The Dockerfile starts both, and `python app.py` runs a worker in-process for local use.

**Other limits** (optional): `MAX_UPLOAD_MB` (2048), `JOB_TTL_HOURS` (24, `0` = keep forever),
`MAX_PENDING_UPLOADS` (3), `MIN_FREE_GB` (5).
