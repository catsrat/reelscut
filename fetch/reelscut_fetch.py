"""Reelscut Fetch — gets a YouTube video onto Reelscut from your own computer.

YouTube doesn't let servers download videos, but it's fine with a person on
their own internet connection. So when someone pastes a YouTube link on
reelscut.si, the page opens this app with a one-time link:

    reelscut://fetch?s=https://www.reelscut.si&t=<token>

and the app downloads the video here (with yt-dlp), then uploads it into that
person's Reelscut account, where the reels get made as usual.

18+ videos (casino streams…) need a signed-in YouTube account. "Sign in to
YouTube" opens a browser window with its own profile; the login stays on this
computer and is only ever sent to YouTube — never to Reelscut.

First run sets itself up in %LOCALAPPDATA%\\ReelscutFetch: copies the app
there, registers reelscut:// for this Windows user (no admin), and downloads
the official yt-dlp and FFmpeg builds (checked against their published SHA-256
sums). yt-dlp updates itself once a day, so YouTube changes don't need a new
version of this app.

    ReelscutFetch.exe                 set up / sign in
    ReelscutFetch.exe "reelscut://…"  fetch a video (what the browser runs)
    ReelscutFetch.exe --uninstall     remove the app, tools and login
"""
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import zipfile

VERSION = "1.0.0"
APP = "Reelscut Fetch"
HOME = os.environ.get("REELSCUT_FETCH_HOME") or os.path.join(
    os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "ReelscutFetch")
TOOLS = os.path.join(HOME, "tools")
YTDLP = os.path.join(TOOLS, "yt-dlp.exe")
FFMPEG_DIR = os.path.join(TOOLS, "ffmpeg")
WORK = os.path.join(HOME, "work")
COOKIES = os.path.join(HOME, "youtube-cookies.txt")
PROFILE = os.path.join(HOME, "browser")  # the sign-in browser's own profile
LOG = os.path.join(HOME, "fetch.log")

GH = "https://github.com/"
YTDLP_URL = GH + "yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe"
YTDLP_SUMS = GH + "yt-dlp/yt-dlp/releases/latest/download/SHA2-256SUMS"
# gyan.dev's release build: widely used (it's what winget installs), so Windows
# Smart App Control trusts it. Fresh nightly builds get blocked there.
FFMPEG_ZIP = "ffmpeg-release-essentials.zip"
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/" + FFMPEG_ZIP
FFMPEG_SUMS = FFMPEG_URL + ".sha256"
SAC_MSG = ("Windows blocked {tool} (Smart App Control). Please try again in a few "
           "hours — new versions are trusted once more people use them.")

# The only sites this app will talk to. A reelscut:// link naming any other
# site is refused, so a web page can't point the app somewhere else.
SITES = {"https://www.reelscut.si", "https://reelscut.si", "https://reelscut.onrender.com"}
# Local test servers, only when REELSCUT_FETCH_DEV=1 is set on this computer.
DEV_SITE = re.compile(r"http://(127\.0\.0\.1|localhost):\d{2,5}")

SIGN_IN_URL = ("https://accounts.google.com/ServiceLogin?service=youtube&passive=true"
               "&continue=https%3A%2F%2Fwww.youtube.com%2F")
UPLOAD_THREADS = 3
NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW: tools run without a console flash


class FetchError(Exception):
    """A problem to show the user as-is."""


class NeedSignIn(FetchError):
    pass


class Cancelled(FetchError):
    pass


def log(msg):
    try:
        os.makedirs(HOME, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + str(msg) + "\n")
    except OSError:
        pass


def is_youtube(url):
    m = re.match(r"https?://([^/?#]+)", (url or "").strip(), re.I)
    host = m.group(1).lower().split("@")[-1].split(":")[0] if m else ""
    return host in ("youtu.be", "youtube.com") or host.endswith(".youtube.com")


def parse_link(link):
    """reelscut://fetch?s=<site>&t=<token> -> (site, token), or FetchError."""
    u = urllib.parse.urlsplit(link.strip())
    q = urllib.parse.parse_qs(u.query)
    site = (q.get("s") or [""])[0].rstrip("/")
    token = (q.get("t") or [""])[0]
    if u.scheme != "reelscut" or u.netloc != "fetch" or not token:
        raise FetchError("That link isn't a Reelscut Fetch link.")
    dev = os.environ.get("REELSCUT_FETCH_DEV") == "1" and DEV_SITE.fullmatch(site)
    if site not in SITES and not dev:
        raise FetchError("That link isn't from Reelscut, so it was ignored.")
    return site, token


def hms(secs):
    secs = int(secs)
    return f"{secs // 3600}:{secs % 3600 // 60:02d}:{secs % 60:02d}"


# ------------------------------------------------------------------ setup

def _download(url, dest, on_progress=None):
    req = urllib.request.Request(url, headers={"User-Agent": f"ReelscutFetch/{VERSION}"})
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if on_progress and total:
                on_progress(done / total)
    os.replace(tmp, dest)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _expected_sum(sums_url, name):
    req = urllib.request.Request(sums_url, headers={"User-Agent": f"ReelscutFetch/{VERSION}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        for line in r.read().decode("utf-8", "replace").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("*") == name:
                return parts[0].lower()
            if len(parts) == 1 and re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
                return parts[0].lower()  # a file with just the hash
    raise FetchError(f"Couldn't verify the {name} download. Please try again later.")


def _runs(exe, *args):
    """True if the tool starts. Smart App Control blocks untrusted files with
    no output at all, which tools then misreport (yt-dlp: "not installed")."""
    try:
        return subprocess.run([exe, *args], capture_output=True, timeout=60,
                              creationflags=NO_WINDOW).returncode == 0
    except OSError:
        return False


def ensure_tools(status):
    """Download yt-dlp and FFmpeg once (verified), and update yt-dlp daily.
    An FFmpeg that's already installed (on PATH) is used as it is."""
    global FFMPEG_DIR
    os.makedirs(TOOLS, exist_ok=True)
    if not os.path.exists(os.path.join(FFMPEG_DIR, "ffmpeg.exe")):
        on_path = shutil.which("ffmpeg")
        if on_path and _runs(on_path, "-version"):
            FFMPEG_DIR = os.path.dirname(on_path)
    if not os.path.exists(YTDLP):
        status("Setting up (one time): downloading yt-dlp…", 0)
        want = _expected_sum(YTDLP_SUMS, "yt-dlp.exe")
        tmp = YTDLP + ".new"
        _download(YTDLP_URL, tmp, lambda f: status(None, f * 15))
        if _sha256(tmp) != want:
            os.remove(tmp)
            raise FetchError("The yt-dlp download didn't match its checksum. Please try again.")
        os.replace(tmp, YTDLP)
    elif time.time() - os.path.getmtime(YTDLP) > 24 * 3600:
        status("Checking for yt-dlp updates…", None)
        try:  # yt-dlp verifies its own update
            subprocess.run([YTDLP, "-U"], capture_output=True, timeout=120,
                           creationflags=NO_WINDOW)
            os.utime(YTDLP)
        except Exception as e:
            log(f"yt-dlp update skipped: {e}")

    if not os.path.exists(os.path.join(FFMPEG_DIR, "ffmpeg.exe")):
        status("Setting up (one time): downloading FFmpeg, about 115 MB…", 15)
        want = _expected_sum(FFMPEG_SUMS, FFMPEG_ZIP)
        zpath = os.path.join(TOOLS, FFMPEG_ZIP)
        _download(FFMPEG_URL, zpath, lambda f: status(None, 15 + f * 80))
        if _sha256(zpath) != want:
            os.remove(zpath)
            raise FetchError("The FFmpeg download didn't match its checksum. Please try again.")
        status("Setting up: unpacking FFmpeg…", 96)
        tmp = FFMPEG_DIR + ".new"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        with zipfile.ZipFile(zpath) as z:
            for m in z.infolist():
                parts = m.filename.split("/")
                # keep just bin/*: ffmpeg.exe, ffprobe.exe and their DLLs
                if len(parts) == 3 and parts[1] == "bin" and parts[2]:
                    with z.open(m) as src, open(os.path.join(tmp, parts[2]), "wb") as dst:
                        shutil.copyfileobj(src, dst)
        os.remove(zpath)
        if not os.path.exists(os.path.join(tmp, "ffmpeg.exe")):
            raise FetchError("FFmpeg didn't unpack properly. Please try again.")
        shutil.rmtree(FFMPEG_DIR, ignore_errors=True)
        os.replace(tmp, FFMPEG_DIR)
        status("Setup finished.", 100)
    if not _runs(YTDLP, "--version"):
        raise FetchError(SAC_MSG.format(tool="yt-dlp"))
    if not _runs(os.path.join(FFMPEG_DIR, "ffmpeg.exe"), "-version"):
        raise FetchError(SAC_MSG.format(tool="FFmpeg"))


def install():
    """Copy the app to HOME and register reelscut:// for this Windows user."""
    if not getattr(sys, "frozen", False) or os.environ.get("REELSCUT_FETCH_NO_INSTALL"):
        return  # running from source / tests: leave Windows alone
    import winreg
    os.makedirs(HOME, exist_ok=True)
    me = os.path.abspath(sys.executable)
    target = os.path.join(HOME, "ReelscutFetch.exe")
    if me.lower() != target.lower():
        try:
            shutil.copy2(me, target)
        except OSError as e:  # an older copy may be running; use this one
            log(f"copy to {target} failed: {e}")
            target = me
    base = r"Software\Classes\reelscut"
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, base) as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, "URL:Reelscut Fetch")
        winreg.SetValueEx(k, "URL Protocol", 0, winreg.REG_SZ, "")
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, base + r"\DefaultIcon") as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, f'"{target}",0')
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, base + r"\shell\open\command") as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, f'"{target}" "%1"')


def uninstall():
    try:
        import winreg
        for sub in (r"\shell\open\command", r"\shell\open", r"\shell", r"\DefaultIcon", ""):
            try:
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\reelscut" + sub)
            except OSError:
                pass
    except ImportError:
        pass
    # The running exe can't delete itself; everything else goes.
    for name in os.listdir(HOME) if os.path.isdir(HOME) else []:
        path = os.path.join(HOME, name)
        if name.lower() != "reelscutfetch.exe":
            shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else _rm(path)


def _rm(path):
    try:
        os.remove(path)
    except OSError:
        pass


# ------------------------------------------------------------------ sign-in

def find_browser():
    """Chrome if installed, else Edge (always on Windows 10/11)."""
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    local = os.environ.get("LOCALAPPDATA", "")
    for path in (
        os.path.join(pf, r"Google\Chrome\Application\chrome.exe"),
        os.path.join(pf86, r"Google\Chrome\Application\chrome.exe"),
        os.path.join(local, r"Google\Chrome\Application\chrome.exe"),
        os.path.join(pf86, r"Microsoft\Edge\Application\msedge.exe"),
        os.path.join(pf, r"Microsoft\Edge\Application\msedge.exe"),
    ):
        if path and os.path.exists(path):
            return path
    raise FetchError("Couldn't find Chrome or Edge to sign in with.")


def sign_in(status, on_browser=None):
    """Open a browser window (own profile) at YouTube sign-in; when the person
    closes it, save the YouTube cookies to COOKIES for yt-dlp."""
    exe = find_browser()
    os.makedirs(PROFILE, exist_ok=True)
    status("Sign in to YouTube in the browser window, then close that window.", None)
    p = subprocess.Popen([exe, f"--user-data-dir={PROFILE}", "--no-first-run",
                          "--no-default-browser-check", "--new-window", SIGN_IN_URL])
    if on_browser:
        on_browser(p)
    p.wait()
    status("Saving your YouTube sign-in on this computer…", None)
    n = export_cookies(exe)
    log(f"sign-in saved ({n} cookies)")


def export_cookies(exe):
    """Read the sign-in profile's cookies through the browser itself (headless,
    DevTools on a random local port) and write them as cookies.txt."""
    import websocket  # websocket-client

    port_file = os.path.join(PROFILE, "DevToolsActivePort")
    _rm(port_file)
    p = subprocess.Popen([exe, f"--user-data-dir={PROFILE}", "--headless=new",
                          "--remote-debugging-port=0", "--no-first-run", "about:blank"],
                         creationflags=NO_WINDOW)
    try:
        deadline = time.time() + 30
        while True:  # the browser writes "<port>\n<path>" once DevTools is up
            try:
                with open(port_file) as f:
                    found = f.read().split()
            except OSError:
                found = []
            if len(found) >= 2:
                port, path = found[:2]
                break
            if time.time() > deadline or p.poll() is not None:
                raise FetchError("Couldn't read the sign-in. Close all windows of "
                                 "that browser and try again.")
            time.sleep(0.2)
        ws = websocket.create_connection(f"ws://127.0.0.1:{port}{path}", timeout=30,
                                         suppress_origin=True)
        try:
            ws.send(json.dumps({"id": 1, "method": "Storage.getCookies"}))
            while True:
                msg = json.loads(ws.recv())
                if msg.get("id") == 1:
                    break
        finally:
            ws.close()
    finally:
        p.kill()
    cookies = [c for c in msg.get("result", {}).get("cookies", [])
               if c["domain"].lstrip(".").endswith(("youtube.com", "google.com"))]
    signed_in = any(c["domain"].lstrip(".").endswith("youtube.com")
                    and c["name"] in ("LOGIN_INFO", "SID", "__Secure-1PSID", "__Secure-3PSID")
                    for c in cookies)
    if not signed_in:
        raise FetchError("You didn't finish signing in to YouTube. Try again and close "
                         "the window only after YouTube shows your account.")
    lines = ["# Netscape HTTP Cookie File"]
    for c in cookies:
        domain = c["domain"]
        lines.append("\t".join([
            domain, "TRUE" if domain.startswith(".") else "FALSE", c.get("path") or "/",
            "TRUE" if c.get("secure") else "FALSE",
            str(int(c["expires"])) if c.get("expires", -1) > 0 else "0",
            c["name"], c["value"],
        ]))
    with open(COOKIES, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(cookies)


# ------------------------------------------------------------------ the job

class Job:
    """One reelscut:// link: ticket -> download -> upload. UI-independent;
    `status(text, pct)` reports progress (pct None = busy, no number)."""

    def __init__(self, site, token, status):
        self.site, self.token, self._status = site, token, status
        self.proc = None
        self.cancelled = False
        self.ticket = None
        self.title = ""
        self._last_sent = 0

    # -- talking to Reelscut
    def api(self, path, data=None, raw=None, method=None):
        headers = {"X-Fetch-Token": self.token, "User-Agent": f"ReelscutFetch/{VERSION}"}
        body = None
        if raw is not None:
            body, headers["Content-Type"] = raw, "application/octet-stream"
        elif data is not None:
            body, headers["Content-Type"] = json.dumps(data).encode(), "application/json"
        req = urllib.request.Request(self.site + path, data=body, headers=headers,
                                     method=method or ("POST" if body is not None else "GET"))
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error")
            except Exception:
                msg = None
            raise FetchError(msg or f"Reelscut answered with an error ({e.code}).")
        except urllib.error.URLError as e:
            raise FetchError(f"Couldn't reach Reelscut — check your internet. ({e.reason})")

    def status(self, text, pct, stage=None, force=False):
        self._status(text, pct)
        now = time.time()
        if stage and (force or now - self._last_sent > 1.0):
            self._last_sent = now
            try:
                self.api("/fetch/api/progress", {"stage": stage, "pct": pct or 0,
                                                 "message": text or "", "title": self.title})
            except FetchError:
                pass  # progress is best-effort

    def fail(self, message):
        try:
            self.api("/fetch/api/fail", {"message": message})
        except FetchError:
            pass

    def cancel(self):
        self.cancelled = True
        if self.proc and self.proc.poll() is None:
            self.proc.kill()

    # -- yt-dlp
    def ytdlp(self, args, on_line=None, cookies=False):
        cmd = [YTDLP, "--ignore-config", "--no-playlist", "--no-warnings",
               "--ffmpeg-location", FFMPEG_DIR]
        if cookies:
            cmd += ["--cookies", COOKIES]
        cmd += args
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace",
                                     creationflags=NO_WINDOW)
        out = []
        for line in self.proc.stdout:
            line = line.rstrip()
            out.append(line)
            if on_line:
                on_line(line)
        self.proc.wait()
        if self.cancelled:
            raise Cancelled("Cancelled.")
        return self.proc.returncode, out

    @staticmethod
    def _needs_login(lines):
        text = "\n".join(lines[-30:]).lower()
        return any(k in text for k in ("sign in to confirm", "confirm your age",
                                       "age-restricted", "inappropriate for some users",
                                       "not a bot", "members-only", "login required"))

    def info(self, url):
        """Video metadata; tries without the YouTube sign-in first."""
        use_cookies = False
        for attempt in (1, 2):
            code, lines = self.ytdlp(["-J", "--skip-download", "--", url], cookies=use_cookies)
            if code == 0:
                for line in reversed(lines):
                    if line.startswith("{"):
                        return json.loads(line), use_cookies
            if self._needs_login(lines) and attempt == 1 and os.path.exists(COOKIES):
                use_cookies = True
                continue
            if self._needs_login(lines):
                raise NeedSignIn("This video needs a signed-in YouTube account "
                                 "(18+ videos, or YouTube asked to confirm you're not a bot).")
            log("info failed:\n" + "\n".join(lines[-15:]))
            last = next((l for l in reversed(lines) if "ERROR" in l), "")
            raise FetchError("YouTube didn't give us this video. "
                             + (last.split("ERROR:", 1)[-1].strip()[:160] if last else ""))

    def run(self):
        t = self.ticket = self.api("/fetch/api/ticket")
        url = t["url"]
        if not is_youtube(url):
            raise FetchError("Reelscut Fetch only downloads YouTube links.")
        ensure_tools(lambda text, pct: self.status(text, pct, "preparing"))
        self.status("Looking at the video…", None, "preparing", force=True)
        meta, use_cookies = self.info(url)
        self.title = (meta.get("title") or "YouTube video")[:200]
        dur = float(meta.get("duration") or 0)
        start, end = t.get("start"), t.get("end")
        if dur and start and start >= dur:
            raise FetchError(f"This video is only {hms(dur)} long — pick a From time inside it.")
        stop = min(end, dur) if (end and dur) else (end or dur)
        length = (stop - (start or 0)) if stop else 0
        if length and length > t["max_minutes"] * 60 + 5:
            raise FetchError(
                (f"This video is {hms(length)} long. " if start is None and end is None
                 else f"That part is {int(length // 60)} minutes long. ")
                + f"Reelscut cuts up to {t['max_minutes']} minutes at a time — set From "
                "and To on reelscut.si to the part you want.")
        left = t.get("minutes_left")
        if left is not None and length and length / 60 > left + 0.01:
            raise FetchError(f"This is {length / 60:.0f} min, but your plan has {left:.0f} "
                             "min left this month. Pick a shorter part or upgrade.")

        path = self.download(url, start, end, use_cookies)
        self.upload(path)
        self.api("/fetch/api/done", {})
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)
        self.status("Sent! Your reels are being made — watch them on reelscut.si.", 100)

    def download(self, url, start, end, use_cookies):
        work = os.path.join(WORK, self.token[:10])
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        args = [
            # 720p is plenty for a 9:16 crop and keeps the upload small
            "-f", "bv*[height<=720][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=720]+ba/b[height<=720]/bv*+ba/b",
            "--merge-output-format", "mp4",
            "--concurrent-fragments", "4", "--retries", "10", "--fragment-retries", "10",
            "--newline", "--progress-template",
            "download:RCPROG %(progress.downloaded_bytes)s %(progress.total_bytes)s "
            "%(progress.total_bytes_estimate)s",
            "-o", os.path.join(work, "video.%(ext)s"),
        ]
        if start is not None or end is not None:
            args += ["--download-sections",
                     f"*{hms(start or 0)}-{hms(end) if end is not None else 'inf'}"]
        part = {"n": 0}  # yt-dlp fetches video, then audio

        def on_line(line):
            if line.startswith("[download] Destination:"):
                part["n"] += 1
            m = re.match(r"RCPROG (\d+) (\S+) (\S+)", line)
            if m:
                done = float(m.group(1))
                total = next((float(x) for x in m.group(2, 3) if x not in ("NA", "None")), 0)
                if total:
                    f = min(done / total, 1.0)
                    pct = f * 85 if part["n"] <= 1 else 85 + f * 15
                    self.status(f"Downloading from YouTube… {pct:.0f}%", pct, "downloading")
            elif "[Merger]" in line:
                self.status("Putting video and sound together…", 100, "downloading")

        self.status("Downloading from YouTube…", None, "downloading", force=True)
        code, lines = self.ytdlp(args + ["--", url], on_line, cookies=use_cookies)
        files = [f for f in os.listdir(work) if f.startswith("video.") and f.endswith(".mp4")]
        if code != 0 or not files:
            log("download failed:\n" + "\n".join(lines[-20:]))
            if self._needs_login(lines):
                raise NeedSignIn("YouTube wants a signed-in account for this video.")
            raise FetchError("The download from YouTube failed. Please try again.")
        return os.path.join(work, files[0])

    def upload(self, path):
        size = os.path.getsize(path)
        if size > self.ticket["max_bytes"]:
            raise FetchError("This video is too big to send. Pick a shorter part with From and To.")
        chunk = self.api("/fetch/api/upload_start", {"size": size}).get("chunk_bytes") or 8 << 20
        offsets = list(range(0, size, chunk))
        sent = {"bytes": 0}
        lock = threading.Lock()
        t0 = time.time()

        def send(offset):
            with open(path, "rb") as f:
                f.seek(offset)
                data = f.read(chunk)
            for attempt in range(6):
                if self.cancelled:
                    raise Cancelled("Cancelled.")
                try:
                    self.api(f"/fetch/api/chunk?offset={offset}", raw=data)
                    break
                except FetchError:
                    if attempt == 5:
                        raise
                    time.sleep(2 * (attempt + 1))
            with lock:
                sent["bytes"] += len(data)
                pct = sent["bytes"] / size * 100
                speed = sent["bytes"] / max(time.time() - t0, 0.1) / 1e6
            self.status(f"Sending to Reelscut… {pct:.0f}% ({speed:.1f} MB/s)", pct, "uploading")

        self.status("Sending to Reelscut…", 0, "uploading", force=True)
        with concurrent.futures.ThreadPoolExecutor(UPLOAD_THREADS) as pool:
            for fut in [pool.submit(send, o) for o in offsets]:
                fut.result()


# ------------------------------------------------------------------ window

def run_window(link):
    import tkinter as tk
    from tkinter import ttk

    BG, SURFACE, TEXT, DIM, RED = "#0b0b0c", "#19191c", "#f4f4f5", "#a1a1aa", "#ff3b4e"
    root = tk.Tk()
    root.title(APP)
    root.configure(bg=BG)
    root.resizable(False, False)
    try:
        root.iconbitmap(default=os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(__file__)),
                                             "icon.ico"))
    except Exception:
        pass
    w, h = 500, 300
    root.geometry(f"{w}x{h}+{(root.winfo_screenwidth() - w) // 2}+{(root.winfo_screenheight() - h) // 3}")
    root.attributes("-topmost", True)
    root.after(1500, lambda: root.attributes("-topmost", False))

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("R.Horizontal.TProgressbar", troughcolor=SURFACE, background=RED,
                    bordercolor=SURFACE, lightcolor=RED, darkcolor=RED, thickness=8)

    head = tk.Frame(root, bg=BG)
    head.pack(fill="x", padx=24, pady=(22, 0))
    tk.Label(head, text="Reel", font=("Segoe UI", 18, "bold"), fg=TEXT, bg=BG).pack(side="left")
    tk.Label(head, text="scut", font=("Segoe UI", 18, "bold"), fg=RED, bg=BG).pack(side="left")
    tk.Label(head, text="  Fetch", font=("Segoe UI", 18), fg=DIM, bg=BG).pack(side="left")
    sub = tk.Label(root, text="", font=("Segoe UI", 10), fg=DIM, bg=BG, anchor="w",
                   justify="left", wraplength=450)
    sub.pack(fill="x", padx=24, pady=(6, 0))
    bar = ttk.Progressbar(root, style="R.Horizontal.TProgressbar", length=452, maximum=100)
    bar.pack(padx=24, pady=(18, 0))
    msg = tk.Label(root, text="", font=("Segoe UI", 10), fg=TEXT, bg=BG, anchor="w",
                   justify="left", wraplength=450)
    msg.pack(fill="x", padx=24, pady=(10, 0))
    buttons = tk.Frame(root, bg=BG)
    buttons.pack(side="bottom", fill="x", padx=24, pady=18)

    def button(text, cmd, primary=False):
        return tk.Button(buttons, text=text, command=cmd, font=("Segoe UI", 10, "bold"),
                         fg="#fff" if primary else TEXT, bg=RED if primary else SURFACE,
                         activebackground="#e2293b" if primary else "#232327",
                         activeforeground="#fff", relief="flat", bd=0, padx=16, pady=7,
                         cursor="hand2")

    state = {"job": None, "busy": False, "browser": None, "closing": False}

    def ui(fn):
        if not state["closing"]:
            root.after(0, fn)

    def set_status(text, pct):
        def apply():
            if text is not None:
                msg.config(text=text)
            if pct is None:
                if str(bar["mode"]) != "indeterminate":
                    bar.config(mode="indeterminate")
                    bar.start(12)
            else:
                if str(bar["mode"]) != "determinate":
                    bar.stop()
                    bar.config(mode="determinate")
                bar["value"] = pct
        ui(apply)

    def set_buttons(*specs):
        def apply():
            for child in buttons.winfo_children():
                child.destroy()
            for text, cmd, primary in specs:
                button(text, cmd, primary).pack(side="right", padx=(8, 0))
        ui(apply)

    def close():
        state["closing"] = True
        job = state["job"]
        if state["busy"] and job:
            job.cancel()
            threading.Thread(target=job.fail, args=("Cancelled in Reelscut Fetch.",),
                             daemon=True).start()
        b = state["browser"]
        if b and b.poll() is None:
            b.kill()
        root.after(300, root.destroy)

    def in_thread(fn):
        def runner():
            state["busy"] = True
            try:
                fn()
            finally:
                state["busy"] = False
        threading.Thread(target=runner, daemon=True).start()

    def do_sign_in(then=None):
        def work():
            try:
                set_buttons(("Cancel", close, False))
                sign_in(set_status, on_browser=lambda p: state.__setitem__("browser", p))
                set_status("Signed in to YouTube. Your login stays on this computer.", 100)
                if then:
                    then()
                else:
                    set_buttons(("Close", close, True))
            except Exception as e:
                show_error(e)
        in_thread(work)

    def show_error(e, job=None):
        if isinstance(e, Cancelled):
            return
        text = str(e) if isinstance(e, FetchError) else "Something went wrong."
        if not isinstance(e, FetchError):
            log(traceback.format_exc())
        set_status(text, 0)
        if isinstance(e, NeedSignIn) and job:
            ui(lambda: sub.config(text=sub.cget("text") + "\nYour login stays on this "
                                  "computer and is only used with YouTube."))
            set_buttons(("Sign in to YouTube", lambda: do_sign_in(then=start_job), True),
                        ("Cancel", close, False))
        else:
            if job:
                job.fail(text)
            set_buttons(("Close", close, False))

    def start_job():
        site, token = parse_link(link)
        job = state["job"] = Job(site, token, set_status)

        def work():
            try:
                set_buttons(("Cancel", close, False))
                ticket = job.api("/fetch/api/ticket")
                ui(lambda: sub.config(text=f"Sending to your Reelscut account: {ticket['account']}"))
                job.run()
                set_buttons(("Close", close, True))
                ui(lambda: root.after(6000, close))
            except Exception as e:
                show_error(e, job)
        in_thread(work)

    def setup_only():
        def work():
            try:
                set_buttons(("Close", close, False))
                ensure_tools(set_status)
                signed = os.path.exists(COOKIES)
                set_status("All set. Go back to reelscut.si, paste a YouTube link and press "
                           "Make my reels." + ("" if not signed else
                                               "\nYou're signed in to YouTube."), 100)
                set_buttons(("Close", close, True),
                            ("Sign in to YouTube again" if signed else
                             "Sign in to YouTube (for 18+ videos)", lambda: do_sign_in(), False))
            except Exception as e:
                show_error(e)
        in_thread(work)

    root.protocol("WM_DELETE_WINDOW", close)
    if link:
        try:
            parse_link(link)
            start_job()
        except FetchError as e:
            set_status(str(e), 0)
            set_buttons(("Close", close, False))
    else:
        sub.config(text="Gets YouTube videos into Reelscut from this computer.")
        setup_only()
    root.mainloop()


def run_console(link):
    """Same job without a window (for testing): python reelscut_fetch.py --console <link>"""
    def status(text, pct):
        if text:
            line = f"[{'..' if pct is None else f'{pct:3.0f}%'}] {text}"
            if sys.stdout:  # None in the windowed .exe
                print(line, flush=True)
            log(line)
    site, token = parse_link(link)
    job = Job(site, token, status)
    try:
        job.run()
    except FetchError as e:
        status(f"ERROR: {e}", 0)
        if not isinstance(e, NeedSignIn):
            job.fail(str(e))
        sys.exit(1)


def main():
    args = sys.argv[1:]
    link = next((a for a in args if a.lower().startswith("reelscut:")), None)
    if "--uninstall" in args:
        uninstall()
        return
    try:
        install()
    except Exception as e:
        log(f"install failed: {e}")
    log(f"start {VERSION} link={'yes' if link else 'no'}")
    if "--console" in args and link:
        run_console(link)
    else:
        run_window(link)


if __name__ == "__main__":
    main()
