# Background music (mood-matched)

When "Add background music" is checked, Claude tags each clip with a **mood**
and the app picks a track from the matching folder below. Drop royalty-free
tracks (`.mp3`, `.m4a`, `.wav`) into the folders that fit:

```
music/
  upbeat/         energetic, happy, high-energy
  calm/           soft, chill, ambient
  dramatic/       tense, cinematic, intense
  inspirational/  uplifting, motivational
  funny/          quirky, playful, comedic
```

How selection works:
1. Clip mood → look in `music/<mood>/` and pick a random track there.
2. If that folder is empty → fall back to any track anywhere under `music/`
   (including the sample pad), so it always plays something.

`sample_ambient_pad.mp3` (in this folder) is a synthesized placeholder so the
feature works before you add tracks. **Replace it with real royalty-free
music** for anything you publish:
- YouTube Audio Library (free, in YouTube Studio)
- Pixabay Music (pixabay.com/music)
- Uppbeat (uppbeat.io)

⚠️ Never use copyrighted songs — they trigger copyright strikes / muting on
TikTok, Reels, and YouTube.
