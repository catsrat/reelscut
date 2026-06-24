# Clip Reels worker — Linux container for cloud hosting (Render/Railway/Fly).
# NOTE: not yet test-built (no Docker on the dev Mac) — expect to debug on first
# deploy. Non-English transcription uses the Sarvam API in the cloud, so only the
# small English Whisper model is baked in.
FROM python:3.11-slim

# System deps: ffmpeg (video), libraqm (Pillow Indic shaping), build tools +
# git/cmake to compile whisper.cpp.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git cmake build-essential pkg-config curl ca-certificates \
        libjpeg-dev zlib1g-dev libfreetype6-dev \
        libraqm-dev libharfbuzz-dev libfribidi-dev \
    && rm -rf /var/lib/apt/lists/*

# Build whisper.cpp -> /usr/local/bin/whisper-cli (static = no separate .so files)
RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp /tmp/whisper \
    && cmake -S /tmp/whisper -B /tmp/whisper/build \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    && cmake --build /tmp/whisper/build -j --config Release \
    && cp /tmp/whisper/build/bin/whisper-cli /usr/local/bin/whisper-cli \
    && rm -rf /tmp/whisper

WORKDIR /app

# Python deps (install Pillow from source so it links libraqm for Indic shaping).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-binary Pillow --force-reinstall Pillow \
    && pip install --no-cache-dir -U "yt-dlp[default]"  # latest yt-dlp for YouTube/SABR fixes

# App code (fonts/, music/, broll/, templates/, *.py)
COPY . .

# English Whisper model (small, ~142MB). Indian languages use Sarvam (cloud).
RUN mkdir -p models \
    && curl -sL -o models/ggml-base.en.bin \
       https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

# Hugging Face Spaces routes to 7860; Render injects its own $PORT (which wins).
ENV PORT=7860
EXPOSE 7860

# Shell form so $PORT expands. Single worker: job state is in memory, one at a time.
CMD gunicorn -w 1 --threads 8 --timeout 0 -b 0.0.0.0:$PORT app:app
