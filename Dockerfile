# Clip Reels worker — Linux container for cloud hosting (Render/Railway/Fly).
# NOTE: not yet test-built (no Docker on the dev Mac) — expect to debug on first
# deploy. Non-English transcription uses the Sarvam API in the cloud, so only the
# small English Whisper model is baked in.
FROM python:3.11-slim

# System deps: ffmpeg (video), libraqm (Pillow Indic shaping), build tools +
# git/cmake to compile whisper.cpp.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libraqm0 libraqm-dev libfreetype6 libharfbuzz0b libfribidi0 \
        git cmake build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build whisper.cpp -> /usr/local/bin/whisper-cli
RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp /tmp/whisper \
    && cmake -S /tmp/whisper -B /tmp/whisper/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/whisper/build -j --config Release \
    && cp /tmp/whisper/build/bin/whisper-cli /usr/local/bin/whisper-cli \
    && rm -rf /tmp/whisper

WORKDIR /app

# Python deps (install Pillow from source so it links libraqm for Indic shaping).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-binary Pillow --force-reinstall Pillow

# App code (fonts/, music/, broll/, templates/, *.py)
COPY . .

# English Whisper model (small, ~142MB). Indian languages use Sarvam (cloud).
RUN mkdir -p models \
    && curl -sL -o models/ggml-base.en.bin \
       https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

ENV PORT=8080
EXPOSE 8080

# Single worker: the app keeps job state in memory and runs one job at a time.
CMD ["gunicorn", "-w", "1", "--threads", "8", "--timeout", "0", \
     "-b", "0.0.0.0:8080", "app:app"]
