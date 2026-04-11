# ── Base image ────────────────────────────────────────────────────────────
# Python 3.10 slim keeps the image small while matching the target runtime.
FROM python:3.10-slim

# ── System metadata ────────────────────────────────────────────────────────
LABEL maintainer="your-email@example.com"
LABEL description="AIPlatformEnv: OpenEnv AI-platform interaction benchmark"
LABEL version="0.1.0"

# ── Working directory ──────────────────────────────────────────────────────
# All application files will live under /app inside the container.
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install fastapi uvicorn

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

CMD ["python", "app.py"]