FROM python:3.12-slim

# Faster, cleaner Python in containers
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY seo_writer.py app.py ./
COPY templates ./templates

# Article output lives here; mounted as a Fly volume for persistence
RUN mkdir -p /app/output

EXPOSE 8080

# 1 worker only: the job store is in-process memory, so a request that starts a
# job and the later SSE stream request must hit the same worker.
# gthread + timeout 0 keeps long Server-Sent Events streams alive during generation.
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--worker-class", "gthread", \
     "--timeout", "0", "--bind", "0.0.0.0:8080", "app:app"]
