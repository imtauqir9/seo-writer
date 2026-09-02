#!/usr/bin/env python3
"""
SEO Writer — minimal web frontend.

Run:
    uv run --with flask --with anthropic --with requests --with markdown --with python-docx app.py
    # or
    pip install flask && python app.py

Then open http://localhost:5000
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

app = Flask(__name__)
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dotenv():
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

_load_dotenv()

# A single high-effort audit call on a 3,500-word article can run for minutes
# with nothing to print. Emit a comment frame while waiting so neither the
# browser nor Fly's edge proxy closes the connection, and only fail on a
# genuinely dead pipeline.
HEARTBEAT_SECS = 15
SILENCE_LIMIT_SECS = 900

# In-memory job store: job_id → queue.Queue
_jobs: dict[str, queue.Queue] = {}
_jobs_lock = threading.Lock()


def list_articles() -> list[dict]:
    """Return metadata for every generated article, newest first."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    articles = []
    for meta_file in sorted(OUTPUT_DIR.glob("*_meta.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        slug = meta_file.stem.replace("_meta", "")
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            meta = {}

        seo = meta.get("seo_meta", {})
        md_path = OUTPUT_DIR / f"{slug}.md"

        # Glob-based matching: find any html/docx starting with this slug
        # (handles old files saved under different names)
        html_files = sorted(OUTPUT_DIR.glob(f"{slug}*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
        docx_files = sorted(OUTPUT_DIR.glob(f"{slug}*.docx"), key=lambda p: p.stat().st_mtime, reverse=True)
        html_file = html_files[0].name if html_files else None
        docx_file = docx_files[0].name if docx_files else None

        word_count = 0
        if md_path.exists():
            word_count = len(md_path.read_text().split())

        generated_at = meta.get("generated_at", "")
        if generated_at:
            try:
                dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
                generated_at = dt.strftime("%b %d, %Y")
            except Exception:
                pass

        articles.append({
            "slug": slug,
            "title": seo.get("title", slug.replace("-", " ").title()),
            "description": seo.get("description", ""),
            "generated_at": generated_at,
            "word_count": word_count,
            "html_file": html_file,
            "docx_file": docx_file,
            "image_count": len(meta.get("images", [])),
        })
    return articles


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    articles = list_articles()
    return render_template("index.html", articles=articles)


@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/api/articles")
def api_articles():
    return jsonify(list_articles())


@app.route("/api/start", methods=["POST"])
def api_start():
    """
    Start a pipeline job. Returns { job_id }.
    Frontend then opens EventSource on /api/stream/<job_id>.
    """
    data = request.get_json(silent=True) or {}
    topic = (data.get("topic") or "").strip()
    intent = (data.get("intent") or "").strip()
    edition = int(data.get("edition") or 0)

    if not topic:
        return jsonify({"error": "topic is required"}), 400

    job_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()
    with _jobs_lock:
        _jobs[job_id] = q

    cmd = [
        sys.executable, str(BASE_DIR / "seo_writer.py"),
        topic,
        "--output-dir", str(OUTPUT_DIR),
        "--edition", str(edition),
    ]
    if intent:
        cmd += ["--intent", intent]

    def run():
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ},
            )
            last_error = ""
            for line in proc.stdout:
                line = line.rstrip()
                # seo_writer.py prefixes fatal, already-human-readable failures
                # with "ERROR:" — prefer that over a bare exit code.
                if line.startswith("ERROR:"):
                    last_error = line[len("ERROR:"):].strip()
                q.put(("log", line))
            proc.wait()
            if proc.returncode == 0:
                q.put(("done", ""))
            else:
                q.put(("error", last_error or f"Pipeline exited with code {proc.returncode}"))
        except Exception as e:
            q.put(("error", str(e)))

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/stream/<job_id>")
def api_stream(job_id):
    """EventSource endpoint — streams log lines then a done/error event."""
    with _jobs_lock:
        q = _jobs.get(job_id)
    if q is None:
        return jsonify({"error": "job not found"}), 404

    def stream():
        silent = 0
        try:
            while True:
                try:
                    kind, msg = q.get(timeout=HEARTBEAT_SECS)
                except queue.Empty:
                    silent += HEARTBEAT_SECS
                    if silent >= SILENCE_LIMIT_SECS:
                        minutes = SILENCE_LIMIT_SECS // 60
                        yield (
                            "event: error\ndata: "
                            + json.dumps({"message": f"The pipeline stopped "
                                                     f"responding after {minutes} "
                                                     f"minutes."})
                            + "\n\n"
                        )
                        break
                    yield ": keepalive\n\n"
                    continue

                silent = 0
                if kind == "log":
                    yield f"event: log\ndata: {json.dumps({'line': msg})}\n\n"
                elif kind == "done":
                    yield f"event: done\ndata: {{}}\n\n"
                    break
                elif kind == "error":
                    yield f"event: error\ndata: {json.dumps({'message': msg})}\n\n"
                    break
        finally:
            with _jobs_lock:
                _jobs.pop(job_id, None)

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting SEO Writer UI at http://localhost:{port}")
    app.run(debug=True, port=port, threaded=True)
