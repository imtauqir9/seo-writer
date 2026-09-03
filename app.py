#!/usr/bin/env python3
"""
SEO Writer — minimal web frontend.

Run:
    uv run --with flask --with anthropic --with requests --with markdown --with python-docx app.py
    # or
    pip install flask && python app.py

Then open http://localhost:5000
"""

import hashlib
import hmac
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from flask import (Flask, Response, jsonify, redirect, render_template, request,
                   send_from_directory, session, url_for)

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


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------
#
# POST /api/start spends real money - roughly fifteen model calls across three
# vendors, two of them high-effort. Deployed without a password that endpoint is
# an open faucet on someone else's card, so set APP_PASSWORD anywhere the app is
# reachable from the internet.

APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
APP_USERNAME = os.environ.get("APP_USERNAME", "admin")

# A browser gets a real login form and a session cookie; scripts and curl keep
# working with basic auth against the same password. Either satisfies the gate.
#
# The signing key defaults to something derived from the password, so sessions
# survive a restart without a second secret to manage - and changing the
# password signs everyone out, which is what you want from a password change.
SECRET_KEY = os.environ.get("SECRET_KEY", "")
app.secret_key = SECRET_KEY or hashlib.sha256(
    ("humanly-session-v1:" + APP_PASSWORD).encode()
).hexdigest()

# Fly sets FLY_APP_NAME, and Fly is always HTTPS. Locally the app is plain HTTP,
# where a Secure cookie would never be sent back and the login would loop.
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=bool(os.environ.get("FLY_APP_NAME")),
)

# Public because a browser must reach them before it can authenticate, and
# because a health check should not need a credential.
_OPEN_PATHS = {"/healthz", "/login"}

# A login form on a public URL is a brute-force target. This is deliberately
# small: a per-IP counter, not a rate-limiting library.
_LOGIN_MAX_ATTEMPTS = 8
_LOGIN_LOCKOUT_SECS = 300
_login_failures: dict[str, list] = {}
_login_lock = threading.Lock()


def _client_ip() -> str:
    fwd = request.headers.get("Fly-Client-IP") or request.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() or request.remote_addr or "unknown")


def _locked_out(ip: str) -> int:
    """Seconds remaining on a lockout, or 0."""
    with _login_lock:
        hits = [t for t in _login_failures.get(ip, [])
                if time.time() - t < _LOGIN_LOCKOUT_SECS]
        _login_failures[ip] = hits
        if len(hits) >= _LOGIN_MAX_ATTEMPTS:
            return int(_LOGIN_LOCKOUT_SECS - (time.time() - hits[0])) + 1
    return 0


def _record_failure(ip: str):
    with _login_lock:
        _login_failures.setdefault(ip, []).append(time.time())


def _clear_failures(ip: str):
    with _login_lock:
        _login_failures.pop(ip, None)


def _password_ok(candidate: str) -> bool:
    return hmac.compare_digest(candidate or "", APP_PASSWORD)


def _basic_auth_ok() -> bool:
    auth = request.authorization
    if not auth or auth.type != "basic":
        return False
    # compare_digest on both halves, so neither the username nor the password
    # leaks its length through response timing.
    return (hmac.compare_digest(auth.username or "", APP_USERNAME)
            and _password_ok(auth.password or ""))


def _logged_in() -> bool:
    return session.get("auth") is True


@app.before_request
def require_password():
    if not APP_PASSWORD or request.path in _OPEN_PATHS:
        return None
    if _logged_in() or _basic_auth_ok():
        return None
    # An API caller wants a 401 it can handle, not an HTML login page.
    if request.path.startswith("/api/"):
        return Response(
            "Authentication required.\n", 401,
            {"WWW-Authenticate": 'Basic realm="Humanly", charset="UTF-8"'},
        )
    return redirect(url_for("login", next=request.full_path.rstrip("?")))


@app.route("/login", methods=["GET", "POST"])
def login():
    if not APP_PASSWORD:
        return redirect(url_for("index"))
    if _logged_in():
        return redirect(url_for("index"))

    target = request.args.get("next") or request.form.get("next") or "/"
    # Only ever bounce to a path on this app, never to another host.
    if not target.startswith("/") or target.startswith("//"):
        target = "/"

    error = None
    if request.method == "POST":
        ip = _client_ip()
        wait = _locked_out(ip)
        if wait:
            error = f"Too many attempts. Try again in {wait} seconds."
        elif _password_ok(request.form.get("password", "")):
            _clear_failures(ip)
            session.clear()
            session["auth"] = True
            session.permanent = False
            return redirect(target)
        else:
            _record_failure(ip)
            error = "That password is not right."

    return render_template("login.html", error=error, next=target), (
        200 if error is None else 401
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "protected": bool(APP_PASSWORD)})


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
        linkedin_path = OUTPUT_DIR / f"{slug}_linkedin.md"
        video_path = OUTPUT_DIR / f"{slug}_video.md"
        review_path = OUTPUT_DIR / f"{slug}_review.json"

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
            "linkedin_file": linkedin_path.name if linkedin_path.exists() else None,
            "video_file": video_path.name if video_path.exists() else None,
            "has_review": review_path.exists(),
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


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------

def read_usage(limit: int = 500) -> list[dict]:
    """Every run the pipeline has logged, newest first."""
    path = OUTPUT_DIR / "usage.jsonl"
    if not path.exists():
        return []
    runs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            runs.append(json.loads(line))
        except ValueError:
            continue          # a torn write should not blank the whole dashboard
    return list(reversed(runs))[:limit]


def usage_rollup(runs: list[dict]) -> dict:
    """Totals overall, per model, and per day."""
    total = {"runs": len(runs), "calls": 0, "input_tokens": 0,
             "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0,
             "fully_priced": True}
    by_model, by_day = defaultdict(lambda: {"calls": 0, "input_tokens": 0,
                                            "output_tokens": 0, "cost_usd": 0.0}), \
        defaultdict(lambda: {"runs": 0, "total_tokens": 0, "cost_usd": 0.0})

    for r in runs:
        total["calls"] += r.get("calls", 0)
        total["input_tokens"] += r.get("input_tokens", 0)
        total["output_tokens"] += r.get("output_tokens", 0)
        total["total_tokens"] += r.get("total_tokens", 0)
        total["cost_usd"] += r.get("cost_usd", 0.0)
        if not r.get("fully_priced", True):
            total["fully_priced"] = False

        day = (r.get("at") or "")[:10] or "unknown"
        by_day[day]["runs"] += 1
        by_day[day]["total_tokens"] += r.get("total_tokens", 0)
        by_day[day]["cost_usd"] += r.get("cost_usd", 0.0)

        for model, m in (r.get("by_model") or {}).items():
            bucket = by_model[model]
            bucket["calls"] += m.get("calls", 0)
            bucket["input_tokens"] += m.get("input_tokens", 0)
            bucket["output_tokens"] += m.get("output_tokens", 0)
            bucket["cost_usd"] += m.get("cost_usd", 0.0)

    runs_n = max(total["runs"], 1)
    total["avg_tokens_per_run"] = round(total["total_tokens"] / runs_n)
    total["avg_cost_per_run"] = round(total["cost_usd"] / runs_n, 3)
    return {
        "total": total,
        "by_model": dict(sorted(by_model.items(),
                                key=lambda kv: -kv[1]["cost_usd"])),
        "by_day": dict(sorted(by_day.items(), reverse=True)),
    }


@app.route("/api/usage")
def api_usage():
    runs = read_usage()
    return jsonify({**usage_rollup(runs), "runs": runs})


@app.route("/usage")
def usage_dashboard():
    runs = read_usage()
    roll = usage_rollup(runs)
    return render_template("usage.html", runs=runs, **roll)


@app.route("/api/start", methods=["POST"])
def api_start():
    """
    Start a pipeline job. Returns { job_id }.
    Frontend then opens EventSource on /api/stream/<job_id>.
    """
    data = request.get_json(silent=True) or {}
    topic = (data.get("topic") or "").strip()
    intent = (data.get("intent") or "").strip()
    try:
        edition = int(data.get("edition") or 0)
    except (TypeError, ValueError):
        return jsonify({"error": "edition must be a number"}), 400
    words = str(data.get("words") or "default")
    if words not in {"default", "1000", "2000"}:
        words = "default"
    linkedin = bool(data.get("linkedin"))
    video = bool(data.get("video"))

    if not topic:
        return jsonify({"error": "topic is required"}), 400

    cmd = [
        sys.executable, str(BASE_DIR / "seo_writer.py"),
        topic,
        "--output-dir", str(OUTPUT_DIR),
        "--edition", str(edition),
        "--words", words,
    ]
    if intent:
        cmd += ["--intent", intent]
    if linkedin:
        cmd.append("--linkedin")
    if video:
        cmd.append("--video")

    return jsonify({"job_id": _spawn(cmd)})


def _spawn(cmd: list[str], review_baseline: set | None = None) -> str:
    """Run seo_writer.py in the background, streaming its output to a job queue.

    Shared by generation and audit: both are the same pipeline script with
    different flags, and both want the same live log.
    """
    job_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()
    with _jobs_lock:
        _jobs[job_id] = q

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
            tail = deque(maxlen=40)
            for line in proc.stdout:
                line = line.rstrip()
                # seo_writer.py prefixes fatal, already-human-readable failures
                # with "ERROR:" — prefer that over a bare exit code.
                if line.startswith("ERROR:"):
                    last_error = line[len("ERROR:"):].strip()
                tail.append(line)
                q.put(("log", line))
            proc.wait()
            if proc.returncode != 0:
                # The child's output only ever went to the browser, so a crash
                # vanished when the tab closed and the server log showed nothing.
                # Echo the tail so it survives in `flyctl logs`.
                print(f"[job] pipeline exited {proc.returncode}; last output:",
                      flush=True)
                for line in tail:
                    print(f"[job]   {line}", flush=True)
                q.put(("error", last_error
                       or f"Pipeline exited with code {proc.returncode}. "
                          f"The server log has the last 40 lines."))
                return

            payload = {}
            if review_baseline is not None:
                # Whichever report is new is this job's. Identifying it by
                # difference beats predicting the slug the pipeline will pick.
                fresh = [p for p in OUTPUT_DIR.glob("*_review.json")
                         if p.name not in review_baseline]
                if fresh:
                    newest = max(fresh, key=lambda p: p.stat().st_mtime)
                    payload["review_slug"] = newest.stem[:-len("_review")]
            q.put(("done", json.dumps(payload)))
        except Exception as e:
            q.put(("error", str(e)))

    threading.Thread(target=run, daemon=True).start()
    return job_id


# ---------------------------------------------------------------------------
# Evaluate a draft you already have
# ---------------------------------------------------------------------------

UPLOAD_DIR = OUTPUT_DIR / "uploads"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024          # a 2 MB article is already enormous
ALLOWED_SUFFIXES = {".md", ".markdown", ".txt", ".docx"}


def _docx_to_markdown(raw: bytes) -> str:
    """Flatten a .docx to text, keeping heading levels so the auditor sees structure."""
    import io
    try:
        from docx import Document
    except ImportError:
        raise ValueError("Reading .docx needs python-docx. Paste the text instead.")
    doc = Document(io.BytesIO(raw))
    lines = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            lines.append("")
            continue
        style = (p.style.name or "").lower()
        if style.startswith("heading"):
            level = "".join(c for c in style if c.isdigit()) or "2"
            lines.append("#" * min(int(level), 6) + " " + text)
        else:
            lines.append(text)
    return "\n".join(lines).strip()


def _draft_from_request() -> tuple[str, str]:
    """The document to audit, as (markdown, source name). Raises ValueError."""
    upload = request.files.get("file")
    if upload and upload.filename:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise ValueError(
                f"{suffix or 'That file type'} is not supported. "
                f"Upload {', '.join(sorted(ALLOWED_SUFFIXES))}, or paste the text."
            )
        raw = upload.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise ValueError("That file is over 2 MB. Trim it or paste the article body.")
        if suffix == ".docx":
            return _docx_to_markdown(raw), Path(upload.filename).stem
        try:
            return raw.decode("utf-8").strip(), Path(upload.filename).stem
        except UnicodeDecodeError:
            raise ValueError("That file is not UTF-8 text. Save it as .md and retry.")

    pasted = (request.form.get("text") or "").strip()
    if pasted:
        return pasted, ""
    raise ValueError("Paste an article or choose a file to evaluate.")


@app.route("/api/audit/start", methods=["POST"])
def api_audit_start():
    """Run the three agents over a document the user supplied. Returns { job_id }."""
    try:
        draft, filename = _draft_from_request()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if len(draft.split()) < 50:
        return jsonify({"error": "That draft is too short to audit meaningfully "
                                 "(under 50 words)."}), 400

    topic = (request.form.get("topic") or "").strip() or filename
    intent = (request.form.get("intent") or "").strip()
    try:
        rounds = max(1, min(int(request.form.get("rounds") or 2), 4))
    except ValueError:
        rounds = 2
    apply_fixes = (request.form.get("apply") or "").lower() in {"1", "true", "on", "yes"}

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    draft_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.md"
    draft_path.write_text(draft, encoding="utf-8")

    # Snapshot what already exists so the new report can be identified afterwards,
    # rather than guessing at the slug the pipeline will choose.
    before = {p.name for p in OUTPUT_DIR.glob("*_review.json")}

    cmd = [
        sys.executable, str(BASE_DIR / "seo_writer.py"),
        "--audit", str(draft_path),
        "--output-dir", str(OUTPUT_DIR),
        "--verify-rounds", str(rounds),
    ]
    if topic:
        cmd.append(topic)
    if intent:
        cmd += ["--intent", intent]
    if apply_fixes:
        cmd.append("--apply")

    return jsonify({"job_id": _spawn(cmd, review_baseline=before)})


@app.route("/api/reviews")
def api_reviews():
    return jsonify(list_reviews())


def list_reviews() -> list[dict]:
    """Every verification report on disk, newest first."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = []
    for path in sorted(OUTPUT_DIR.glob("*_review.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True):
        slug = path.stem[:-len("_review")]
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rounds = record.get("rounds") or []
        raised = sum(len(r.get("issues") or []) for r in rounds)
        applied = sum(len(r.get("applied") or []) for r in rounds)
        out.append({
            "slug": slug,
            "outcome": record.get("outcome", ""),
            "rounds": len(rounds),
            "raised": raised,
            "applied": applied,
            "at": record.get("finished_at") or record.get("started_at") or "",
            "agents": {k: v.get("model") for k, v in (record.get("agents") or {}).items()},
        })
    return out


@app.route("/review/<slug>")
def review_page(slug):
    path = OUTPUT_DIR / f"{Path(slug).name}_review.json"
    if not path.exists():
        return render_template("review.html", record=None, slug=slug), 404
    record = json.loads(path.read_text(encoding="utf-8"))
    rounds = record.get("rounds") or []
    counts = {
        "rounds": len(rounds),
        "raised": sum(len(r.get("issues") or []) for r in rounds),
        "applied": sum(len(r.get("applied") or []) for r in rounds),
        "disputed": sum(1 for r in rounds
                        for x in (r.get("responses") or [])
                        if x.get("stance") == "dispute"),
    }
    return render_template("review.html", record=record, slug=slug, counts=counts)


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
                    yield f"event: done\ndata: {msg or '{}'}\n\n"
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
