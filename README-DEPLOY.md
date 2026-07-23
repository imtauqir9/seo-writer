# Deploying the SEO Writer to Fly.io

The app is a single Flask service (`app.py`) that runs the `seo_writer.py` pipeline
as a subprocess and streams progress to the browser. This guide gets it live on Fly.

## Prerequisites

- A [Fly.io account](https://fly.io/app/sign-up) (a card is required even on the free-ish tier).
- Your `ANTHROPIC_API_KEY` (required) and `SERPAPI_KEY` (optional) — already in your local `.env`.

## 1. Install flyctl

**Windows (PowerShell):**
```powershell
iwr https://fly.io/install.ps1 -useb | iex
```
Then restart your terminal so `flyctl` is on PATH (or use `~\.fly\bin\flyctl.exe`).

## 2. Log in
```bash
flyctl auth login
```
This opens a browser to authenticate.

## 3. Create the app

The app name in `fly.toml` (`seo-writer-app`) must be **globally unique**. If it's
taken, pick another name and update the `app = ` line in `fly.toml` to match.

```bash
flyctl apps create seo-writer-app        # or your chosen unique name
```

## 4. Create the persistent volume

Articles are written to `/app/output`, mounted from a volume named `seo_data`
(matching `[[mounts]]` in `fly.toml`). Create it in the same region as `primary_region`:

```bash
flyctl volumes create seo_data --region iad --size 1   # 1 GB
```

## 5. Set secrets

Import both keys straight from your local `.env` (never committed, never in the image):
```bash
flyctl secrets import < .env
```
Or set them individually:
```bash
flyctl secrets set ANTHROPIC_API_KEY=sk-ant-... SERPAPI_KEY=...
```

## 6. Deploy
```bash
flyctl deploy
```

## 7. Open it
```bash
flyctl open        # opens https://<your-app>.fly.dev
```

---

## Notes & tuning

- **Single machine only.** The job store is in-process memory, so the Dockerfile runs
  gunicorn with `--workers 1`. Do **not** scale to multiple machines/workers or the SSE
  progress stream won't find its job. Scale *up* (bigger VM) instead of *out*.
- **Scale to zero.** `auto_stop_machines = "stop"` parks the machine when idle; it wakes
  on the next request (a few seconds of cold start). An in-flight generation keeps the SSE
  connection open, which keeps the machine awake.
- **Cost.** One `shared-cpu-1x` / 1 GB machine that sleeps when idle + a 1 GB volume is
  a few dollars a month at most. Claude API usage is billed separately by Anthropic.
- **Logs:** `flyctl logs`   •   **Status:** `flyctl status`   •   **Redeploy:** `flyctl deploy`
- **Rotate a key:** `flyctl secrets set ANTHROPIC_API_KEY=sk-ant-newvalue` (triggers a restart).
