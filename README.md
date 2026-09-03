# SEO Article Agent

An AI agent that writes publication-ready SEO articles. You give it a topic and a goal — it does the research, finds images, and produces a finished article in Markdown, HTML, and DOCX. It also learns from your existing articles, so the output matches your voice and style rather than sounding generic.

The pipeline runs in 10 stages: live SERP research, title refinement, key takeaways, outline, writing, a three-pass humanization stage that strips AI patterns, a direct-answer block for answer engines, an adversarial verification stage where a second model audits the draft and the writer gets to argue back, meta description, and image sourcing. Everything is automated.

---

## What it does

- **Deep research first** — pulls live Google search results via SerpAPI before writing a single word. The agent reads what's ranking, identifies gaps, and builds the article around those findings.
- **Learns from your sample articles** — drop your best articles into `sample-articles/` and the agent uses them as style references. The output reads like you wrote it, not like ChatGPT.
- **Finds real images** — searches Google Images (with SerpAPI) or Unsplash and embeds them directly into the DOCX. No placeholder images.
- **Humanization pass built in** — after writing, the agent rewrites the draft to remove AI patterns before you ever see it. The last pass is measured, not vibed: a scanner counts the tells that survived and hands the model the exact phrase list to repair.
- **The draft has to survive an argument** — a second model audits the finished article against the brief it was written from. The writer can dispute findings it thinks are wrong, a third model settles what stays contested, and only what survives gets applied.
- **Three roles, three sets of weights** — writer, auditor and judge are pushed onto the most distinct vendors your keys allow. The auditor moves off the writer's vendor first, because a blind spot shared between those two means the finding is never raised at all.
- **Ships SEO and GEO metadata, not just prose** — the HTML carries a real head (title, description, canonical, Open Graph, Twitter cards) plus `Article` and `FAQPage` JSON-LD built from the article's own FAQ. Citations render with a publisher name and access date, and a short extractable answer sits under the H1 for answer engines to quote.

---

## Pipeline

```
Topic + Intent
      │
      ▼
1. SERP Research      — pulls live Google results via SerpAPI, reads what's ranking
      │
      ▼
2. Title Refinement   — picks the best angle and target keyword
      │
      ▼
3. Key Takeaways      — identifies what the article must cover to outrank competitors
      │
      ▼
4. Outline            — structures the article before writing begins
      │
      ▼
5. Write              — full draft grounded in research and your sample articles
      │
      ▼
6. Humanize           — strips AI patterns, rewrites to match your voice
      │
      ├─► pass 1  pattern removal against a 24-item checklist
      ├─► pass 2  the model self-audits its own draft
      └─► pass 3  a scanner counts what actually survived — banned vocabulary,
      │           title-cased headings, inline-header bullets, em dashes,
      │           uniform sentence length — and the model repairs the named
      │           phrases only. If the count goes up, pass 2 is kept.
      ▼
6.4 Answer block      — a 40–60 word extractable answer placed under the H1,
      │                 written to survive being quoted on its own
      ▼
6.5 Verify            — a second model audits the draft against the brief
      │
      ├─► audit    a different vendor scores factual support, outline fidelity,
      │            voice and coverage, returning findings anchored to quotes
      ├─► rebuttal the writer accepts or disputes each finding
      │            (silence counts as acceptance)
      ├─► judge    a third model rules on what is still contested
      └─► fix      only accepted and upheld findings are applied,
      │            then re-audit — up to 2 rounds
      ▼
7. Meta               — generates SEO title, meta description, and slug
      │
      ▼
8. Images             — finds real images via Google Images or Unsplash, embeds in DOCX
      │
      ▼
Output: .md  .html  .docx  _meta.json
```

---

## Setup

### 1. Clone and configure

```bash
git clone <repo-url>
cd SEO-writer
cp .env.example .env
```

Open `.env` and add your keys:

```ini
ANTHROPIC_API_KEY=sk-ant-...   # required
SERPAPI_KEY=...                 # optional but recommended

# Step 6.5 roster. Each extra key buys a more independent audit.
OPENAI_API_KEY=sk-proj-...      # optional; moves the auditor off Anthropic
GEMINI_API_KEY=...              # optional; moves the judge off both
AUDITOR_PROVIDER=auto           # auto | anthropic | openai | gemini
JUDGE_PROVIDER=auto
OPENAI_JUDGE_MODEL=gpt-5.5
GEMINI_MODEL=gemini-2.5-pro

# Publication identity — canonical URL, og:url, Article/Person schema.
SITE_URL=https://yoursite.com/blog   # blank omits canonical rather than guessing
AUTHOR_NAME=Imran Tauqir
AUTHOR_URL=https://imrantauqir.com/
```

Which roles run where, by the keys you have:

| Keys present | Writer | Auditor | Judge |
|---|---|---|---|
| Anthropic only | `claude-sonnet-5` | `claude-opus-5` | `claude-opus-5` |
| Anthropic + OpenAI | `claude-sonnet-5` | `gpt-5.5` | `claude-opus-5` |
| Anthropic + OpenAI + Gemini | `claude-sonnet-5` | `gpt-5.5` | `gemini-2.5-pro` |

The writer always stays on Anthropic — the style prompts and sample-article matching were tuned against it, so swapping it changes the product rather than checking it. If a vendor is down mid-run, that role falls back to Claude and says so, because a vendor outage must not destroy an article that already cost a dozen calls.

### 2. Install dependencies

With `uv` (no setup needed):
```bash
uv run --with anthropic --with requests --with flask --with markdown --with python-docx app.py
```

With pip:
```bash
pip install -r requirements.txt
python app.py
```

Then open [http://localhost:8080](http://localhost:8080).

---

## CLI

```bash
# Basic
python seo_writer.py "What is RAG in AI"

# With intent — the agent figures out the right angle and keywords
python seo_writer.py "ReAct Agents" \
  --intent "explain to developers how ReAct agents think step by step"

# Custom output folder and edition number
python seo_writer.py "Semantic Caching for LLMs" --output-dir ./articles --edition 31
```

| Flag | Description |
|------|-------------|
| `topic` | What to write about (required) |
| `--intent` | What you want readers to take away — agent uses this to pick keywords and angle |
| `--keywords` | Explicit keywords if you already know what to target |
| `--output-dir` | Where to save output (default: `./output`) |
| `--edition` | Newsletter edition number |
| `--no-verify` | Skip the step 6.5 audit — faster and cheaper, but nothing checks the article's claims or structure before it hits disk |
| `--verify-rounds` | Maximum audit/fix rounds before accepting the article (default: `2`) |
| `--audit FILE` | Audit a document you already have instead of writing a new one |
| `--apply` | With `--audit`, also save the revised document |

---

## Password protection

`POST /api/start` spends real money — roughly fifteen model calls across three
vendors, two of them at high effort. Deployed without a password it is an open
faucet on your card, so set one anywhere the app is reachable from the internet:

```bash
APP_PASSWORD=something-long     # in .env, or as a Fly secret
APP_USERNAME=admin              # optional, defaults to admin
```

Every route is then behind HTTP basic auth, including the JSON APIs. Two
exceptions by design: `/healthz` stays open so a health check doesn't need a
credential, and leaving `APP_PASSWORD` blank leaves the app open — which is fine
on localhost and is not fine on a public URL.

```bash
flyctl secrets set APP_PASSWORD='something-long' -a your-app
```

`GET /healthz` reports `{"ok": true, "protected": true|false}`, so you can check
from the outside whether the deployed app is actually locked.

---

## Token usage

Every run appends a line to `output/usage.jsonl` and writes a per-article
`<slug>_usage.json`. **`/usage`** renders it: totals, cost per article, a
per-model breakdown and a per-day history.

Token counts come from what each API actually reports — `usage.input_tokens` and
`usage.output_tokens` on Anthropic, `prompt_tokens`/`completion_tokens` on
OpenAI, `usage_metadata` on Gemini. They are counted, never estimated.

Cost is a softer layer. The built-in table carries Anthropic's first-party list
prices (`claude-sonnet-5` $2/$10 per MTok, `claude-opus-5` $5/$25). **A model
with no price on file still has its tokens counted and simply reports no dollar
figure** rather than a wrong one — the dashboard flags those runs and treats the
total as a floor. Add your own rates rather than trusting a guess:

```bash
MODEL_PRICES={"gpt-5.5": [1.25, 10.0], "gemini-2.5-pro": [1.25, 10.0]}
```

The CLI prints the same summary at the end of every run:

```
  Tokens: 184,203 in + 27,410 out = 211,613 across 17 calls
    claude-sonnet-5       11 calls   120,400 in   19,900 out  $0.440
    claude-opus-5          4 calls    48,100 in    6,200 out  $0.396
    gpt-5.5                2 calls    15,703 in    1,310 out  unpriced
  Estimated cost: $0.84 plus unpriced models
```

---

## SEO and GEO

The `.html` export is meant to be pasted into a CMS as-is, so the metadata the
pipeline computes actually ships with it rather than sitting in a sidecar JSON
file no crawler will read.

**In the head:** `<title>`, meta description, `lang`, viewport, author, canonical
(only when `SITE_URL` is set — a wrong canonical is worse than none), Open Graph,
and Twitter card tags.

**Structured data:** an `Article` node with `datePublished`, `wordCount`, author
and publisher as a `Person`, the article's images, and every source as a named
`citation`. When the article has a FAQ section, a `FAQPage` node is built from it
— which is why the outline requires FAQ questions to be H3s ending in a question
mark, and answers that stand on their own. Anything that isn't a real question is
skipped rather than shipped as malformed data.

**For answer engines specifically:**

| | What it does |
|---|---|
| Answer block | 40–60 words directly under the H1, written to make sense quoted with no surrounding page. Generated *before* verification, so the auditor fact-checks it like any other passage. |
| Named citations | `IDC — https://idc.com (accessed September 02, 2026)` rather than a bare URL. Attributable citations are weighted; naked links aren't. |
| Comparison table | The outline requires one where the topic genuinely has something to compare. Tables are the passage most often quoted whole. |
| FAQ pairs | Question-shaped headings match how people actually query an assistant. |

Set `SITE_URL` before publishing. Without it the canonical, `og:url` and schema
`mainEntityOfPage` are all omitted.

---

## Auditing your own writing

The three agents don't only work on articles this tool wrote. Point them at any
document and they'll argue about it.

**In the browser:** open the app, switch to the **Evaluate a draft** tab, and
either upload a `.md`/`.txt`/`.docx` or paste the article in. The live log runs
as it does for generation, and the finished report opens as a page showing every
finding with all three voices on it. Reports stay available at `/review/<slug>`.

**From the CLI:**

```bash
# Analyse a document you already have — no article, images or meta are generated
python seo_writer.py --audit ./drafts/my-post.md

# Name the topic so the report and filenames read properly, and tell the auditor
# what you were going for
python seo_writer.py "Semantic caching for LLMs" \
  --audit ./drafts/my-post.md \
  --intent "convince platform engineers this is worth the infra cost"

# Also write out the corrected version
python seo_writer.py --audit ./drafts/my-post.md --apply
```

The auditor works by comparing an article against the brief it was written from,
and a file you hand it has no brief. So one call first reconstructs the brief the
document *appears* to be written to — its own heading structure and the points it
promises the reader — and the agents argue against that. `--intent` feeds your
actual goal into that reconstruction, which makes the coverage findings sharper.

You get back `<slug>_review.md`: the roster, the per-round scores, and every
finding with all three voices on it.

```
### i3 - factual (high)

> cuts p99 latency by roughly 40%

**Auditor:** No citation, and the figure is presented as general fact. _Wants:_ Attribute it or cut it.
**Writer:** disputed. The source is cited two paragraphs down.
**Judge:** uphold. The citation two paragraphs down covers a different claim.

**Result:** applied
```

---

## Output

Each run produces these files in `./output/`:

| File | What it is |
|------|------------|
| `<slug>.md` | Full article in Markdown |
| `<slug>.html` | Styled HTML, ready to copy into a CMS |
| `<slug>.docx` | Word document with embedded images |
| `<slug>_meta.json` | SEO title, meta description, slug, image URLs |
| `<slug>_review.md` | What the three agents argued about, and what survived |
| `<slug>_review.json` | The same argument as raw data |
| `<slug>_usage.json` | Tokens and cost for this run, per model and per step |
| `usage.jsonl` | One line per run — the rolling log behind `/usage` |

An `--audit` run writes only the two review files, plus `<slug>_revised.md` if you
passed `--apply`.

---

## Project structure

```
SEO-writer/
├── seo_writer.py       # Core agent + CLI
├── app.py              # Web UI (Flask)
├── requirements.txt
├── .env.example        # Copy to .env and fill in keys
├── templates/
│   └── index.html
├── sample-articles/    # Your reference articles — agent uses these for style
└── n8n/                # Original n8n workflow this was built from
```

---

## Keys

| Variable | Required | Notes |
|----------|----------|-------|
| `ANTHROPIC_API_KEY` | Yes | [console.anthropic.com](https://console.anthropic.com/) |
| `SERPAPI_KEY` | No | [serpapi.com](https://serpapi.com/) — enables live research and real images |
| `SITE_URL` | No | Your blog's base URL. Enables canonical, `og:url` and schema `mainEntityOfPage` |
| `AUTHOR_NAME` | No | Author in the head and the Article/Person schema |
| `AUTHOR_URL` | No | Author profile URL used in the schema |
| `OPENAI_API_KEY` | No | [platform.openai.com](https://platform.openai.com/api-keys) — moves the step 6.5 auditor off Anthropic |
| `GEMINI_API_KEY` | No | [aistudio.google.com](https://aistudio.google.com/apikey) — moves the judge off both. Needs `pip install google-genai` |
| `AUDITOR_PROVIDER` | No | `auto` \| `anthropic` \| `openai` \| `gemini` |
| `JUDGE_PROVIDER` | No | Same values as above |
| `OPENAI_JUDGE_MODEL` | No | Default `gpt-5.5` — set to a model your key can reach |
| `GEMINI_MODEL` | No | Default `gemini-2.5-pro` — set to a model your key can reach |
| `APP_PASSWORD` | No | **Set it on any public deployment.** Enables HTTP basic auth on every route |
| `APP_USERNAME` | No | Username for that auth, default `admin` |
| `MODEL_PRICES` | No | JSON `{"model": [in_per_mtok, out_per_mtok]}` to price models the built-in table doesn't cover |
| `PORT` | No | Web server port, default `8080` |
