#!/usr/bin/env python3
"""
SEO Article Writer
==================
Generates a complete SEO article with web-sourced images for a given topic.

Mirrors the n8n "SEO Blog Writer Agent Technical Blog" workflow using Claude.

Usage:
    # With uv (no install needed):
    uv run --with anthropic --with requests seo_writer.py "Your Topic Here"
    uv run --with anthropic --with requests seo_writer.py "Your Topic Here" --intent "I want to explain to developers how ReAct agents work and why they're better than standard LLMs"
    uv run --with anthropic --with requests seo_writer.py "Your Topic Here" --keywords "kw1, kw2"

    # Or install deps first:
    pip install anthropic requests
    python seo_writer.py "Your Topic Here" --intent "natural language description of what you want" --output-dir ./articles

Environment Variables:
    ANTHROPIC_API_KEY   (required) Claude API key
    SERPAPI_KEY         (optional) SerpAPI key — enables live SERP data + Google Image search
                        Without it, Claude knowledge is used and Unsplash links are provided.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import anthropic
import requests

# ---------------------------------------------------------------------------
# Load .env file if present (no python-dotenv required)
# ---------------------------------------------------------------------------

def _load_dotenv():
    env_path = Path(__file__).parent / ".env"
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-5"

# Three roles argue about the article: the writer produces it, the auditor attacks
# it, the judge settles what they cannot. The point of the stage is decorrelated
# error, so the roles are resolved onto the most distinct weights the available
# keys allow, rather than being pinned to one vendor.
#
# The auditor matters more than the judge here. The judge only ever rules on
# findings the auditor already raised, so a blind spot shared between writer and
# auditor means the finding never exists to be argued about. Distinctness is
# therefore spent on the auditor first.

VERIFIER_MODEL = "claude-opus-5"    # Anthropic-side auditor: not the writer's weights
VERIFIER_EFFORT = "high"
JUDGE_MODEL = "claude-opus-5"
JUDGE_EFFORT = "high"

OPENAI_JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

# Per-role vendor override. "auto" picks the most distinct vendor that has a key.
#   auto | anthropic | openai | gemini
AUDITOR_PROVIDER = os.getenv("AUDITOR_PROVIDER", "auto").strip().lower()
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "auto").strip().lower()

_PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def available_providers() -> list[str]:
    """Vendors this machine actually holds a key for."""
    return [p for p, env in _PROVIDER_KEYS.items() if os.getenv(env)]


def _provider_model(provider: str, anthropic_model: str) -> str:
    if provider == "openai":
        return OPENAI_JUDGE_MODEL
    if provider == "gemini":
        return GEMINI_MODEL
    return anthropic_model


def resolve_agents() -> dict:
    """Assign writer, auditor and judge to the most distinct vendors available.

    The writer stays on Anthropic: the style prompts and sample-article matching
    were tuned against it, and swapping it changes the product rather than
    checking it. The other two are pushed off the writer's vendor, and off each
    other's, whenever a key exists to do so.
    """
    have = available_providers()

    def pick(override: str, avoid: list[list[str]], preference: list[str]) -> str:
        """avoid is tiered: the first list is the ideal, later lists relax it."""
        if override in _PROVIDER_KEYS:
            return override                      # explicit wins, key checked at call time
        for tier in avoid + [[]]:
            for p in preference:
                if p in have and p not in tier:
                    return p
        return "anthropic"

    auditor = pick(AUDITOR_PROVIDER, avoid=[["anthropic"]],
                   preference=["openai", "gemini", "anthropic"])
    # With only two vendors the judge has to reuse one. Reusing the writer's
    # vendor on a bigger model beats reusing the auditor's exact model, which
    # would have the judge rubber-stamp the finding it just made.
    judge = pick(JUDGE_PROVIDER, avoid=[["anthropic", auditor], [auditor]],
                 preference=["gemini", "openai", "anthropic"])

    return {
        "writer": {"provider": "anthropic", "model": MODEL, "effort": "low"},
        "auditor": {"provider": auditor,
                    "model": _provider_model(auditor, VERIFIER_MODEL),
                    "effort": VERIFIER_EFFORT},
        "judge": {"provider": judge,
                  "model": _provider_model(judge, JUDGE_MODEL),
                  "effort": JUDGE_EFFORT},
    }


def describe_agents(agents: dict) -> str:
    roles = " | ".join(f"{r}: {a['model']} ({a['provider']})" for r, a in agents.items())
    vendors = {a["provider"] for a in agents.values()}
    models = {a["model"] for a in agents.values()}
    return f"{roles}\n  {len(models)} distinct models across {len(vendors)} vendor(s)"


def judge_provider() -> str:
    """Which vendor rules on disputes, after resolving 'auto'. Kept for callers."""
    return resolve_agents()["judge"]["provider"]

# Publication identity. Used for the canonical URL, Open Graph tags and the
# Article/Person schema. Without SITE_URL the canonical and og:url are omitted
# rather than guessed - a wrong canonical is worse than none.
SITE_URL = os.getenv("SITE_URL", "").rstrip("/")
AUTHOR_NAME = os.getenv("AUTHOR_NAME", "Imran Tauqir")
AUTHOR_URL = os.getenv("AUTHOR_URL", "https://imrantauqir.com/")

# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------
#
# A single article makes fifteen-plus calls across three vendors at two effort
# levels, and until now nothing counted them. Tokens are taken from what each
# API actually reports, never estimated. Cost is a second, softer layer: it is
# only as right as the table below, so an unpriced model still gets its tokens
# counted and simply reports no dollar figure rather than a wrong one.
#
# USD per million tokens, (input, output). The Anthropic rows are first-party
# list prices. VERIFY the OpenAI and Gemini rows against your provider's own
# pricing page before trusting the totals - override with MODEL_PRICES, a JSON
# object of {"model": [input, output]}.
MODEL_PRICES = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
try:
    MODEL_PRICES.update({k: tuple(v) for k, v in
                         json.loads(os.getenv("MODEL_PRICES", "{}")).items()})
except (ValueError, TypeError, AttributeError):
    print("Warning: MODEL_PRICES is not valid JSON; using built-in prices.")

# Anthropic bills a cache read at roughly a tenth of the input rate, and a cache
# write at roughly 1.25x. Close enough to be useful, flagged as approximate.
_CACHE_READ_MULTIPLIER = 0.10
_CACHE_WRITE_MULTIPLIER = 1.25

_USAGE_LOG: list[dict] = []
_CURRENT_STEP = "startup"


def _price(model: str, tokens_in: int, tokens_out: int,
           cache_read: int = 0, cache_write: int = 0) -> float | None:
    """Dollars for one call, or None when the model has no price on file."""
    rates = MODEL_PRICES.get(model)
    if not rates:
        return None
    rate_in, rate_out = rates
    billable_in = tokens_in + cache_read * _CACHE_READ_MULTIPLIER \
        + cache_write * _CACHE_WRITE_MULTIPLIER
    return (billable_in * rate_in + tokens_out * rate_out) / 1_000_000


def record_usage(provider: str, model: str, tokens_in: int, tokens_out: int,
                 cache_read: int = 0, cache_write: int = 0):
    """Append one call to the ledger. Called by every vendor wrapper."""
    entry = {
        "step": _CURRENT_STEP,
        "provider": provider,
        "model": model,
        "input_tokens": int(tokens_in or 0),
        "output_tokens": int(tokens_out or 0),
        "cache_read_tokens": int(cache_read or 0),
        "cache_write_tokens": int(cache_write or 0),
        "at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    entry["cost_usd"] = _price(model, entry["input_tokens"], entry["output_tokens"],
                               entry["cache_read_tokens"], entry["cache_write_tokens"])
    _USAGE_LOG.append(entry)


def reset_usage():
    _USAGE_LOG.clear()


def usage_summary() -> dict:
    """Totals overall, per model, and per pipeline step."""
    def blank():
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                "cache_read_tokens": 0, "cost_usd": 0.0, "priced": True}

    total, by_model, by_step = blank(), {}, {}
    for e in _USAGE_LOG:
        for bucket in (total,
                       by_model.setdefault(e["model"], blank()),
                       by_step.setdefault(e["step"], blank())):
            bucket["calls"] += 1
            bucket["input_tokens"] += e["input_tokens"]
            bucket["output_tokens"] += e["output_tokens"]
            bucket["cache_read_tokens"] += e["cache_read_tokens"]
            if e["cost_usd"] is None:
                bucket["priced"] = False
            else:
                bucket["cost_usd"] += e["cost_usd"]

    total["total_tokens"] = total["input_tokens"] + total["output_tokens"]
    return {"total": total, "by_model": by_model, "by_step": by_step,
            "calls": list(_USAGE_LOG)}


def print_usage_summary(summary: dict | None = None):
    summary = summary or usage_summary()
    t = summary["total"]
    if not t["calls"]:
        return
    print(f"\n  Tokens: {t['input_tokens']:,} in + {t['output_tokens']:,} out "
          f"= {t['total_tokens']:,} across {t['calls']} calls")
    for model, m in sorted(summary["by_model"].items(),
                           key=lambda kv: -kv[1]["output_tokens"]):
        cost = f"${m['cost_usd']:.3f}" if m["priced"] else "unpriced"
        print(f"    {model:<20} {m['calls']:>3} calls  "
              f"{m['input_tokens']:>8,} in  {m['output_tokens']:>7,} out  {cost}")
    if t["priced"]:
        print(f"  Estimated cost: ${t['cost_usd']:.2f}")
    else:
        print(f"  Estimated cost: ${t['cost_usd']:.2f} plus unpriced models "
              f"(add them to MODEL_PRICES for a complete figure)")


def write_usage(slug: str, output_dir: Path, title: str = "") -> Path:
    """Save this run's ledger, and append one line to the rolling usage log."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = usage_summary()
    path = output_dir / f"{slug}_usage.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    t = summary["total"]
    line = {
        "at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "slug": slug, "title": title,
        "calls": t["calls"], "input_tokens": t["input_tokens"],
        "output_tokens": t["output_tokens"], "total_tokens": t["total_tokens"],
        "cost_usd": round(t["cost_usd"], 4), "fully_priced": t["priced"],
        "by_model": {m: {"calls": v["calls"],
                         "input_tokens": v["input_tokens"],
                         "output_tokens": v["output_tokens"],
                         "cost_usd": round(v["cost_usd"], 4)}
                     for m, v in summary["by_model"].items()},
    }
    with open(output_dir / "usage.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    return path


# ---------------------------------------------------------------------------
# Article length
# ---------------------------------------------------------------------------
#
# Word count is not one instruction, it is a shape. Telling the writer "1,000
# words" while the outline still demands seven sections, eight images and five
# FAQ questions produces a thin, cramped article rather than a short one, so
# every structural number moves together.

LENGTH_PROFILES = {
    "default": {
        "label": "Default (2,500-3,500 words)",
        "words": "2,500-3,500",
        "sections": "5-7", "subsections": "2-3", "images": "6-8", "faq": "4-5",
        "intro": "150-200", "conclusion": "150-200",
    },
    "2000": {
        "label": "Medium (about 2,000 words)",
        "words": "1,800-2,200",
        "sections": "4-6", "subsections": "2", "images": "4-5", "faq": "4",
        "intro": "120-150", "conclusion": "120-150",
    },
    "1000": {
        "label": "Short (about 1,000 words)",
        "words": "900-1,100",
        "sections": "3-4", "subsections": "1-2", "images": "2-3", "faq": "3",
        "intro": "80-110", "conclusion": "80-110",
    },
}


def length_profile(name: str | None) -> dict:
    """Resolve a --words value to a profile, falling back to the default."""
    return LENGTH_PROFILES.get(str(name or "default").strip().lower(),
                               LENGTH_PROFILES["default"])


SERPAPI_BASE = "https://serpapi.com/search.json"
UNSPLASH_BASE = "https://unsplash.com/s/photos"

client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# Author branding blocks (prepended / appended to every article)
# ---------------------------------------------------------------------------

AUTHOR_INTRO_TEMPLATE = """\
👋 Hi everyone, Imran here.
Welcome to Edition #{edition} of a newsletter that people around the world actually look forward to reading.
"""

AUTHOR_CTA = """\

---

That's a wrap for this edition. If it gave you something useful, the best next step is to try one idea for real this week.

**Let's connect.** 👉 I share what I'm building and learning with AI agents on [LinkedIn](https://www.linkedin.com/in/imrantauqir/) — come say hi, and see my work in my portfolio at [imrantauqir.com](https://imrantauqir.com/).

Until next time,
**Imran**
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:80]


CONSOLE_LIMITS_URL = "https://console.anthropic.com/settings/limits"


class ClaudeError(RuntimeError):
    """An Anthropic API failure, already phrased for a human to read."""


def _api_message(e: "anthropic.APIStatusError") -> str:
    """The API's own error sentence.

    Prefer the parsed body over e.message, which the SDK builds as
    "Error code: 400 - {'type': 'error', ...}" — the whole payload repr.
    """
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        message = (body.get("error") or {}).get("message")
        if message:
            return str(message)
    return e.message


def call_claude(prompt: str, system: str = "", max_tokens: int = 16000,
                model: str = MODEL, effort: str = "low") -> str:
    messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        # Sonnet 5 runs adaptive thinking when `thinking` is omitted, so state it
        # explicitly. Low effort keeps the token cost close to the old no-thinking
        # behaviour while still buying Sonnet 5's better planning.
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort},
    }
    if system:
        kwargs["system"] = system
    try:
        response = client.messages.create(**kwargs)
    except anthropic.BadRequestError as e:
        # The spend cap set in the Console arrives as a 400, not a 429, and the
        # SDK does not retry it — surface the reset date the API gives us.
        message = _api_message(e)
        if "usage limit" in message.lower():
            raise ClaudeError(
                f"Anthropic API usage limit reached. {message} "
                f"Raise or remove the cap here: {CONSOLE_LIMITS_URL}"
            ) from e
        raise ClaudeError(f"Anthropic rejected the request: {message}") from e
    except anthropic.AuthenticationError as e:
        raise ClaudeError(
            "Anthropic rejected the API key. Check that ANTHROPIC_API_KEY is set "
            f"to a current, un-revoked key. ({_api_message(e)})"
        ) from e
    except anthropic.PermissionDeniedError as e:
        detail = (
            "This usually means the account is out of credits."
            if e.type == "billing_error"
            else f"The key may lack access to {model}."
        )
        raise ClaudeError(
            f"Anthropic denied the request. {detail} ({_api_message(e)})"
        ) from e
    except anthropic.RateLimitError as e:
        # The SDK already retried this twice, so it is not a momentary spike.
        raise ClaudeError(
            f"Anthropic rate limit hit and retries were exhausted. Wait a minute "
            f"and run again. ({_api_message(e)})"
        ) from e
    except anthropic.APIStatusError as e:
        raise ClaudeError(
            f"Anthropic returned an error (HTTP {e.status_code}): {_api_message(e)}"
        ) from e
    except anthropic.APIConnectionError as e:
        raise ClaudeError(
            f"Could not reach the Anthropic API. Check your network connection. ({e})"
        ) from e

    u = getattr(response, "usage", None)
    if u is not None:
        record_usage("anthropic", model,
                     getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0),
                     getattr(u, "cache_read_input_tokens", 0) or 0,
                     getattr(u, "cache_creation_input_tokens", 0) or 0)

    if response.stop_reason == "refusal":
        raise ClaudeError(
            "Claude declined to answer this prompt for safety reasons. "
            "Try rephrasing the topic."
        )

    # With thinking enabled, content[0] is a thinking block — pick the text block
    # out rather than indexing blindly.
    text = next((b.text for b in response.content if b.type == "text"), None)

    # A response cut off at the cap used to be returned as if it were complete
    # whenever it had any text at all. Half a JSON object then failed further
    # down with a misleading error about the model "answering in prose". Thinking
    # tokens count toward this cap, so a high-effort step reaches it sooner than
    # its output length suggests.
    if response.stop_reason == "max_tokens":
        written = len((text or "").split())
        raise ClaudeError(
            f"{model} hit the {max_tokens:,}-token output cap and its answer was "
            f"cut off after about {written} words. Thinking tokens count toward "
            f"that cap, so raise max_tokens for this step or lower its effort."
        )

    if text is None or not text.strip():
        raise ClaudeError(
            f"Claude returned no text (stop_reason: {response.stop_reason})."
        )
    return text.strip()


def call_openai(prompt: str, system: str = "", max_tokens: int = 4000) -> str:
    """Second-vendor call, used only for judging disputes."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ClaudeError(
            "JUDGE_PROVIDER is openai but the openai package is not installed. "
            "Run: pip install openai"
        ) from e

    if not os.getenv("OPENAI_API_KEY"):
        raise ClaudeError(
            "JUDGE_PROVIDER is openai but OPENAI_API_KEY is not set. Add it to .env, "
            "or set JUDGE_PROVIDER=anthropic to keep the Claude judge."
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = OpenAI().chat.completions.create(
            model=OPENAI_JUDGE_MODEL,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
    except Exception as e:
        raise ClaudeError(
            f"The OpenAI judge failed ({type(e).__name__}: {e}). If the model name is "
            f"wrong, set OPENAI_JUDGE_MODEL to one your key can reach "
            f"(currently '{OPENAI_JUDGE_MODEL}'), or set JUDGE_PROVIDER=anthropic."
        ) from e

    u = getattr(response, "usage", None)
    if u is not None:
        record_usage("openai", OPENAI_JUDGE_MODEL,
                     getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))

    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ClaudeError(f"The OpenAI judge returned no text (model: {OPENAI_JUDGE_MODEL}).")
    return text


def call_gemini(prompt: str, system: str = "", max_tokens: int = 4000) -> str:
    """Third-vendor call, used for the auditor or judge when a Gemini key is set."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ClaudeError(
            "A role is set to gemini but the google-genai package is not installed. "
            "Run: pip install google-genai"
        ) from e

    if not os.getenv("GEMINI_API_KEY"):
        raise ClaudeError(
            "A role is set to gemini but GEMINI_API_KEY is not set. Add it to .env, "
            "or set the role's provider to anthropic or openai."
        )

    config = {"max_output_tokens": max_tokens}
    if system:
        config["system_instruction"] = system

    try:
        response = genai.Client(api_key=os.getenv("GEMINI_API_KEY")).models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(**config),
        )
    except Exception as e:
        raise ClaudeError(
            f"The Gemini call failed ({type(e).__name__}: {e}). If the model name is "
            f"wrong, set GEMINI_MODEL to one your key can reach (currently "
            f"'{GEMINI_MODEL}'), or set the role's provider to anthropic."
        ) from e

    u = getattr(response, "usage_metadata", None)
    if u is not None:
        record_usage("gemini", GEMINI_MODEL,
                     getattr(u, "prompt_token_count", 0) or 0,
                     getattr(u, "candidates_token_count", 0) or 0,
                     getattr(u, "cached_content_token_count", 0) or 0)

    text = (getattr(response, "text", "") or "").strip()
    if not text:
        raise ClaudeError(f"Gemini returned no text (model: {GEMINI_MODEL}).")
    return text


def call_agent(role: dict, prompt: str, system: str = "", max_tokens: int = 4000,
               agents: dict | None = None) -> str:
    """Run one role on its assigned vendor, falling back to Claude if that vendor is down.

    A vendor outage must not destroy an article that already cost a dozen calls,
    so the fallback is unconditional - but it is announced, because an article
    audited by the writer's own family is a weaker article than the roster claims.
    """
    provider = role["provider"]

    if provider != "anthropic":
        caller = call_openai if provider == "openai" else call_gemini
        try:
            return caller(prompt, system=system, max_tokens=max_tokens)
        except ClaudeError as e:
            reason = " ".join(str(e).split())[:160]
            print(f"  {provider} unavailable, falling back to {JUDGE_MODEL}.")
            print(f"    {reason}")
            return call_claude(prompt, system=system, max_tokens=max_tokens,
                               model=JUDGE_MODEL, effort=role.get("effort", "high"))

    return call_claude(prompt, system=system, max_tokens=max_tokens,
                       model=role["model"], effort=role.get("effort", "high"))


def extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response.

    Eight call sites depend on this, and the responses now come from three
    vendors with different habits: fenced blocks, a sentence of preamble, a
    trailing note. A ValueError escaping here used to crash the pipeline with a
    bare traceback, so every failure now raises ClaudeError carrying the text
    that could not be parsed.
    """
    raw = (text or "").strip()

    # ```json ... ``` fences, which some models add and others never do.
    fenced = re.search(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL)
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.append(raw)

    # Greedy: the outermost braces. Non-greedy: the first complete-looking
    # object, which survives a trailing "Let me know if..." with a brace in it.
    for pattern in (r"\{.*\}", r"\{.*?\}"):
        for source in list(candidates):
            match = re.search(pattern, source, re.DOTALL)
            if match:
                candidates.append(match.group())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed

    preview = " ".join(raw.split())[:300] or "(the response was empty)"
    raise ClaudeError(
        f"A model returned something that is not JSON, where JSON was required. "
        f"This usually means the model answered in prose. It said: {preview}"
    )


def log(step: str, msg: str = ""):
    # The ledger attributes each call to whichever step was last announced, so
    # the dashboard can say which stage of the pipeline spent the tokens.
    global _CURRENT_STEP
    _CURRENT_STEP = step
    print(f"\n[{step}] {msg}" if msg else f"\n[{step}]", flush=True)


# ---------------------------------------------------------------------------
# Intent → Search Query
# ---------------------------------------------------------------------------

def extract_search_query(topic: str, intent: str) -> str:
    """
    Use Claude to convert a natural language intent description into the best
    2–4 word search query for SerpAPI and keyword targeting.
    Returns a short keyword string.
    """
    prompt = f"""You are an SEO keyword research expert.

A writer wants to publish an article on this topic: "{topic}"

Their intent / what they want to achieve:
"{intent}"

Extract the single best search query (2–4 words) that:
1. Captures the core topic for Google search
2. Reflects the angle described in the intent
3. Has high search volume potential

Return ONLY the search query string — no explanation, no quotes, no punctuation."""

    return call_claude(prompt, max_tokens=2000).strip().strip('"').strip("'")


# ---------------------------------------------------------------------------
# Step 1: SERP Research
# ---------------------------------------------------------------------------

def serp_research(title: str, keywords: str, intent: str = "") -> dict:
    log("STEP 1", f"SERP Research for: {keywords}")

    serp_context = ""
    serpapi_key = os.getenv("SERPAPI_KEY")
    if serpapi_key:
        try:
            resp = requests.get(SERPAPI_BASE, params={
                "q": keywords, "num": 5, "api_key": serpapi_key
            }, timeout=15)
            resp.raise_for_status()
            results = resp.json().get("organic_results", [])
            snippets = "\n".join(
                f"- {r.get('title','')}: {r.get('snippet','')}"
                for r in results[:5]
            )
            serp_context = f"\nLive SERP results for '{keywords}':\n{snippets}\n"
            print("  Using live SerpAPI data.")
        except Exception as e:
            print(f"  SerpAPI error ({e}), falling back to Claude knowledge.")

    intent_context = f"\nWriter's intent: {intent}\n" if intent else ""

    prompt = f"""Analyze the topic and keyword below, then return a JSON object with research insights.

Title: {title}
Primary Keyword: {keywords}
{intent_context}{serp_context}

Return ONLY valid JSON, no extra text:
{{
  "search_intent": "<informational | transactional | navigational | commercial>",
  "writing_style": "<e.g. engaging and storytelling | data-driven and technical | etc.>",
  "writing_tone": "<e.g. friendly and conversational | formal and authoritative | etc.>",
  "hidden_insight": "<a unique angle or insight not covered by most articles, or 'No significant insights detected'>",
  "target_audience": "<who this article is for>",
  "article_goal": "<main objective of the article>",
  "semantic_analysis": {{
    "common_subtopics": ["<subtopic 1>", "<subtopic 2>", "<subtopic 3>", "<subtopic 4>"],
    "related_questions": ["<question 1>", "<question 2>", "<question 3>"]
  }},
  "keywords": {{
    "primary_keyword": "<main focus keyword>",
    "secondary_keywords": ["<kw 1>", "<kw 2>", "<kw 3>"],
    "semantic_keywords": ["<kw 1>", "<kw 2>", "<kw 3>"],
    "long_tail_keywords": ["<kw 1>", "<kw 2>", "<kw 3>"]
  }}
}}"""

    response = call_claude(prompt)
    result = extract_json(response)
    # Kept so the Step 6.5 auditor can check the article against what actually
    # ranks, not just against Claude's summary of it.
    result["serp_context"] = serp_context.strip()
    print("  Research complete.")
    return result


# ---------------------------------------------------------------------------
# Step 2: Refine Title
# ---------------------------------------------------------------------------

def refine_title(title: str, keywords: str, research: dict, intent: str = "") -> str:
    log("STEP 2", "Refining title")

    secondary_kws = research.get("keywords", {}).get("secondary_keywords", [])
    intent_note = (
        f"\nWriter's intent (IMPORTANT — use this to stay on topic): {intent}\n"
        f"The revised title must reflect this intent. Do NOT overweight incidental "
        f"details or hardware/product names mentioned in the working title unless "
        f"they are genuinely central to the article's purpose.\n"
        if intent else ""
    )
    prompt = f"""Revise the blog post title to be more SEO-optimized and compelling.

Working title: {title}
Primary keyword: {keywords}
Secondary keywords: {", ".join(secondary_kws)}
Search intent: {research.get("search_intent")}
Writing style: {research.get("writing_style")}
Writing tone: {research.get("writing_tone")}
Article goal: {research.get("article_goal")}
{intent_note}
Rules:
- The title must clearly reflect what the article is ACTUALLY about
- Do not latch onto incidental details, hardware names, or asides from the working title
- Keep the primary subject (the main tool, concept, or framework) front and center

Return ONLY valid JSON:
{{
  "revised_title": "<improved title>",
  "reasoning": "<brief explanation>"
}}"""

    response = call_claude(prompt, max_tokens=4000)
    result = extract_json(response)
    refined = result.get("revised_title", title)
    print(f"  New title: {refined}")
    return refined


# ---------------------------------------------------------------------------
# Step 3: Key Takeaways
# ---------------------------------------------------------------------------

def generate_key_takeaways(title: str, keywords: str, research: dict) -> str:
    log("STEP 3", "Generating key takeaways")

    secondary_kws = research.get("keywords", {}).get("secondary_keywords", [])
    prompt = f"""Create 4–5 key takeaways for this article.

Title: {title}
Primary keyword: {keywords}
Secondary keywords: {", ".join(secondary_kws)}
Search intent: {research.get("search_intent")}
Semantic analysis: {json.dumps(research.get("semantic_analysis", {}))}
Writing style: {research.get("writing_style")}
Writing tone: {research.get("writing_tone")}
Article goal: {research.get("article_goal")}

Write the takeaways as a concise bullet list (4–5 items). Each should be a clear, actionable insight a reader will gain. Start each with "- "."""

    result = call_claude(prompt, max_tokens=4000)
    print(f"  Takeaways generated ({len(result.split(chr(10)))} lines).")
    return result


# ---------------------------------------------------------------------------
# Step 4: Outline
# ---------------------------------------------------------------------------

def generate_outline(title: str, keywords: str, research: dict, key_takeaways: str,
                     profile: dict | None = None) -> str:
    profile = profile or length_profile(None)
    log("STEP 4", f"Generating article outline - {profile['label']}")

    kw_data = research.get("keywords", {})
    secondary_kws = ", ".join(kw_data.get("secondary_keywords", []))
    semantic_kws = ", ".join(kw_data.get("semantic_keywords", []))
    long_tail_kws = ", ".join(kw_data.get("long_tail_keywords", []))

    prompt = f"""You are an expert SEO content strategist. Create a detailed article outline.

ARTICLE SPECIFICATIONS
Title: {title}
Primary Keyword: {keywords}
Target Word Count: {profile['words']} words
Writing Tone: {research.get("writing_tone")}
Writing Style: {research.get("writing_style")}

STRATEGIC FOUNDATION
Search Intent: {research.get("search_intent")}
Semantic Context: {json.dumps(research.get("semantic_analysis", {}))}
Article Goal: {research.get("article_goal")}
Hidden Insight to Weave In: {research.get("hidden_insight")}
Target Audience: {research.get("target_audience")}

KEY CONTENT ELEMENTS
Key Takeaways (must be featured prominently):
{key_takeaways}

SEO KEYWORD STRATEGY
Primary Keyword: {keywords}
Secondary Keywords: {secondary_kws}
Semantic Keywords: {semantic_kws}
Long-tail Keywords: {long_tail_kws}

OUTLINE REQUIREMENTS
Produce a detailed markdown outline with:
1. H1 (the article title)
2. Introduction section ({profile['intro']} words)
3. {profile['sections']} H2 main sections, each with {profile['subsections']} H3 subsections
4. For each section: brief description of what to cover (1–2 sentences)
5. {profile['images']} image placement markers formatted as:
   [IMAGE: <descriptive alt text> | Query: <google image search query>]
6. At least one comparison table, placed in whichever section it genuinely belongs
   to. Note its columns in the outline. Tables are the passage an answer engine is
   most likely to quote whole, so give it real rows: prices, versions, tradeoffs,
   or a this-vs-that. Do not invent a table where the topic has nothing to compare.
7. A FAQ section with {profile['faq']} questions. Each MUST be an H3 phrased as a real question
   ending in a question mark, and each answer must stand on its own without the
   surrounding page - they are extracted into FAQPage structured data.
8. Conclusion section ({profile['conclusion']} words with CTA)
9. Supplementary metadata block at the end:
   - URL slug suggestion
   - 5–7 internal linking opportunities
   - Keyword density targets

Format as clean markdown. Be specific — each section note should guide the writer clearly."""

    result = call_claude(prompt, max_tokens=8000)
    print(f"  Outline generated ({len(result.split(chr(10)))} lines).")
    return result


# ---------------------------------------------------------------------------
# Step 5: Write Content
# ---------------------------------------------------------------------------

def write_content(title: str, keywords: str, outline: str, research: dict,
                  key_takeaways: str, profile: dict | None = None) -> str:
    profile = profile or length_profile(None)
    log("STEP 5", f"Writing article content, target {profile['words']} words "
                  f"(this may take a moment...)")

    kw_data = research.get("keywords", {})
    secondary_kws = ", ".join(kw_data.get("secondary_keywords", []))

    system = """You are an expert content writer. Write clear, structured, value-driven articles that rank well in search engines. Use active voice, short paragraphs (3–4 sentences max), and cite sources inline as 'Source: https://...' when referencing external data or studies."""

    prompt = f"""Write a complete, high-quality SEO article based on the inputs below.

INPUTS
Title: {title}
Primary Keyword: {keywords}
Secondary Keywords: {secondary_kws}
Outline to follow strictly:
{outline}

Key Takeaways (must be reflected in writing):
{key_takeaways}

WRITING CONTEXT
Writing Style: {research.get("writing_style")}
Writing Tone: {research.get("writing_tone")}
Search Intent: {research.get("search_intent")}
Hidden Insight to highlight: {research.get("hidden_insight")}
Target Audience: {research.get("target_audience")}
Article Goal: {research.get("article_goal")}

INSTRUCTIONS
1. Follow the outline structure strictly (H1, H2, H3 headings).
2. Keep each paragraph to 3–4 sentences maximum.
3. Integrate keywords naturally — no stuffing.
4. Cite sources inline where relevant: "Source: https://..."
5. Preserve all [IMAGE: ...] markers from the outline exactly as-is — do not remove them.
6. Include the FAQ section and Conclusion from the outline. Every FAQ question must
   be an H3 ending in a question mark, and its answer must make sense quoted on its
   own - those pairs become FAQPage structured data.
7. Build any comparison table the outline calls for as a real markdown table with a
   header row. It is the passage most likely to be quoted whole by an answer engine.
8. Target {profile['words']} words total. This is a real constraint, not a
   suggestion: cut depth rather than sections, and never pad to reach it.
9. Bold key terms on first use.
10. End with a strong call-to-action.

Write the full article now. Output the article content ONLY."""

    result = call_claude(prompt, max_tokens=16000)
    word_count = len(result.split())
    print(f"  Article written ({word_count} words).")
    return result


# ---------------------------------------------------------------------------
# Step 6: Humanize
# ---------------------------------------------------------------------------

_HUMANIZER_PATTERNS = """
## AI writing patterns to detect and fix

### Content patterns
1. Significance inflation — "stands as", "serves as a testament", "pivotal moment", "evolving landscape",
   "underscores", "highlights its importance", "setting the stage for", "indelible mark", "deeply rooted"
   → Replace with plain factual statements.

2. Notability puffery — "active social media presence", "written by a leading expert", "featured in X, Y, Z"
   → Keep only if specific and sourced; otherwise cut.

3. Superficial -ing analyses — tacking "-ing" participle phrases onto sentences to fake depth:
   "highlighting...", "symbolizing...", "contributing to...", "showcasing...", "underscoring..."
   → Delete or fold the point into the sentence directly.

4. Promotional language — "boasts", "vibrant", "rich cultural heritage", "nestled", "breathtaking",
   "groundbreaking", "renowned", "stunning", "must-visit"
   → Replace with neutral, specific description.

5. Vague attributions — "Experts argue", "Industry reports", "Some critics say", "Observers note"
   → Name the source or cut the claim.

6. Formulaic challenges sections — "Despite its X, it faces challenges… Despite these challenges…"
   → Describe the specific problem with specifics; drop the frame.

### Language / grammar patterns
7. AI vocabulary — additionally, align with, crucial, delve, emphasizing, enduring, enhance, fostering,
   garner, highlight (verb), interplay, intricate/intricacies, key (adj.), landscape (abstract), pivotal,
   showcase, tapestry, testament, underscore (verb), valuable, vibrant
   → Use plain alternatives or cut.

8. Copula avoidance — "serves as", "stands as", "marks", "represents", "boasts", "features", "offers"
   used where "is/are/has" would do → Replace with simple copulas.

9. Negative parallelisms — "Not only X but Y", "It's not just about X; it's about Y"
   → Flatten into a direct statement.

10. Rule of three overuse — forcing ideas into groups of three
    → Use as many items as there actually are.

11. Synonym cycling — rotating synonyms to avoid repeating a word ("the protagonist… the main character…
    the central figure… the hero") → Repeat the word or restructure.

12. False ranges — "from X to Y, from A to B" where X/Y aren't on a meaningful scale → List or summarize plainly.

### Style patterns
13. Em dash overuse — replace em dashes (—) with commas, parentheses, or restructured sentences where possible.

14. Excessive boldface — bold only terms that genuinely need emphasis; remove decorative bolding.

15. Inline-header bullet lists — "- **Speed:** Faster because…" → Convert to prose or clean bullets without bold headers.

16. Title Case In Headings — change to sentence case (first word + proper nouns only).

17. Emojis in headings/bullets → Remove unless they are in the original brand intro block.

18. Curly quotation marks → Keep as-is (they're fine); just don't introduce new ones inconsistently.

### Communication patterns
19. Chatbot artifacts — "Great question!", "I hope this helps!", "Let me know if…", "Here is a…"
    → Delete entirely.

20. Knowledge-cutoff disclaimers — "As of my last update…", "While specific details are limited…"
    → Delete or replace with a real source.

21. Sycophantic tone — "You're absolutely right", "That's an excellent point", "Of course!"
    → Delete.

### Filler and hedging
22. Filler phrases — "In order to" → "To"; "Due to the fact that" → "Because"; "At this point in time" → "Now";
    "It is important to note that" → cut it; "has the ability to" → "can".

23. Excessive hedging — "could potentially possibly be argued that… might" → pick one hedge or none.

24. Generic positive conclusions — "The future looks bright", "exciting times lie ahead", "a step in the right direction"
    → End with a specific fact, next step, or genuine observation.

### Voice and personality (beyond removing patterns)
- Vary sentence length. Short punchy sentences mixed with longer ones.
- Have opinions where appropriate — "I keep coming back to…", "Here's what gets me…"
- Acknowledge complexity — "This is impressive but also kind of unsettling."
- Be specific about feelings rather than vague ("there's something unsettling about…" not "this is concerning").
- Let some imperfection in — perfect parallel structure feels algorithmic.
"""


def humanize_content(content: str) -> str:
    log("STEP 6", "Humanizing content (pass 1 — pattern removal)")

    system = (
        "You are an expert human editor. Your job is to make AI-generated writing sound like it was "
        "written by a knowledgeable, opinionated human blogger. You know every tell-tale AI pattern "
        "and ruthlessly eliminate them while keeping the article's structure, SEO value, and accuracy intact."
    )

    prompt = f"""Rewrite the article below to remove AI writing patterns and add genuine human voice.

STRUCTURAL CONSTRAINTS (never break these):
- Preserve ALL markdown headings (H1, H2, H3) exactly as written
- Preserve ALL [IMAGE: alt text | Query: ...] markers exactly — do not move, rename, or remove them
- Preserve ALL "Source: ..." citations exactly
- Keep short paragraphs (3–4 sentences max)
- Do NOT remove any sections or change the article structure
- Do NOT add new factual claims

AI PATTERN CHECKLIST — fix every instance you find:
{_HUMANIZER_PATTERNS}

ARTICLE TO REWRITE:
{content}

Return ONLY the rewritten article. No preamble, no commentary."""

    draft = call_claude(prompt, max_tokens=16000)
    print(f"  Pass 1 complete ({len(draft.split())} words). Running self-audit...")

    # Pass 2: self-audit and final polish
    log("STEP 6", "Humanizing content (pass 2 — self-audit)")

    audit_prompt = f"""You are a sharp editor. Read the article below and answer:

QUESTION 1: What still makes this obviously AI-generated? List the remaining tells as brief bullet points (5 words max each). If none, say "None found."

QUESTION 2: Now rewrite the article fixing those remaining tells. Apply the same structural constraints:
- Preserve ALL markdown headings (H1, H2, H3) exactly
- Preserve ALL [IMAGE: ...] markers exactly
- Preserve ALL "Source: ..." citations exactly
- Keep paragraphs to 3–4 sentences max
- Do NOT add new factual claims or remove sections

Output format — use these exact labels:
REMAINING TELLS:
<bullet list or "None found">

FINAL ARTICLE:
<the full rewritten article>

ARTICLE:
{draft}"""

    audit_result = call_claude(audit_prompt, max_tokens=16000)

    # Extract the final article from the audit output
    if "FINAL ARTICLE:" in audit_result:
        tells_section = audit_result.split("FINAL ARTICLE:")[0]
        final = audit_result.split("FINAL ARTICLE:", 1)[1].strip()
        # Log the remaining tells for visibility
        if "REMAINING TELLS:" in tells_section:
            tells = tells_section.split("REMAINING TELLS:", 1)[1].strip()
            print(f"  Remaining tells fixed: {tells[:200]}")
    else:
        # Fallback: use the draft if audit output is malformed
        final = draft
        print("  Self-audit parse failed, using pass 1 output.")

    final = _strip_em_dashes(final)

    # Pass 3: repair the tells a scanner can name, rather than asking again in general.
    scan = scan_ai_tells(final)
    print_tell_scan(scan, "Scan after pass 2")

    uniform = scan["sentence_count"] > 4 and scan["sentence_cv"] < 0.45
    if scan["total"] or uniform:
        log("STEP 6", "Humanizing content (pass 3 - targeted repair)")
        repaired = _strip_em_dashes(repair_ai_tells(final, scan))
        after = scan_ai_tells(repaired)
        print_tell_scan(after, "Scan after pass 3")
        # A repair pass that made the article worse is not a repair. Keep pass 2.
        if after["total"] > scan["total"]:
            print(f"  Pass 3 raised the count {scan['total']} -> {after['total']}; "
                  f"keeping the pass 2 text.")
        else:
            print(f"  Pass 3 removed {scan['total'] - after['total']} tell(s).")
            final = repaired
    else:
        print("  Scanner found nothing to repair.")

    word_count = len(final.split())
    print(f"  Humanized ({word_count} words).")
    return final


def repair_ai_tells(article: str, scan: dict) -> str:
    """Rewrite only the phrases the scanner named. Everything else stays put."""
    report = format_tell_report(scan)

    prompt = f"""A scanner found these AI tells in your article. Fix each one where it is
genuinely a tell, and leave it alone where the word is doing real work.

WHAT THE SCANNER FOUND
{report}

These are candidates, not verdicts. A scanner cannot read context: "key" in "API key" is
correct, a heading of proper nouns is not title case, and a technical term with no plain
synonym should stay. Judge each one, then fix the genuine ones.

STRUCTURAL CONSTRAINTS (never break these):
- Preserve ALL markdown headings unless the finding is that a heading is title-cased,
  in which case change only its capitalisation
- Preserve ALL [IMAGE: alt text | Query: ...] markers exactly
- Preserve ALL "Source: ..." citations exactly
- Do NOT add new factual claims, and do NOT remove sections
- Do NOT rewrite sentences that contain none of the phrases above, except where the
  finding is that sentence length is too uniform

If the scanner reported uniform sentence length, vary it for real: cut some sentences to
under eight words, let others run long. Do not simply split every sentence in half.

ARTICLE
{article}

Return ONLY the revised article. No preamble, no list of what you changed."""

    return call_claude(prompt, max_tokens=16000)


def _strip_em_dashes(text: str) -> str:
    """
    Replace em dashes with natural punctuation.
    Rules:
      " — "  (spaced em dash mid-sentence)  → ", "
      "—"    (tight em dash, e.g. compound) → "-"
    Headings get their own rule. A comma reads wrong in a heading, so a spaced em
    dash there becomes a colon, or a hyphen when the heading already has one. This
    used to skip headings entirely, which is why every shipped article still had
    nine to thirteen em dashes in its section titles - the exact tell the verifier
    is told to look for.
    Image markers and source citations are still skipped: those are structural
    strings the rest of the pipeline matches on, not prose.
    """
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.lstrip()
        if (stripped.startswith("[IMAGE:")
                or stripped.startswith("*Source:")
                or stripped.startswith("Source:")
                or stripped == "---"):
            result.append(line)
            continue
        if stripped.startswith("#"):
            if "—" in line:
                sep = " - " if ":" in line else ": "
                line = line.replace(" — ", sep).replace("—", "-")
            result.append(line)
            continue
        # Spaced em dash → comma (most common inline use)
        line = line.replace(" — ", ", ")
        # Tight em dash → hyphen (compound words / ranges)
        line = line.replace("—", "-")
        result.append(line)

    removed = text.count("—")
    if removed:
        print(f"  Em dashes removed/replaced: {removed}")
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Deterministic AI-tell scanner
# ---------------------------------------------------------------------------
#
# The self-audit in step 6 asks a model whether its own draft still reads as AI.
# That is the same judgment that produced the tells in the first place, and it
# has no way to be wrong out loud. This scanner is the boring counterweight: it
# holds no opinion, it counts. What it finds goes back to the model as an exact
# phrase list, so the repair pass fixes named strings instead of trying harder
# in general.
#
# These lists mirror the numbered items in _HUMANIZER_PATTERNS. Keep them in step.

_TELL_VOCAB = (
    # 7 - AI vocabulary
    "additionally", "crucial", "delve", "delves", "delving", "emphasizing",
    "enduring", "enhance", "enhances", "enhancing", "fostering", "garner",
    "interplay", "intricate", "intricacies", "pivotal", "showcase", "showcases",
    "showcasing", "tapestry", "testament", "underscore", "underscores",
    "underscoring", "vibrant", "boasts", "nestled", "renowned", "groundbreaking",
    "breathtaking", "myriad", "realm", "seamless", "seamlessly", "robust",
    "leverage", "leveraging", "navigate", "navigating", "unlock", "unlocking",
    "harness", "harnessing", "elevate", "profound", "paramount", "meticulous",
    "meticulously", "bustling", "captivating", "unwavering", "transformative",
)

_TELL_PHRASES = (
    # 1 - significance inflation
    "stands as", "serves as a testament", "pivotal moment", "evolving landscape",
    "setting the stage for", "indelible mark", "deeply rooted", "plays a vital role",
    "plays a crucial role", "in today's world", "in the world of",
    # 5 - vague attribution
    "experts say", "experts argue", "industry reports", "some critics say",
    "observers note", "studies show", "research suggests", "it is widely",
    # 6 / 24 - formulaic frames and conclusions
    "despite these challenges", "the future looks bright", "exciting times",
    "a step in the right direction", "only time will tell", "one thing is clear",
    "when it comes to", "at the end of the day",
    # 9 - negative parallelism
    "not only", "it's not just about", "it is not just about",
    # 19 / 21 / 22 - chatbot artifacts, sycophancy, filler
    "great question", "i hope this helps", "let me know if", "here is a",
    "you're absolutely right", "that's an excellent point",
    "it is important to note", "it's important to note", "in order to",
    "due to the fact that", "at this point in time", "has the ability to",
    "it is worth noting", "needless to say",
    # 20 - knowledge-cutoff disclaimers
    "as of my last update", "while specific details are limited",
)

# Words that stay lowercase in sentence case, so a heading full of them is not
# evidence of title casing.
_HEADING_STOPWORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into", "of",
    "on", "or", "the", "to", "vs", "with", "over", "via", "per",
}


def _scannable_lines(text: str) -> list[str]:
    """Body prose only. Citations, image markers and code are not the writer's voice."""
    lines, in_code = [], False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if (stripped.startswith("[IMAGE:")
                or stripped.startswith("Source:")
                or stripped.startswith("*Source:")
                or stripped.startswith("> Source:")):
            continue
        lines.append(line)
    return lines


def _is_title_case(heading: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", heading)
    if len(words) < 3:
        return False
    # The first word is capitalised in both conventions, so it proves nothing.
    rest = [w for w in words[1:] if w.lower() not in _HEADING_STOPWORDS]
    if len(rest) < 2:
        return False
    capped = sum(1 for w in rest if w[0].isupper())
    return capped / len(rest) >= 0.7


def _sentence_lengths(lines: list[str]) -> list[int]:
    """Word counts per sentence, skipping headings, bullets and blank lines."""
    prose = [l for l in lines
             if l.strip() and not l.lstrip().startswith(("#", "-", "*", "|", ">"))]
    lengths = []
    for sentence in re.split(r"(?<=[.!?])\s+", " ".join(prose)):
        n = len(sentence.split())
        if n:
            lengths.append(n)
    return lengths


def scan_ai_tells(text: str) -> dict:
    """Count concrete AI tells. No model call, no judgment - just occurrences."""
    lines = _scannable_lines(text)
    body = "\n".join(lines)
    low = body.lower()

    hits = []

    for word in _TELL_VOCAB:
        n = len(re.findall(rf"\b{re.escape(word)}\b", low))
        if n:
            hits.append({"kind": "vocab", "phrase": word, "count": n})

    for phrase in _TELL_PHRASES:
        n = low.count(phrase)
        if n:
            hits.append({"kind": "phrase", "phrase": phrase, "count": n})

    structural = []
    title_cased = [l.strip() for l in lines
                   if l.lstrip().startswith("#") and _is_title_case(l.lstrip("# ").strip())]
    if title_cased:
        structural.append({"kind": "title_case_heading", "count": len(title_cased),
                           "examples": title_cased[:5]})

    inline_headers = [l.strip() for l in lines
                      if re.match(r"\s*[-*]\s+\*\*[^*]+:\*\*", l)]
    if inline_headers:
        structural.append({"kind": "inline_header_bullet", "count": len(inline_headers),
                           "examples": inline_headers[:5]})

    em_dashes = body.count("—")
    if em_dashes:
        structural.append({"kind": "em_dash", "count": em_dashes, "examples": []})

    emoji_headings = [l.strip() for l in lines
                      if l.lstrip().startswith("#")
                      and re.search(r"[\U0001F300-\U0001FAFF☀-➿]", l)]
    if emoji_headings:
        structural.append({"kind": "emoji_heading", "count": len(emoji_headings),
                           "examples": emoji_headings[:5]})

    # Uniform sentence length is the tell no wordlist catches. Human prose varies;
    # generated prose clusters around one comfortable length.
    lengths = _sentence_lengths(lines)
    cv = 0.0
    if len(lengths) > 4:
        mean = sum(lengths) / len(lengths)
        if mean:
            var = sum((n - mean) ** 2 for n in lengths) / len(lengths)
            cv = (var ** 0.5) / mean

    words = max(len(body.split()), 1)
    total = sum(h["count"] for h in hits) + sum(s["count"] for s in structural)

    return {
        "hits": sorted(hits, key=lambda h: -h["count"]),
        "structural": structural,
        "sentence_cv": round(cv, 3),
        "sentence_count": len(lengths),
        "words": words,
        "total": total,
        "per_1k": round(total / words * 1000, 2),
    }


def format_tell_report(scan: dict) -> str:
    """The scan as a phrase list a model can act on, most frequent first."""
    parts = []
    for h in scan["hits"][:40]:
        parts.append(f'- "{h["phrase"]}" x{h["count"]}')
    for s in scan["structural"]:
        line = f'- {s["kind"].replace("_", " ")} x{s["count"]}'
        for ex in s.get("examples", [])[:3]:
            line += f'\n    e.g. {ex[:90]}'
        parts.append(line)
    if scan["sentence_count"] > 4 and scan["sentence_cv"] < 0.45:
        parts.append(
            f'- sentence length is too uniform (variation {scan["sentence_cv"]}, '
            f'want 0.45+ across {scan["sentence_count"]} sentences)'
        )
    return "\n".join(parts)


def print_tell_scan(scan: dict, label: str):
    top = ", ".join(f'{h["phrase"]}x{h["count"]}' for h in scan["hits"][:6])
    print(f"  {label}: {scan['total']} tells ({scan['per_1k']}/1k words), "
          f"sentence variation {scan['sentence_cv']}")
    if top:
        print(f"    most frequent: {top}")
    for s in scan["structural"]:
        print(f"    {s['kind'].replace('_', ' ')}: {s['count']}")


# ---------------------------------------------------------------------------
# Step 7: Meta Description
# ---------------------------------------------------------------------------

def generate_meta(title: str, keywords: str, content: str) -> dict:
    log("STEP 7", "Generating SEO meta description")

    # Pass a trimmed preview of the article to stay within token limits
    content_preview = content[:3000]

    prompt = f"""Generate an SEO-optimized meta description for this article.

Title: {title}
Primary Keyword: {keywords}
Article Preview:
{content_preview}

Requirements:
- Exactly 150–160 characters including spaces
- Primary keyword in the first 30 characters, naturally integrated
- Include 1–2 secondary keywords organically
- Address what the reader will gain
- Sound conversational and human — not robotic

Return ONLY valid JSON:
{{
  "seo_meta": {{
    "title": "{title}",
    "description": "<150-160 char meta description>",
    "slug": "<url-friendly-slug>"
  }}
}}"""

    response = call_claude(prompt, max_tokens=4000)
    result = extract_json(response)
    meta = result.get("seo_meta", {})
    desc = meta.get("description", "")
    print(f"  Meta description ({len(desc)} chars): {desc[:80]}...")
    return meta


# ---------------------------------------------------------------------------
# Step 8: Image Search
# ---------------------------------------------------------------------------

def search_images(content: str, research: dict) -> dict[str, dict]:
    """
    Find image placements in the article and resolve URLs.
    Returns a dict mapping the original [IMAGE: ...] marker to image data.
    """
    log("STEP 8", "Resolving images")

    # Extract all [IMAGE: alt text | Query: search query] markers
    pattern = re.compile(r"\[IMAGE:\s*([^|]+)\|\s*Query:\s*([^\]]+)\]", re.IGNORECASE)
    markers = pattern.findall(content)

    if not markers:
        print("  No [IMAGE: ...] markers found in article.")
        return {}

    serpapi_key = os.getenv("SERPAPI_KEY")
    images = {}

    for alt_text, query in markers:
        alt_text = alt_text.strip()
        query = query.strip()
        marker_key = f"[IMAGE: {alt_text} | Query: {query}]"

        image_url = None
        source_url = None

        if serpapi_key:
            try:
                resp = requests.get(SERPAPI_BASE, params={
                    "engine": "google_images",
                    "q": query,
                    "api_key": serpapi_key,
                    "num": 3,
                }, timeout=15)
                resp.raise_for_status()
                img_results = resp.json().get("images_results", [])
                if img_results:
                    top = img_results[0]
                    image_url = top.get("original")
                    source_url = top.get("source") or top.get("link")
                    print(f"  [Google Images] {alt_text[:50]}: {image_url[:60] if image_url else 'none'}...")
            except Exception as e:
                print(f"  SerpAPI image search error ({e}), using Unsplash fallback.")

        if not image_url:
            # Fallback: Unsplash search URL
            unsplash_query = quote_plus(query)
            image_url = f"{UNSPLASH_BASE}/{unsplash_query}"
            source_url = f"{UNSPLASH_BASE}/{unsplash_query}"
            print(f"  [Unsplash fallback] {alt_text[:50]}")

        images[marker_key] = {
            "alt": alt_text,
            "query": query,
            "url": image_url,
            "source": source_url,
        }

    print(f"  Resolved {len(images)} image(s).")
    return images


# ---------------------------------------------------------------------------
# Inject Images into Article
# ---------------------------------------------------------------------------

def inject_images(content: str, images: dict[str, dict]) -> str:
    """Replace [IMAGE: ...] markers with actual markdown image blocks."""
    for marker, img in images.items():
        # Build markdown image with source attribution (matches sample article style)
        is_unsplash = "unsplash.com" in img["url"]
        source_note = (
            f"*Source: [Unsplash — search '{img['query']}']({img['source']}) — "
            "select and attribute your chosen image*"
            if is_unsplash
            else f"*Source: {img['source']}*"
        )
        replacement = f"\n![{img['alt']}]({img['url']})\n{source_note}\n"

        # Match the marker even if the content slightly altered whitespace
        escaped = re.escape(marker)
        content = re.sub(escaped, replacement, content)

    return content


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def extract_sources(article: str) -> list[str]:
    """Extract all URLs cited in the article (Source: lines + inline links)."""
    urls = []
    # Source: https://... lines
    for m in re.finditer(r'\(Source:\s*(https?://[^\s\)]+)\)', article):
        urls.append(m.group(1))
    # Inline markdown links [text](url)
    for m in re.finditer(r'\]\((https?://[^\s\)]+)\)', article):
        urls.append(m.group(1))
    # Plain Source: https://... lines
    for m in re.finditer(r'Source\s*:\s*(https?://\S+)', article):
        urls.append(m.group(1))
    # Deduplicate while preserving order
    seen = set()
    result = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


# Domains whose display name is not just the second-level label capitalised.
_PUBLISHER_NAMES = {
    "arxiv.org": "arXiv", "nytimes.com": "The New York Times", "wsj.com": "The Wall Street Journal",
    "ft.com": "Financial Times", "bbc.co.uk": "BBC", "bbc.com": "BBC", "theverge.com": "The Verge",
    "techcrunch.com": "TechCrunch", "arstechnica.com": "Ars Technica", "github.com": "GitHub",
    "openai.com": "OpenAI", "anthropic.com": "Anthropic", "nvidia.com": "NVIDIA",
    "developer.nvidia.com": "NVIDIA Developer", "aws.amazon.com": "AWS", "cloud.google.com":
    "Google Cloud", "learn.microsoft.com": "Microsoft Learn", "en.wikipedia.org": "Wikipedia",
    "huggingface.co": "Hugging Face", "stackoverflow.com": "Stack Overflow",
}


def publisher_name(url: str) -> str:
    """A human name for the site behind a URL, for citations that read as citations."""
    host = re.sub(r"^https?://", "", url).split("/")[0].lower().lstrip("www.")
    if host in _PUBLISHER_NAMES:
        return _PUBLISHER_NAMES[host]
    for domain, name in _PUBLISHER_NAMES.items():
        if host.endswith("." + domain):
            return name
    label = host.split(".")[0] if host.count(".") <= 1 else host.split(".")[-2]
    if not label:
        return host
    # A three-letter domain is nearly always an acronym: idc, ibm, bbc, acm.
    if len(label) <= 3 and label.isalpha():
        return label.upper()
    return label.replace("-", " ").title()


def build_sources_section(article: str, accessed: str = "") -> str:
    """Sources as named, dated citations.

    A bare list of naked URLs is worth far less than the same list with a
    publisher and a date on it: answer engines weight attributable citations,
    and a reader cannot judge a link they cannot identify without clicking it.
    """
    urls = extract_sources(article)
    if not urls:
        return ""
    accessed = accessed or datetime.now().strftime("%B %d, %Y")
    lines = ["", "---", "", "## Sources", ""]
    for i, url in enumerate(urls, 1):
        lines.append(f"{i}. {publisher_name(url)} — [{url}]({url}) (accessed {accessed})")
    return "\n".join(lines) + "\n"


def parse_faq(article: str) -> list[dict]:
    """Pull question/answer pairs out of the article's FAQ section.

    Only the FAQ section, and only headings that are actually questions - a
    FAQPage schema containing things that are not questions is worse than none.
    """
    lines = article.split("\n")
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^#{2,3}\s", line) and (
                "faq" in line.lower() or "frequently asked" in line.lower()):
            start = i + 1
            break
    if start is None:
        return []

    def clean(q: str) -> str:
        # Articles often label questions "Q: ...". The label is presentation;
        # a FAQPage question name that starts with "Q:" reads as malformed data.
        return re.sub(r"^\s*(Q\s*[:.\-]|Question\s*[:.\-])\s*", "", q).strip()

    faqs, question, answer = [], None, []
    for line in lines[start:]:
        heading = re.match(r"^(#{2,4})\s+(.*)", line)
        if heading:
            level, text = len(heading.group(1)), heading.group(2).strip()
            if question and answer:
                faqs.append({"question": question, "answer": " ".join(answer).strip()})
            question, answer = None, []
            if level <= 2:
                break                     # left the FAQ section
            if text.endswith("?"):
                question = clean(text)
            continue
        bold_q = re.match(r"^\*\*(.+\?)\*\*\s*$", line.strip())
        if bold_q:
            if question and answer:
                faqs.append({"question": question, "answer": " ".join(answer).strip()})
            question, answer = clean(bold_q.group(1)), []
            continue
        if question and line.strip() and not line.strip().startswith(("[IMAGE:", "Source:", "!")):
            answer.append(line.strip())

    if question and answer:
        faqs.append({"question": question, "answer": " ".join(answer).strip()})
    return [f for f in faqs if f["answer"]]


def build_jsonld(slug: str, article: str, meta: dict, images: dict,
                 generated_at: str) -> str:
    """Article + FAQPage structured data.

    Everything here is already computed elsewhere in the pipeline and was
    previously written only to _meta.json, where no crawler will ever see it.
    """
    seo_title = meta.get("title") or ""
    description = meta.get("description") or ""
    canonical = f"{SITE_URL}/{slug}" if SITE_URL else ""

    article_node = {
        "@type": "Article",
        "headline": seo_title[:110],          # schema.org caps headline at 110 chars
        "description": description,
        "datePublished": generated_at,
        "dateModified": generated_at,
        "author": {"@type": "Person", "name": AUTHOR_NAME, "url": AUTHOR_URL},
        "publisher": {"@type": "Person", "name": AUTHOR_NAME, "url": AUTHOR_URL},
        "inLanguage": "en",
        "wordCount": len(article.split()),
    }
    if canonical:
        article_node["url"] = canonical
        article_node["mainEntityOfPage"] = {"@type": "WebPage", "@id": canonical}
    image_urls = [v["url"] for v in images.values() if v.get("url")]
    if image_urls:
        article_node["image"] = image_urls[:6]
    citations = extract_sources(article)
    if citations:
        article_node["citation"] = [
            {"@type": "CreativeWork", "name": publisher_name(u), "url": u}
            for u in citations[:20]
        ]

    graph = [article_node]

    faqs = parse_faq(article)
    if faqs:
        graph.append({
            "@type": "FAQPage",
            "mainEntity": [
                {"@type": "Question", "name": f["question"],
                 "acceptedAnswer": {"@type": "Answer", "text": f["answer"]}}
                for f in faqs
            ],
        })

    payload = {"@context": "https://schema.org", "@graph": graph}
    # </script> inside a JSON string would close the tag early.
    return json.dumps(payload, indent=2, ensure_ascii=False).replace("</", "<\\/")


def generate_answer_block(title: str, article: str, research: dict) -> str:
    """A short, extractable answer placed directly under the H1.

    Answer engines quote the first self-contained passage that answers the
    query. Key takeaways are bullets about the article; this is an answer to the
    question, written to survive being lifted out of the page on its own.
    """
    log("STEP 6.4", "Writing the direct-answer block")

    kw = research.get("keywords", {})
    prompt = f"""Write the short answer that belongs directly under this article's title.

THE QUESTION A READER IS ASKING
{kw.get("primary_keyword", title)}

RULES
- 40 to 60 words. Not a word more.
- Answer the question in the first sentence. No preamble, no "in this article".
- Lead with a definition or a direct claim: "X is ...", "X costs ...", "Yes, because ...".
- Include the single most useful specific: a number, a price, a version, a timeframe.
- It must make complete sense quoted on its own, with no surrounding page.
- Plain sentences. No bullets, no heading, no bold, no em dashes.
- Claim nothing the article does not already support.

THE ARTICLE
{article[:6000]}

Return ONLY the answer paragraph."""

    answer = _strip_em_dashes(call_claude(prompt, max_tokens=600).strip())
    words = len(answer.split())
    print(f"  Answer block: {words} words")
    if words > 90:
        print("  Answer block came back too long; skipping it rather than "
              "burying the intro.")
        return ""
    return answer


def generate_linkedin_post(title: str, article: str, research: dict,
                           article_url: str = "") -> str:
    """A LinkedIn post drawn from the finished article.

    Written from the verified article rather than the topic, so the post can
    only claim things the article actually established - a post generated from
    the brief would be free to invent a statistic the article never supports.
    """
    log("STEP 9", "Writing the LinkedIn post")

    kw = research.get("keywords", {})
    link = article_url or (f"{SITE_URL}/{slugify(title)}" if SITE_URL else "")

    prompt = f"""Write a LinkedIn post for the article below, in the author's own voice.

WHO IS POSTING
Imran Tauqir, an engineer who builds with AI agents and writes about it. He posts
as a practitioner, not a commentator. He is not selling anything in this post.

WHAT THE POST HAS TO DO
Earn the click from someone scrolling past. LinkedIn shows roughly the first two
lines before "see more", so those two lines carry the entire post.

RULES
- 150 to 220 words. Longer gets truncated and skipped.
- Open with the single most surprising or useful specific thing in the article -
  a distinction, a number, a failure mode. Never open with a question, never with
  "I've been thinking about", never with "In today's world".
- One idea per line. Blank line between them. Dense paragraphs do not get read.
- Use only claims the article actually makes. Invent nothing.
- Plain language. No emoji except at most one, and only if it earns its place.
- No em dashes. No "game-changer", "unlock", "dive into", "leverage", "in the
  ever-evolving landscape".
- No engagement bait: no "thoughts?", no "agree?", no "comment below".
- End with one line pointing to the full article{f", linking {link}" if link else ""}.
- At most 3 hashtags, on the final line, specific rather than generic.

TOPIC
{kw.get("primary_keyword", title)}

THE ARTICLE
{article[:8000]}

Return ONLY the post text, ready to paste."""

    post = _strip_em_dashes(call_claude(prompt, max_tokens=1500).strip())
    words = len(post.split())
    print(f"  LinkedIn post: {words} words")
    if words > 320:
        print("  Post came back long; LinkedIn will truncate it in the feed.")
    return post


def generate_video_script(title: str, article: str, research: dict) -> str:
    """A 2-3 minute video script drawn from the finished article.

    Written as timed beats with a visual note per beat, because the failure mode
    of an AI-written script is a spoken list of abstractions that no footage can
    carry. Naming the visual next to the line forces the script to be about
    something showable.
    """
    log("STEP 10", "Writing the video script")

    kw = research.get("keywords", {})

    prompt = f"""Write a 2 to 3 minute video script from the article below.

WHO IS SPEAKING
Imran Tauqir, an engineer who builds with AI agents and writes about it. He
speaks to other engineers as a peer. Confident and plain-spoken, never hyped.

LENGTH
380 to 440 words of narration. That is 2:30 to 3:00 at a natural speaking pace.
Count them. Going over means the video runs long and gets abandoned halfway.

STRUCTURE
Six beats, each with a timestamp, the narration, and the visual that carries it.

1. HOOK, about 15 seconds. Open on the single most surprising or most useful
   specific in the article - a distinction people get wrong, a number, a failure
   mode. No throat-clearing, no "in today's world", no question.
2. THE CORE IDEA, about 30 seconds. Define the thing plainly.
3. WHY IT MATTERS, about 20 seconds. The problem it solves, concretely.
4. THE SUBSTANCE, about 45 seconds. The part a viewer could not have guessed.
5. THE DISTINCTION, about 30 seconds. The comparison or contrast the article
   makes best. This is the line people will repeat, so make it quotable.
6. WHAT BREAKS, AND CLOSE, about 30 seconds. Failure modes, then a single line
   pointing at the full article.

RULES
- Every claim must already be in the article. Invent nothing.
- Write for the ear. Short sentences. Vary their length. Contractions are fine.
- Do NOT narrate a list of abstract nouns. If a beat covers several components,
  give each one a concrete consequence rather than a label.
- No em dashes, no "delve", "unlock", "leverage", "game-changer",
  "in the ever-evolving landscape".
- The visual note must describe something actually showable: a diagram that
  builds, text on screen, a screen recording, a comparison filling in. Never
  "stock footage of a developer typing", which illustrates nothing.

TOPIC
{kw.get("primary_keyword", title)}

THE ARTICLE
{article[:10000]}

FORMAT - return exactly this markdown and nothing else:

# {title} - video script

**Runtime:** <your estimate>  **Narration:** <word count> words

## 1. Hook (0:00-0:15)
**Visual:** <what is on screen>

<the narration>

## 2. ... (and so on through beat 6)

---

## Narration only

<every narration block, in order, with nothing else - ready to paste into a
teleprompter or a text-to-speech tool>"""

    script = _strip_em_dashes(call_claude(prompt, max_tokens=4000).strip())

    # The narration block is what determines runtime, so measure that, not the
    # whole document with its headings and visual notes.
    tail = script.split("## Narration only")
    narration = tail[-1] if len(tail) > 1 else script
    words = len(narration.split())
    print(f"  Video script: {words} words of narration, "
          f"about {words // 150}:{(words % 150) * 60 // 150:02d} at 150 wpm")
    if words > 520:
        print("  That will run past three minutes.")
    return script


def insert_answer_block(article: str, answer: str) -> str:
    """Put the answer immediately after the H1, before anything else."""
    if not answer:
        return article
    lines = article.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("# "):
            rest = lines[i + 1:]
            # Skip blank lines so the block lands tight against the title.
            while rest and not rest[0].strip():
                rest.pop(0)
            return "\n".join(lines[:i + 1] + ["", answer, ""] + rest)
    return answer + "\n\n" + article


def wrap_with_branding(article: str, edition: int) -> str:
    intro = AUTHOR_INTRO_TEMPLATE.format(edition=edition)
    sources = build_sources_section(article)
    return intro + article + sources + AUTHOR_CTA


def write_outputs(slug: str, article: str, meta: dict, images: dict, output_dir: Path, edition: int = 0):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Wrap with author branding + sources
    branded = wrap_with_branding(article, edition)

    # Markdown article
    md_path = output_dir / f"{slug}.md"
    md_path.write_text(branded, encoding="utf-8")
    print(f"\n  Article saved: {md_path}")

    # Meta JSON
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    faqs = parse_faq(branded)
    meta_payload = {
        "generated_at": generated_at,
        "seo_meta": meta,
        "canonical": f"{SITE_URL}/{slug}" if SITE_URL else None,
        "author": {"name": AUTHOR_NAME, "url": AUTHOR_URL},
        "faq_count": len(faqs),
        "sources": [
            {"publisher": publisher_name(u), "url": u} for u in extract_sources(branded)
        ],
        "images": [
            {"alt": v["alt"], "url": v["url"], "source": v["source"]}
            for v in images.values()
        ],
    }
    meta_path = output_dir / f"{slug}_meta.json"
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    print(f"  Meta JSON saved: {meta_path}")

    # HTML export
    html_path = _write_html(slug, branded, output_dir, meta=meta, images=images,
                            generated_at=generated_at)
    print(f"  HTML saved:     {html_path}")
    print(f"  Structured data: Article"
          + (f" + FAQPage ({len(faqs)} questions)" if faqs else " (no FAQ found)")
          + ("" if SITE_URL else ", no canonical (set SITE_URL)"))

    # DOCX export
    docx_path = _write_docx(slug, branded, output_dir)
    print(f"  DOCX saved:     {docx_path}")

    return md_path, meta_path


def _linkify(text: str) -> str:
    """Convert bare URLs in text to HTML anchor tags."""
    return re.sub(
        r'(?<!["\(])(https?://[^\s<>")\]]+)',
        r'<a href="\1">\1</a>',
        text,
    )


def _esc(text: str) -> str:
    """Escape a value going into an HTML attribute."""
    return (str(text).replace("&", "&amp;").replace('"', "&quot;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def _write_html(slug: str, branded: str, output_dir: Path, meta: dict | None = None,
                images: dict | None = None, generated_at: str = "") -> Path:
    try:
        import markdown as md_lib
    except ImportError:
        return None

    def replace_image_block(m):
        alt, img_url = m.group(1), m.group(2)
        src_txt = m.group(3).strip().lstrip('*Source:').strip().rstrip('*').strip()
        return (
            f'<figure>'
            f'<img src="{img_url}" alt="{alt}" style="max-width:100%;height:auto;">'
            f'<figcaption style="font-size:0.8em;color:#666;">'
            f'Source: {src_txt} &nbsp;|&nbsp; '
            f'<a href="{img_url}" style="color:#3366cc;word-break:break-all;">{img_url}</a>'
            f'</figcaption></figure>'
        )

    src_patched = re.sub(
        r'!\[([^\]]*)\]\(([^\)]+)\)\n\*Source:([^\n]+)\*',
        replace_image_block,
        branded,
    )
    html_body = md_lib.markdown(src_patched, extensions=['tables', 'fenced_code'])
    html_body = _linkify(html_body)

    canonical = f"{SITE_URL}/{slug}" if SITE_URL else ""
    seo_title = (meta or {}).get("title") or slug.replace("-", " ").title()
    description = (meta or {}).get("description") or ""
    og_image = next((v["url"] for v in (images or {}).values() if v.get("url")), "")

    head = [
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_esc(seo_title)}</title>",
    ]
    if description:
        head.append(f'<meta name="description" content="{_esc(description)}">')
    head.append(f'<meta name="author" content="{_esc(AUTHOR_NAME)}">')
    if canonical:
        head.append(f'<link rel="canonical" href="{_esc(canonical)}">')
    head += [
        '<meta property="og:type" content="article">',
        f'<meta property="og:title" content="{_esc(seo_title)}">',
    ]
    if description:
        head.append(f'<meta property="og:description" content="{_esc(description)}">')
    if canonical:
        head.append(f'<meta property="og:url" content="{_esc(canonical)}">')
    if og_image:
        head.append(f'<meta property="og:image" content="{_esc(og_image)}">')
    if generated_at:
        head.append(f'<meta property="article:published_time" content="{_esc(generated_at)}">')
    head.append(f'<meta property="article:author" content="{_esc(AUTHOR_NAME)}">')
    head.append('<meta name="twitter:card" content="summary_large_image">')
    head.append(f'<meta name="twitter:title" content="{_esc(seo_title)}">')
    if description:
        head.append(f'<meta name="twitter:description" content="{_esc(description)}">')
    if og_image:
        head.append(f'<meta name="twitter:image" content="{_esc(og_image)}">')

    jsonld = build_jsonld(slug, branded, meta or {}, images or {}, generated_at)
    head.append(f'<script type="application/ld+json">\n{jsonld}\n</script>')
    head_html = "\n".join(head)

    full_html = f"""<!DOCTYPE html>
<html lang="en"><head>
{head_html}
<style>
  body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; line-height: 1.7; color: #222; }}
  h1,h2,h3 {{ color: #111; }}
  a {{ color: #3366cc; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; }}
  th {{ background: #f4f4f4; }}
  blockquote {{ border-left: 4px solid #ccc; margin: 0; padding: 0.5em 1em; color: #555; }}
  figure {{ margin: 1.5em 0; }}
  figcaption {{ margin-top: 6px; }}
  code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }}
  ol li {{ margin-bottom: 4px; word-break: break-all; }}
</style>
</head><body>
{html_body}
</body></html>"""

    html_path = output_dir / f"{slug}.html"
    html_path.write_text(full_html, encoding="utf-8")
    return html_path


def _write_docx(slug: str, branded: str, output_dir: Path) -> Path:
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor, Inches
        from docx.oxml.ns import qn
        import io, requests as req
    except ImportError:
        return None

    doc = Document()
    for s in doc.styles:
        try: s.font.name = 'Arial'
        except: pass
    for section in doc.sections:
        section.left_margin = section.right_margin = Inches(1.2)
        section.top_margin = section.bottom_margin = Inches(1)

    def set_arial(run, size=None):
        run.font.name = 'Arial'
        rPr = run._r.get_or_add_rPr()
        rFonts = rPr.find(qn('w:rFonts'))
        if rFonts is None:
            from docx.oxml import OxmlElement
            rFonts = OxmlElement('w:rFonts'); rPr.insert(0, rFonts)
        for attr in (qn('w:ascii'), qn('w:hAnsi'), qn('w:cs')):
            rFonts.set(attr, 'Arial')
        if size: run.font.size = size

    def add_inline(para, text):
        for part in re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)', text):
            if part.startswith('**') and part.endswith('**'):
                r = para.add_run(part[2:-2]); r.bold = True; set_arial(r)
            elif part.startswith('*') and part.endswith('*'):
                r = para.add_run(part[1:-1]); r.italic = True; set_arial(r)
            elif part.startswith('`') and part.endswith('`'):
                r = para.add_run(part[1:-1]); set_arial(r, Pt(10))
            else:
                r = para.add_run(part); set_arial(r)

    def embed_image(img_url, alt, src_txt):
        try:
            resp = req.get(img_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            if "image" in resp.headers.get("content-type","") and len(resp.content) > 1000:
                doc.add_paragraph().add_run().add_picture(io.BytesIO(resp.content), width=Inches(5.5))
                cap = doc.add_paragraph()
                cap.paragraph_format.space_after = Pt(12)
                r1 = cap.add_run(f"Source: {src_txt}  |  "); r1.italic = True
                r1.font.color.rgb = RGBColor(0x55,0x55,0x55); set_arial(r1, Pt(9))
                r2 = cap.add_run(img_url); r2.font.color.rgb = RGBColor(0x33,0x66,0xCC); set_arial(r2, Pt(9))
                return
        except: pass
        p = doc.add_paragraph()
        r = p.add_run(f"[ IMAGE: {alt} ]"); r.bold = True
        r.font.color.rgb = RGBColor(0x33,0x66,0xCC); set_arial(r, Pt(10))
        cap = doc.add_paragraph(); cap.paragraph_format.space_after = Pt(10)
        r1 = cap.add_run(f"Source: {src_txt}  |  "); r1.italic = True
        r1.font.color.rgb = RGBColor(0x55,0x55,0x55); set_arial(r1, Pt(9))
        r2 = cap.add_run(img_url); r2.font.color.rgb = RGBColor(0x33,0x66,0xCC); set_arial(r2, Pt(9))

    lines = branded.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r'^-{3,}$', line.strip()): i += 1; continue
        img_m = re.match(r'!\[([^\]]*)\]\(([^\)]+)\)', line.strip())
        if img_m:
            alt, url = img_m.group(1), img_m.group(2)
            src_txt = url
            if i+1 < len(lines) and lines[i+1].strip().startswith('*Source:'):
                src_txt = re.sub(r'^\*Source:\s*', '', lines[i+1].strip()).rstrip('*'); i += 1
            embed_image(url, alt, src_txt); i += 1; continue
        if line.strip().startswith('*Source:'): i += 1; continue
        if line.startswith('*') and line.endswith('*') and not line.startswith('**'):
            p = doc.add_paragraph(); r = p.add_run(line.strip('*')); r.italic = True; set_arial(r); i += 1; continue
        if line.startswith('# ') and not line.startswith('## '):
            h = doc.add_heading(line[2:], level=1); [set_arial(r) for r in h.runs]; i += 1; continue
        if line.startswith('## '):
            h = doc.add_heading(line[3:], level=2); [set_arial(r) for r in h.runs]; i += 1; continue
        if line.startswith('### '):
            h = doc.add_heading(line[4:], level=3); [set_arial(r) for r in h.runs]; i += 1; continue
        if line.startswith('|'):
            tbl_lines = []
            while i < len(lines) and lines[i].startswith('|'):
                if not re.match(r'^\|[-| :]+\|$', lines[i]): tbl_lines.append(lines[i])
                i += 1
            if tbl_lines:
                headers = [c.strip() for c in tbl_lines[0].strip('|').split('|')]
                tbl = doc.add_table(rows=1, cols=len(headers)); tbl.style = 'Table Grid'
                for j, h in enumerate(headers):
                    cell = tbl.rows[0].cells[j]; cell.text = ''
                    r = cell.paragraphs[0].add_run(h); r.bold = True; set_arial(r)
                for rl in tbl_lines[1:]:
                    cells = [c.strip() for c in rl.strip('|').split('|')]
                    rc = tbl.add_row().cells
                    for j, c in enumerate(cells[:len(headers)]): rc[j].text = c.replace('**','')
                doc.add_paragraph()
            continue
        if line.startswith('> '): p = doc.add_paragraph(style='Quote'); add_inline(p, line[2:]); i += 1; continue
        if line.startswith('- '): p = doc.add_paragraph(style='List Bullet'); add_inline(p, line[2:]); i += 1; continue
        if line.strip() == '': i += 1; continue
        p = doc.add_paragraph(); add_inline(p, line); p.paragraph_format.space_after = Pt(8); i += 1

    docx_path = output_dir / f"{slug}.docx"
    doc.save(docx_path)
    return docx_path


# ---------------------------------------------------------------------------
# Step 6.5: Verify -> Rebut -> Judge
#
# An independent auditor reads the finished article. Anything it flags goes
# back to the writer, which may accept the finding or dispute it. Disputes are
# settled by a third model that sees both sides and rules. Only findings that
# survive that process are applied.
# ---------------------------------------------------------------------------

VERIFY_SYSTEM = (
    "You are an independent editorial auditor. You did not write this article and you "
    "owe its author nothing. Find what is actually wrong with it: claims stated as fact "
    "without support, structure that drifted from the brief, AI writing tells that "
    "survived editing, coverage the brief required and the draft skipped. "
    "Do not pad the list with nitpicks - a short list of real problems beats a long list "
    "of style opinions. Report strict JSON only."
)

WRITER_SYSTEM = (
    "You are the writer who produced this article. An auditor has raised findings against "
    "it. Concede the ones that are right - defensiveness costs you nothing here and helps "
    "no one. Dispute only where the auditor is factually mistaken, misread the article, or "
    "is asserting a style preference as an error. Report strict JSON only."
)

JUDGE_SYSTEM = (
    "You are the deciding editor. An auditor and a writer disagree about specific findings "
    "on an article. You see the article, the finding, and the writer's response. Rule on "
    "each one. You are not splitting the difference and you are not deferring to either "
    "party - decide which reading of the text is correct. Report strict JSON only."
)


def verify_content(article: str, outline: str, key_takeaways: str, research: dict,
                   round_no: int = 1, agents: dict | None = None) -> dict:
    """Independent audit of the finished article. Returns {verdict, scores, issues}."""
    agents = agents or resolve_agents()
    auditor = agents["auditor"]
    log("STEP 6.5", f"Verification pass (round {round_no}) "
                    f"- auditor {auditor['model']} ({auditor['provider']})")

    kw = research.get("keywords", {})
    serp_context = research.get("serp_context") or "(no live SERP data was available)"

    prompt = f"""Audit the article below against the brief it was written from.

BRIEF IT WAS WRITTEN FROM
Primary keyword: {kw.get("primary_keyword", "")}
Secondary keywords: {", ".join(kw.get("secondary_keywords", []))}
Search intent: {research.get("search_intent")}
Target audience: {research.get("target_audience")}
Article goal: {research.get("article_goal")}

Required takeaways:
{key_takeaways}

Outline it was told to follow:
{outline}

What is currently ranking for this keyword:
{serp_context}

WHAT TO CHECK
1. FACTUAL - Any claim presented as fact with no citation and no way for a reader to
   check it. Numbers, dates, prices, version names, and company claims are the highest
   risk. Flag anything you believe is outdated or wrong, and say why.
2. CITATIONS - "Source: <url>" lines that do not plausibly support the sentence they
   follow, or a bare domain used as if it were evidence.
3. STRUCTURE - Sections in the outline that are missing, merged, or renamed beyond
   recognition. Count the [IMAGE: ... | Query: ...] markers still present and compare
   with the outline.
4. AI TELLS - Patterns that survived editing: significance inflation, vague attribution
   ("experts say"), participle padding, title case headings, em dashes inside headings,
   formulaic conclusions.
5. COVERAGE - Required takeaways that never actually land in the body, or a subtopic the
   ranking pages all cover and this article does not.
6. CONTRADICTION - Places where the article states two incompatible things.

ARTICLE
{article}

Return ONLY valid JSON in exactly this shape:
{{
  "verdict": "pass" or "revise",
  "scores": {{
    "factual_support": 0-10,
    "outline_fidelity": 0-10,
    "human_voice": 0-10,
    "coverage": 0-10
  }},
  "issues": [
    {{
      "id": "i1",
      "category": "factual|citation|structure|ai_tell|coverage|contradiction",
      "severity": "high|medium|low",
      "quote": "<the exact phrase or heading from the article, under 15 words>",
      "problem": "<what is wrong, one sentence>",
      "fix": "<the specific change you want, one sentence>"
    }}
  ]
}}

Use "pass" only when there is nothing above low severity. Number the ids i1, i2, i3 in
order. Return at most 12 issues, most severe first."""

    response = call_agent(auditor, prompt, system=VERIFY_SYSTEM, max_tokens=16000)
    report = extract_json(response)

    issues = report.get("issues", []) or []

    # The loop that follows keys findings by id, so a finding the auditor left
    # unnumbered - or numbered the same as an earlier one - would vanish before
    # anyone argued about it. Give those a fresh id instead.
    seen = set()
    for issue in issues:
        iid = issue.get("id")
        if not iid or iid in seen:
            n = 1
            while f"x{n}" in seen:
                n += 1
            iid = f"x{n}"
            issue["id"] = iid
        seen.add(iid)

    scores = report.get("scores", {}) or {}
    if scores:
        print("  Scores: " + ", ".join(f"{k}={v}" for k, v in scores.items()))
    counts = {}
    for issue in issues:
        sev = issue.get("severity", "low")
        counts[sev] = counts.get(sev, 0) + 1
    summary = ", ".join(f"{n} {sev}" for sev, n in counts.items()) or "none"
    print(f"  Verdict: {report.get('verdict', 'revise')} ({summary})")
    for issue in issues:
        print(f"    [{issue.get('severity','?')}] {issue.get('id','?')} "
              f"{issue.get('category','?')}: {issue.get('problem','')[:110]}")
    return report


def writer_rebuttal(article: str, issues: list) -> dict:
    """Give the writer a right of reply. Returns {responses: [{id, stance, reason}]}."""
    log("STEP 6.5", f"Writer responding to {len(issues)} finding(s)")

    issue_block = json.dumps(
        [{k: i.get(k) for k in ("id", "category", "severity", "quote", "problem", "fix")}
         for i in issues],
        indent=2,
    )

    prompt = f"""An auditor raised these findings against your article.

FINDINGS
{issue_block}

YOUR ARTICLE
{article}

For each finding, decide:
- "accept" - the auditor is right, the change should be made.
- "dispute" - the auditor is wrong. Only use this when you can point to something concrete:
  the quoted text does not say what the auditor claims, the claim IS cited elsewhere in the
  article, the structure change was required by the brief, or the auditor is calling a
  deliberate stylistic choice an error.

A dispute with no concrete reason will be overruled, so do not dispute to save face.

Return ONLY valid JSON:
{{
  "responses": [
    {{"id": "i1", "stance": "accept" or "dispute", "reason": "<one sentence>"}}
  ]
}}

Include exactly one response per finding id."""

    response = call_claude(prompt, system=WRITER_SYSTEM, max_tokens=8000)
    result = extract_json(response)

    responses = result.get("responses", []) or []
    n_disputed = sum(1 for r in responses if r.get("stance") == "dispute")
    print(f"  Writer accepted {len(responses) - n_disputed}, disputed {n_disputed}.")
    for r in responses:
        if r.get("stance") == "dispute":
            print(f"    disputes {r.get('id')}: {r.get('reason','')[:110]}")
    return result


def judge_disputes(article: str, disputes: list, agents: dict | None = None) -> dict:
    """Settle contested findings with a third model. Returns {rulings: [...]}."""
    agents = agents or resolve_agents()
    judge = agents["judge"]
    log("STEP 6.5", f"Escalating {len(disputes)} dispute(s) to judge "
                    f"({judge['model']} / {judge['provider']})")

    case_block = json.dumps(disputes, indent=2)

    prompt = f"""An auditor and the writer disagree about the findings below. Rule on each.

CONTESTED FINDINGS
Each entry has the auditor's finding and the writer's reason for disputing it.
{case_block}

THE ARTICLE IN FULL
{article}

For each finding, read the article text yourself and decide who is right:
- "uphold" - the auditor's finding stands and the fix should be applied.
- "overrule" - the writer is right and the article should be left alone.

Judge the substance, not the confidence of either side. If the disputed text is a matter
of taste rather than accuracy or structure, overrule. If the writer's reason does not
survive a look at the actual text, uphold.

Return ONLY valid JSON:
{{
  "rulings": [
    {{"id": "i1", "ruling": "uphold" or "overrule", "reasoning": "<one sentence>"}}
  ]
}}

Include exactly one ruling per contested finding."""

    # call_agent carries the fallback: a judge outage must not destroy an article
    # that already cost a dozen calls to produce.
    response = call_agent(judge, prompt, system=JUDGE_SYSTEM, max_tokens=8000)
    result = extract_json(response)

    for r in result.get("rulings", []) or []:
        print(f"    {r.get('ruling','?').upper():8} {r.get('id','?')}: "
              f"{r.get('reasoning','')[:110]}")
    return result


def apply_fixes(article: str, upheld: list) -> str:
    """Rewrite the article to address only the findings that survived."""
    log("STEP 6.5", f"Applying {len(upheld)} upheld finding(s)")

    fix_block = "\n".join(
        f"- [{i.get('severity','?')}] {i.get('quote','')}\n"
        f"  Problem: {i.get('problem','')}\n"
        f"  Fix: {i.get('fix','')}"
        for i in upheld
    )

    prompt = f"""Revise the article to address the findings below. Change nothing else.

FINDINGS TO ADDRESS
{fix_block}

STRUCTURAL CONSTRAINTS (never break these):
- Preserve ALL markdown headings unless a finding explicitly asks you to change one
- Preserve ALL [IMAGE: alt text | Query: ...] markers exactly
- Preserve ALL "Source: ..." citations except where a finding says one is wrong
- Keep paragraphs to 3-4 sentences
- Do NOT rewrite passages no finding mentions
- Do NOT invent a citation. If a finding says a claim is unsupported and you have no real
  source, soften the claim or cut it instead of attaching a made-up URL.

ARTICLE
{article}

Return ONLY the revised article. No preamble, no list of what you changed."""

    revised = call_claude(prompt, max_tokens=16000)
    revised = _strip_em_dashes(revised)
    print(f"  Revised ({len(revised.split())} words).")
    return revised


def verification_loop(article: str, outline: str, key_takeaways: str, research: dict,
                      max_rounds: int = 2, record: dict | None = None) -> str:
    """Audit, argue, judge, fix - up to max_rounds times or until the audit passes.

    Pass a dict as `record` to keep the argument itself. Everything the three
    agents say is otherwise printed once and lost, which leaves you with a
    changed article and no way to see who changed it or why.
    """
    # Resolve the roster once so every round is argued by the same three agents.
    agents = resolve_agents()
    log("STEP 6.5", "Agents\n  " + describe_agents(agents))
    if len({a["provider"] for a in agents.values()}) == 1:
        print("  Warning: all three roles are on one vendor. The audit is weaker "
              "than it looks - set OPENAI_API_KEY or GEMINI_API_KEY.")

    if record is not None:
        record["agents"] = agents
        record["rounds"] = []
        record["started_at"] = datetime.now().isoformat()

    def close(outcome: str) -> str:
        if record is not None:
            record["outcome"] = outcome
            record["finished_at"] = datetime.now().isoformat()
        return article

    for round_no in range(1, max_rounds + 1):
        # The article at this point has cost a dozen calls to produce. A stage
        # whose whole job is to check it must not be the thing that destroys it,
        # so a failed round ends verification and keeps the text as it stands.
        try:
            report = verify_content(article, outline, key_takeaways, research,
                                    round_no, agents=agents)
        except ClaudeError as e:
            reason = " ".join(str(e).split())[:200]
            print(f"  Verification could not run: {reason}")
            print(f"  Keeping the article as written. It was not checked.")
            if record is not None:
                record["error"] = reason
            return close(f"verification failed on round {round_no}")

        issues = report.get("issues", []) or []

        entry = {"round": round_no, "verdict": report.get("verdict"),
                 "scores": report.get("scores", {}), "issues": issues,
                 "responses": [], "rulings": [], "applied": []}
        if record is not None:
            record["rounds"].append(entry)

        if report.get("verdict") == "pass" or not issues:
            print(f"  Verification passed on round {round_no}. No changes applied.")
            return close(f"passed on round {round_no}")

        by_id = {i.get("id"): i for i in issues if i.get("id")}
        try:
            rebuttal = writer_rebuttal(article, issues)
        except ClaudeError as e:
            print(f"  The writer could not answer: {' '.join(str(e).split())[:160]}")
            print("  Applying every finding unanswered, as silence implies.")
            rebuttal = {"responses": []}
        entry["responses"] = rebuttal.get("responses", []) or []

        upheld, disputed = [], []
        answered = set()
        for r in entry["responses"]:
            issue = by_id.get(r.get("id"))
            if not issue:
                continue
            answered.add(r.get("id"))
            if r.get("stance") == "dispute":
                disputed.append({**issue, "writer_reason": r.get("reason", "")})
            else:
                upheld.append(issue)
        # A finding the writer never answered is not a dispute - apply it.
        unanswered = [i for k, i in by_id.items() if k not in answered]
        entry["unanswered"] = [i.get("id") for i in unanswered]
        upheld += unanswered

        if disputed:
            try:
                rulings = judge_disputes(article, disputed, agents=agents)
            except ClaudeError as e:
                print(f"  The judge could not rule: {' '.join(str(e).split())[:160]}")
                print("  Unruled disputes leave the text standing.")
                rulings = {"rulings": []}
            entry["rulings"] = rulings.get("rulings", []) or []
            ruled = {r.get("id"): r.get("ruling") for r in entry["rulings"]}
            for issue in disputed:
                # An unruled dispute defaults to the writer keeping the text.
                if ruled.get(issue.get("id")) == "uphold":
                    upheld.append(issue)
                elif issue.get("id") not in ruled:
                    entry.setdefault("unruled", []).append(issue.get("id"))

        entry["applied"] = [i.get("id") for i in upheld]

        if not upheld:
            print("  Every finding was overruled. Article left as written.")
            return close(f"every finding overruled on round {round_no}")

        try:
            article = apply_fixes(article, upheld)
        except ClaudeError as e:
            reason = " ".join(str(e).split())[:200]
            print(f"  The fix pass failed: {reason}")
            print("  Keeping the last good version of the article.")
            if record is not None:
                record["error"] = reason
            return close(f"fix pass failed on round {round_no}")
        entry["words_after_fix"] = len(article.split())

    print(f"  Reached the {max_rounds}-round limit. Using the latest revision.")
    return close(f"hit the {max_rounds}-round limit")


# ---------------------------------------------------------------------------
# Step 6.5 review report
# ---------------------------------------------------------------------------

def format_review(record: dict, title: str = "") -> str:
    """The argument as a readable document: who said what, and what survived."""
    if not record.get("rounds"):
        return "# Verification review\n\nThe audit did not run.\n"

    heading = f"# Verification review: {title}" if title else "# Verification review"
    out = [heading, "", "## Who argued", "", "| Role | Model | Vendor |", "|---|---|---|"]
    for role, a in (record.get("agents") or {}).items():
        out.append(f"| {role} | `{a['model']}` | {a['provider']} |")
    out += ["", f"Outcome: **{record.get('outcome', 'unknown')}**", ""]

    for entry in record["rounds"]:
        out += [f"## Round {entry['round']} - verdict: {entry.get('verdict', '?')}", ""]
        scores = entry.get("scores") or {}
        if scores:
            out.append("| " + " | ".join(scores) + " |")
            out.append("|" + "---|" * len(scores))
            out.append("| " + " | ".join(str(v) for v in scores.values()) + " |")
            out.append("")

        issues = entry.get("issues") or []
        if not issues:
            out += ["No findings.", ""]
            continue

        stances = {r.get("id"): r.get("stance") for r in entry.get("responses") or []}
        reasons = {r.get("id"): r.get("reason", "") for r in entry.get("responses") or []}
        rulings = {r.get("id"): r.get("ruling") for r in entry.get("rulings") or []}
        why = {r.get("id"): r.get("reasoning", "") for r in entry.get("rulings") or []}
        applied = set(entry.get("applied") or [])

        for issue in issues:
            iid = issue.get("id")
            stance = stances.get(iid)
            out += [f"### {iid} - {issue.get('category', '?')} "
                    f"({issue.get('severity', '?')})", ""]
            if issue.get("quote"):
                out += [f'> {issue["quote"]}', ""]
            out.append(f"**Auditor:** {issue.get('problem', '')} "
                       f"_Wants:_ {issue.get('fix', '')}")
            if stance is None:
                out.append("**Writer:** did not respond - counted as accepted.")
            else:
                label = {"accept": "accepted", "dispute": "disputed"}.get(stance, stance)
                out.append(f"**Writer:** {label}. {reasons.get(iid, '')}")
            if iid in rulings:
                out.append(f"**Judge:** {rulings[iid]}. {why.get(iid, '')}")
            elif stance == "dispute":
                out.append("**Judge:** no ruling returned - the text stands.")
            out += ["", f"**Result:** {'applied' if iid in applied else 'not applied'}", ""]

    return "\n".join(out) + "\n"


def write_review(slug: str, record: dict, output_dir: Path, title: str = "") -> tuple:
    """Save the argument next to the article, as JSON and as something readable."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{slug}_review.json"
    md_path = output_dir / f"{slug}_review.md"
    json_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    md_path.write_text(format_review(record, title), encoding="utf-8")
    return md_path, json_path


def review_stats(record: dict) -> dict:
    """Headline numbers for the run summary."""
    stats = dict(rounds=0, raised=0, accepted=0, disputed=0, upheld=0,
                 overruled=0, applied=0)
    for entry in record.get("rounds", []):
        stats["rounds"] += 1
        stats["raised"] += len(entry.get("issues") or [])
        for r in entry.get("responses") or []:
            key = "disputed" if r.get("stance") == "dispute" else "accepted"
            stats[key] += 1
        for r in entry.get("rulings") or []:
            stats["upheld" if r.get("ruling") == "uphold" else "overruled"] += 1
        stats["applied"] += len(entry.get("applied") or [])
    return stats


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run(title: str, keywords: str, output_dir: Path, edition: int = 0, intent: str = "",
        verify: bool = True, verify_rounds: int = 2, words: str = "default",
        linkedin: bool = False, video: bool = False):
    profile = length_profile(words)
    # If intent is given and no explicit keywords, derive optimized search keywords
    if intent and not keywords:
        log("INTENT", "Extracting search keywords from intent...")
        keywords = extract_search_query(title, intent)
        print(f"  Derived keywords: {keywords}")
    elif not keywords:
        keywords = title

    print(f"\n{'='*60}")
    print(f"SEO Article Writer")
    print(f"Topic   : {title}")
    print(f"Keywords: {keywords}")
    if intent:
        print(f"Intent  : {intent[:80]}{'...' if len(intent) > 80 else ''}")
    print(f"Length  : {profile['label']}")
    print(f"Output  : {output_dir}")
    print(f"{'='*60}")

    # Step 1: SERP Research
    research = serp_research(title, keywords, intent=intent)

    # Step 2: Refine Title
    refined_title = refine_title(title, keywords, research, intent=intent)

    # Step 3: Key Takeaways
    key_takeaways = generate_key_takeaways(refined_title, keywords, research)

    # Step 4: Outline
    outline = generate_outline(refined_title, keywords, research, key_takeaways,
                               profile=profile)

    # Step 5: Write Content
    content = write_content(refined_title, keywords, outline, research, key_takeaways,
                            profile=profile)

    # Step 6: Humanize
    humanized = humanize_content(content)

    # Step 6.4: Direct-answer block. Inserted before verification, so the auditor
    # checks it against the brief like any other passage.
    humanized = insert_answer_block(
        humanized, generate_answer_block(refined_title, humanized, research)
    )

    # Step 6.5: Verify -> rebut -> judge -> fix
    record = {}
    if verify:
        humanized = verification_loop(
            humanized, outline, key_takeaways, research, max_rounds=verify_rounds,
            record=record,
        )
    else:
        log("STEP 6.5", "Verification skipped (--no-verify)")

    # Step 7: Meta
    meta = generate_meta(refined_title, keywords, humanized)

    # Step 8: Image Search
    images = search_images(humanized, research)

    # Inject images into article
    final_article = inject_images(humanized, images)

    # Write outputs
    slug = slugify(refined_title)
    md_path, meta_path = write_outputs(slug, final_article, meta, images, output_dir, edition=edition)

    review_path = None
    if record.get("rounds"):
        review_path, _ = write_review(slug, record, output_dir, refined_title)
    linkedin_path = None
    if linkedin:
        post = generate_linkedin_post(refined_title, humanized, research)
        linkedin_path = output_dir / f"{slug}_linkedin.md"
        linkedin_path.write_text(post, encoding="utf-8")

    video_path = None
    if video:
        script = generate_video_script(refined_title, humanized, research)
        video_path = output_dir / f"{slug}_video.md"
        video_path.write_text(script, encoding="utf-8")

    usage_path = write_usage(slug, output_dir, refined_title)

    # Summary
    word_count = len(final_article.split())
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Title      : {refined_title}")
    print(f"  Word count : {word_count:,}")
    print(f"  Images     : {len(images)}")
    print(f"  Article    : {md_path}")
    print(f"  Meta JSON  : {meta_path}")
    if review_path:
        s = review_stats(record)
        print(f"  Review     : {review_path}")
        print(f"  Argument   : {s['raised']} raised, {s['disputed']} disputed, "
              f"{s['upheld']} upheld, {s['overruled']} overruled, "
              f"{s['applied']} applied over {s['rounds']} round(s)")
    if linkedin_path:
        print(f"  LinkedIn   : {linkedin_path}")
    if video_path:
        print(f"  Video      : {video_path}")
    print(f"  Usage JSON : {usage_path}")
    print_usage_summary()
    print(f"{'='*60}\n")


def audit_only(path: Path, output_dir: Path, topic: str = "", intent: str = "",
               verify_rounds: int = 2, apply: bool = False):
    """Point the three agents at a document you already have.

    The auditor works by comparing an article against the brief it was written
    from, and a file you hand it has no brief. So one call reconstructs the brief
    the document appears to be written to - its own outline and the takeaways it
    seems to promise - and the agents argue against that.
    """
    article = path.read_text(encoding="utf-8")
    title = topic or path.stem.replace("-", " ")

    print(f"\n{'='*60}")
    print(f"Audit only - no article is written")
    print(f"Document: {path}  ({len(article.split()):,} words)")
    print(f"Rounds  : {verify_rounds}   Apply fixes: {'yes' if apply else 'no'}")
    print(f"{'='*60}")

    log("BRIEF", "Reconstructing the brief this document was written to...")
    brief_prompt = f"""Read the document and infer the brief it appears to have been
written to. Do not judge it yet - only describe what it is trying to do.

{f"The author says the goal was: {intent}" if intent else ""}

Return ONLY valid JSON:
{{
  "primary_keyword": "<the phrase this is optimised for>",
  "secondary_keywords": ["<up to 6>"],
  "search_intent": "<informational | commercial | transactional | navigational>",
  "target_audience": "<one line>",
  "article_goal": "<one line>",
  "outline": "<the document's actual heading structure, as markdown headings>",
  "key_takeaways": "<the points it promises the reader, as a markdown list>"
}}

DOCUMENT
{article}"""
    brief = extract_json(call_claude(brief_prompt, max_tokens=4000))

    research = {
        "keywords": {"primary_keyword": brief.get("primary_keyword", title),
                     "secondary_keywords": brief.get("secondary_keywords", [])},
        "search_intent": brief.get("search_intent", ""),
        "target_audience": brief.get("target_audience", ""),
        "article_goal": brief.get("article_goal", ""),
        "serp_context": "(no live SERP data - this document was audited, not researched)",
    }
    print(f"  Keyword : {research['keywords']['primary_keyword']}")
    print(f"  Audience: {research['target_audience']}")

    record = {}
    revised = verification_loop(article, brief.get("outline", ""),
                                brief.get("key_takeaways", ""), research,
                                max_rounds=verify_rounds, record=record)

    slug = slugify(title)
    review_path, json_path = write_review(slug, record, output_dir, title)

    revised_path = None
    if apply and revised != article:
        revised_path = output_dir / f"{slug}_revised.md"
        revised_path.write_text(revised, encoding="utf-8")

    s = review_stats(record)
    print(f"\n{'='*60}")
    print(f"AUDIT COMPLETE")
    print(f"  Rounds     : {s['rounds']}")
    print(f"  Raised     : {s['raised']}")
    print(f"  Accepted   : {s['accepted']}   Disputed: {s['disputed']}")
    print(f"  Upheld     : {s['upheld']}   Overruled: {s['overruled']}")
    print(f"  Applied    : {s['applied']}")
    print(f"  Review     : {review_path}")
    print(f"  Raw JSON   : {json_path}")
    if revised_path:
        print(f"  Revised    : {revised_path}")
    elif apply:
        print(f"  Revised    : nothing changed, no file written")
    print(f"  Usage JSON : {write_usage(slug, output_dir, title)}")
    print_usage_summary()
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a full SEO article with web-sourced images."
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default=None,
        help=('Article topic, e.g. "What is RAG in AI". Optional with --audit, '
              'where it only names the document being audited.'),
    )
    parser.add_argument(
        "--intent",
        default=None,
        help=(
            'Natural language description of what you want to achieve, e.g. '
            '"I want to explain to developers how ReAct agents work and why '
            'they are better than standard LLMs for tool use". '
            'Claude will extract the best search keywords from this description.'
        ),
    )
    parser.add_argument(
        "--keywords",
        default=None,
        help=(
            'Primary keyword(s) to target directly, e.g. "retrieval augmented generation". '
            'Use --intent instead for a richer, intent-driven search. '
            'If neither is provided, defaults to the topic.'
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory to save the article and meta JSON (default: ./output)",
    )
    parser.add_argument(
        "--edition",
        type=int,
        default=0,
        help="Newsletter edition number shown in the author intro (e.g. --edition 31)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help=(
            "Skip the Step 6.5 audit. Faster and cheaper, but nothing checks the "
            "article's claims or structure before it is written to disk."
        ),
    )
    parser.add_argument(
        "--verify-rounds",
        type=int,
        default=2,
        help="Maximum audit/fix rounds before accepting the article (default: 2)",
    )
    parser.add_argument(
        "--words",
        choices=sorted(LENGTH_PROFILES),
        default="default",
        help=(
            "Article length. Every structural number moves with it - sections, "
            "subsections, images and FAQ count - so a short article is short "
            "rather than cramped. Default is 2,500-3,500 words."
        ),
    )
    parser.add_argument(
        "--linkedin",
        action="store_true",
        help=("Also write a LinkedIn post from the finished article, saved as "
              "<slug>_linkedin.md"),
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help=("Also write a 2-3 minute video script from the finished article, "
              "with a visual note per beat, saved as <slug>_video.md"),
    )
    parser.add_argument(
        "--audit",
        metavar="FILE",
        default=None,
        help=(
            "Audit a document you already have instead of writing a new one. The "
            "three agents argue about your text and a review report is written; "
            "no article, images or meta are generated."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="With --audit, also save the revised document as <slug>_revised.md",
    )
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    if not args.audit and not args.topic:
        parser.error("a topic is required unless you pass --audit FILE")

    # --intent takes priority; --keywords is the legacy shorthand; topic is the fallback
    intent = args.intent or ""
    keywords = args.keywords or ("" if intent else args.topic)
    output_dir = Path(args.output_dir)

    try:
        if args.audit:
            doc = Path(args.audit)
            if not doc.exists():
                print(f"ERROR: no such file: {doc}", file=sys.stderr)
                sys.exit(1)
            audit_only(
                doc,
                output_dir=output_dir,
                topic=args.topic or "",
                intent=intent,
                verify_rounds=args.verify_rounds,
                apply=args.apply,
            )
            return
        run(
            title=args.topic,
            keywords=keywords,
            output_dir=output_dir,
            edition=args.edition,
            intent=intent,
            verify=not args.no_verify,
            verify_rounds=args.verify_rounds,
            words=args.words,
            linkedin=args.linkedin,
            video=args.video,
        )
    except ClaudeError as e:
        # Flattened to one line so the web UI, which reads the log line by line,
        # can surface the whole message as a single error.
        print(f"ERROR: {' '.join(str(e).split())}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("ERROR: Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        # Anything unhandled used to leave the web UI showing "exited with code
        # 1" and nothing else, because a raw traceback carries no ERROR: line
        # for the job runner to pick up. Print both: a one-line summary it can
        # surface, and the traceback underneath for whoever reads the log.
        import traceback
        print(f"ERROR: {type(e).__name__}: {' '.join(str(e).split())[:300]}",
              file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
