# What is an agent harness? — video script

**Runtime:** ~2 min 40 s at 150 wpm (405 words)
**Source:** `what-is-an-agent-harness-core-components-architecture-and-how-it-differs-from-ag.md`

---

### 1 — Hook (0:00–0:18)

Hi everyone, Imran here.

There's a word getting thrown around in every AI engineering thread right now,
and almost nobody uses it the same way twice. Agent harness. People use it
interchangeably with agent SDK. They're not the same thing, and the difference
is the reason most homegrown agents fall over in production.

### 2 — What it actually is (0:18–0:45)

An agent harness is the orchestration layer that wraps a language model with
tools, memory, sandboxes, and feedback loops. It turns a single-turn text
generator into something that can run a multi-step task on its own.

The name comes from software testing. A test harness doesn't write the code —
it runs it, captures the result, and checks it against what you expected. An
agent harness does exactly that around a model's output. It runs the action the
model proposed, watches what happened, and feeds that back before the next
decision.

### 3 — Why the model can't do this alone (0:45–1:05)

A raw LLM is single-turn. Prompt in, completion out, stop. It cannot open a
file, run a test, or retry something that failed. There is no mechanism inside
the model to act on its own output. Everything that makes an agent feel
autonomous lives outside the weights.

### 4 — The four components (1:05–1:45)

Every real harness converges on the same four pieces.

A control loop — plan, act, observe, repeat, until the task is done or fails
hard enough to stop.

Tool interfaces — structured function calls with real schemas, validated before
they run.

State and memory — because context windows are finite, and a long task will
bury the detail the model actually needs.

And an execution environment — a sandbox with real permissions, so a bad step
breaks something disposable instead of something that matters.

### 5 — Harness versus SDK (1:45–2:10)

Here's the distinction worth remembering. An SDK gives you primitives — the API
calls, the formatting, sometimes a session helper. It stops short of owning task
state at runtime.

A harness owns that state. It's the assembled system running on top of the SDK.
That's the whole difference, and it explains most of the confusion in developer
forums.

Claude Code and Codex are harnesses. LangChain and the Anthropic SDK are SDKs.
You build one with the other.

### 6 — What breaks, and close (2:10–2:40)

Three failure modes account for most of it. Thin error handling that lets
failures cascade. Memory leakage, where stale context quietly degrades planning
over a long session. And unbounded tool permissions.

All three are far cheaper to fix before production than after.

Full breakdown, including the architecture checklist, is in this week's edition.
Link below.

I'm Imran — see you next time.
