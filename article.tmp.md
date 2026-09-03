👋 Hi everyone, Imran here.
Welcome to Edition #0 of a newsletter that people around the world actually look forward to reading.
# What is an agent harness? Core components, architecture, and how it differs from agent SDKs

## Introduction

An **agent harness** is the orchestration layer that wraps a large language model with tools, memory, sandboxes, and feedback loops. It turns a single-turn text generator into something that can run multi-step tasks on its own. People in the field often mix this up with **agent SDK**, using both terms for the same thing, since the ecosystem is young and vendors sell overlapping capabilities under different names. An **AI agent harness** still occupies a different architectural layer than the SDKs and models it coordinates, and it's worth asking [what is an AI agent](#) in the first place before comparing the layers that support one.

This article works through that distinction using a comparison grounded in real production systems. We'll look at the components that make a harness functional (control loops, tool interfaces, memory systems, sandboxes), then explain how it differs from an SDK and from the narrower idea of a "code harness." We'll reference Claude Code and OpenAI Codex throughout, since many readers already use them, and we'll get into [Codex vs. Claude Code](#) directly in Section 6. By the end you'll have a concrete checklist for evaluating or building your own **agentic harness**, including guidance on when it makes sense to [build a custom agent harness](#) rather than adopt one off the shelf.


![Diagram of an AI agent harness wrapping an LLM with tools and memory](https://miro.medium.com/1*JO74jIwiGmZNZzDS_8K6AA.png)
*Source: Medium*


---

## 1. Defining the agent harness: beyond static scaffolding

People often treat the agent harness as static **scaffolding**, a fixed template that structures prompts and outputs. In practice it's a runtime system that governs execution and adapts based on intermediate results. Static scaffolding can't handle failure, replanning, or multi-step dependencies. A real harness is built to manage exactly those conditions.

### 1.1 What "harness" means in agentic AI systems

The term comes from software testing, where a **test harness** provides the infrastructure to run test cases, capture outputs, and check them against expected results. Agentic AI borrowed the idea because the problem is similar: you need infrastructure to execute actions, watch what happens, and decide what to do next. It's operational infrastructure around the model, separate from the model's own weights and training.

That borrowed meaning carries over almost exactly. A test harness doesn't write the code under test; it runs it, captures the result, and compares that result against an expectation. An agent harness plays the same role around a model's outputs: it runs the action the model proposed, captures what happened, and feeds that back so the next decision can account for it. The word "harness" itself is also a useful physical metaphor, a structure that holds something in place and directs its movement without dictating every step, which maps onto how these systems constrain an LLM's actions without hardcoding the exact sequence of steps it takes. Understanding the term this way helps explain why so many teams describe very different systems as "agent harnesses": the label refers to a role in the architecture, not a single fixed implementation.

### 1.2 Why LLMs alone can't act autonomously

Raw LLMs are single-turn systems. They take a prompt, produce a completion, and stop there. There's no built-in way to act on that output, check results, or keep working toward a goal. Multi-step execution needs something outside the model that can call tools, parse results, and decide what happens next. An LLM on its own can't open a file, run a test, or retry a failed operation.

### 1.3 The hidden distinction: runtime layer vs. underlying model

The harness runs as a layer separate from the model or the SDK used to build it. The model generates reasoning and text. The SDK gives you the primitives to call that model and format its output, and in some cases adds session or memory interfaces on top, but it generally stops short of owning task state at runtime. The harness assembles those primitives into a live, iterative system that actually manages that state, and that separation clears up most of the confusion seen in developer forums.

---

## 2. Core components of an agent harness

The harnesses examined in this article, including Claude Code and Codex, converge on a similar set of architectural pieces, even though implementations vary by vendor. They work together to turn a stateless model into something capable of sustained, goal-directed behavior. Looking at each one shows what separates a real harness from a prompt template with extra steps.

### 2.1 Control loop and planning logic

At the center of every agent harness is a **control loop**, an iterative plan-act-observe cycle that decides what happens next. The model proposes an action, the harness executes it, observes the result, and feeds that observation back into the model for the next step. This repeats until the task is done or something fails badly enough to stop it. Source: https://www.anthropic.com/engineering/building-effective-agents

### 2.2 Tool and function-calling interfaces

Harnesses expose outside capabilities through structured **tool calls**: function schemas with named parameters and expected return types. When the model requests a tool call, the harness validates it, runs the underlying function or API call, and returns the result to the model's context. This is how an agent reads files, queries databases, runs code, or calls third-party APIs without touching the system directly.

### 2.3 State, memory, and context management

LLMs have finite context windows, so this layer has to manage short-term working memory and longer-term state across a task. Short-term context usually holds the current conversation, recent tool outputs, and planning notes; persistent memory stores summaries, prior decisions, or project knowledge across sessions. Context bloat and stale state degrade planning quality over long sessions, since irrelevant history crowds out the details the model actually needs to make its next decision. Good memory management keeps context from overflowing while preserving continuity for longer, multi-step work.

Some harnesses solve this with rolling summarization: periodically compressing older turns into a condensed record so the working context stays small. Others use structured external memory: writing key facts or decisions to a file or database the agent can query on demand instead of keeping everything in the prompt. The choice matters more as tasks stretch across dozens of tool calls, where naive context accumulation eventually pushes out the information the model needs most.


![Flowchart of agent harness control loop with tool calls and memory](https://miro.medium.com/v2/resize:fit:1400/0*AqZXSZB2OmCmYW_6.png)
*Source: Medium*


---

## 3. Execution environments: sandboxes and safety layers

Beyond planning and memory, an agent harness needs a safe, isolated place for actions to happen. These execution environments keep agent actions, especially code execution, from damaging production systems.

### 3.1 Sandboxing for code execution and file systems

Coding agents run generated code inside **sandboxes** that isolate file system access, network calls, and process execution from the host, though the isolation mechanism varies by product and deployment mode. This lets the agent run tests, install dependencies, or edit files without putting the underlying infrastructure at risk. See our guide on [sandboxing techniques](#) for a deeper walkthrough of how these isolation boundaries are typically implemented.

### 3.2 Permissioning and guardrails

Production harnesses use granular permissioning: restricting which tools an agent can call, rate-limiting expensive operations, and adding human-approval gates for high-risk actions. These guardrails stop runaway tool usage, unauthorized data access, or destructive file operations. Some permission systems are designed to expand access over time as trust in an agent's behavior grows, though neither of the two systems discussed later in this article implements that kind of trust-based escalation; both instead rely on fixed, per-action permission gates.

Permissioning also has to account for the difference between reversible and irreversible actions. Reading a file or running a query in a sandboxed database carries little risk, while deleting production data or sending an external email does not. Harnesses that route irreversible actions through an explicit approval step, rather than treating all tool calls the same way, tend to fail more gracefully when the model misjudges a situation.

### 3.3 Validation and output verification

Before a task proceeds or closes out, the harness typically runs automated checks (unit tests, linters, schema validators) against the output. This gives an objective read on success instead of relying on the model's own judgment of its work. Validation matters most in coding harnesses, where a change can look plausible but be functionally broken.


![Sandboxed execution environment for AI coding agent](https://www.penligent.ai/hackinglabs/wp-content/uploads/2026/04/Sandboxes-for-Coding-Agents.png)
*Source: Penligent*


---

## 4. Feedback loops: the defining feature of true agentic harnesses

Feedback loops separate a genuine agentic harness from scripted automation. A scripted pipeline runs a fixed sequence no matter what happens. An agent harness watches results and adjusts. For a longer treatment of how to design these loops, see our guide on building feedback loops.

### 4.1 Error handling and self-correction

When a tool call fails or returns something unexpected, the harness has to catch it and decide whether to retry, change parameters, or try something else. This is what lets agents recover from network timeouts or malformed outputs without a human stepping in. Without it, agents stall, or one bad step cascades into several.

Effective error handling generally distinguishes between a few failure classes rather than treating every failure the same way. Transient failures, such as a network timeout or a rate-limited API call, are usually worth an automatic retry, sometimes with backoff, since the same action may simply succeed on a second attempt. Failures caused by a malformed request, like an invalid file path or a missing argument, call for the harness to adjust the parameters and try again rather than repeating the identical call. Failures that stem from a flawed underlying plan, where the action executed correctly but didn't move the task forward, are a signal that the harness needs to hand control back to the planning step rather than keep retrying the same action. Harnesses that only implement blind retries, without this kind of failure classification, tend to burn through time and API budget on errors that were never going to resolve on their own.

### 4.2 Result verification and reflection

Many harnesses build in reflection: prompting the model to critique its own output, or grading it automatically against defined criteria. This catches errors that pass initial execution but don't actually meet the task requirements. That matters most on open-ended tasks where success isn't a simple pass/fail check, such as writing a report or refactoring a module against a loosely specified goal.

### 4.3 Adaptive replanning based on execution outcomes

A clear sign of a mature harness is adaptive replanning: revising the plan mid-task based on what's happened so far. Instead of following a fixed script, the agent can drop an approach that isn't working and try something else toward the same goal. This adaptive behavior, not the raw language capability of the underlying model, is what makes an **autonomous agent** meaningfully different from a scripted macro.


![Feedback loop diagram showing error handling and self-correction in AI agents](https://miro.medium.com/v2/resize:fit:2000/1*T_cbAuHmCbyIvgfK78eWKA.png)
*Source: Towards AI*


---

## 5. Agent harness vs. agent SDK vs. code harness: a comparative framework

Harness, SDK, and code harness describe different layers of the agent development stack. They're not interchangeable products.

### 5.1 What an agent SDK provides

An **agent SDK** is a developer library with the building-block primitives for agent behavior: API wrappers for calling models, function-calling schemas, message formatting utilities, and sometimes basic tool-calling helpers. SDKs vary in how much runtime they provide. The OpenAI Agents SDK, for example, ships session/memory helpers, guardrails, and tracing, but it still leaves gaps like sandboxed execution and long-horizon replanning for developers to build themselves. Our agent SDK comparison guide breaks down how different SDKs handle these gaps in more detail. Source: https://platform.openai.com/docs/guides/agents

### 5.2 What a code harness specifically means

A **code harness** is narrower and task-specific: it's the execution and testing infrastructure coding agents use, running generated code, executing test suites, and checking outputs against expected results. It overlaps with the sandboxing and validation pieces described above, but it's scoped to code correctness rather than general orchestration. Code harnesses used by coding agents are typically a subset of a broader agent harness, though not every agent harness deals with code at all.

### 5.3 Comparison table: harness vs. SDK vs. code harness

| Dimension | Agent Harness | Agent SDK | Code Harness |
|---|---|---|---|
| **Primary Role** | Runtime orchestration of tools, memory, feedback loops | Building-block APIs/primitives for agent development | Execution/testing layer for code-generating agents |
| **Scope** | System-level, assembled architecture | Library-level, developer toolkit | Task-specific execution environment |
| **Manages State?** | Yes, actively | Partially, some SDKs offer session/memory primitives, but no runtime that owns task state | Partially (execution context only) |
| **Examples** | Claude Code, Codex agent runtime | OpenAI Agents SDK, LangChain, Anthropic SDK | Codex sandbox, coding-agent test runners |
| **Typical User** | System architects, ML engineers deploying agents | App developers building custom agents | Coding agent developers validating output |
| **Includes Feedback Loops?** | Yes, core feature | No, must be implemented separately | Yes, but scoped to code correctness |


![Comparison chart of agent harness, SDK, and code harness](https://miro.medium.com/v2/resize:fit:2000/1*OWn0y2HG8LzQwkcdnaljcQ.png)
*Source: Cobus Greyling - Medium*


---

## 6. Real-world implementations: Claude Code and Codex

The architecture makes more sense once you look at production systems people actually use.

### 6.1 Claude Code as an agent harness

Claude Code is a complete agent harness: planning logic, file system access, tool use, and permission-gated command execution combined into one coding assistant. By default it runs shell commands directly on the user's machine behind explicit permission prompts, and it can be configured to run inside a more restricted sandbox for additional isolation. It keeps context across multi-file changes and iterates on its own output based on test results and error messages. Anthropic documents this combination of components as enabling Claude Code to work through complex engineering tasks with limited human input; see our Claude Code case study for a closer look at how these pieces fit together in practice. Source: https://docs.anthropic.com/en/docs/claude-code/overview

### 6.2 OpenAI Codex and its execution architecture

OpenAI Codex ships in more than one form, and the execution model differs between them. The Codex CLI, run locally, defaults to OS-level sandboxing (Seatbelt on macOS, Landlock on Linux) rather than a container, isolating file system and network access at the operating system layer. Codex's cloud/agent variant, by contrast, runs tasks inside isolated cloud containers. In both cases the harness supports an iterative loop where the agent can run tests, see what fails, and revise its implementation. Source: https://developers.openai.com/codex/

This maps onto the core harness components above: control loop, tool interface, sandbox, feedback mechanism, all working together, though the specific isolation mechanism depends on which Codex surface is in use. Read OpenAI's Codex documentation for the current default sandboxing behavior on each surface, since these defaults are the kind of detail vendors adjust between releases. The general pattern, an OS-level sandbox for local, interactive use and a container for cloud-run, less-supervised tasks, mirrors a common tradeoff between developer-machine flexibility and stronger isolation for less-supervised execution.

The contrast with Claude Code is instructive. Codex's local CLI defaults to OS-level sandboxing with permission prompts for actions outside that boundary, similar in spirit to Claude Code's default of local execution behind explicit permission prompts, while Codex's cloud variant trades some of that host-level flexibility for the stronger default isolation of a container. Neither approach is strictly better; they represent different points on the same tradeoff between containment and context, and the right choice depends on whether the workload is an interactive local session or an unattended cloud task. Codex shows that a code-focused harness can still carry the full feedback and replanning behavior of a more general agent harness, regardless of which sandboxing layer is doing the isolating.

### 6.3 Lessons for building or evaluating your own harness

Three design properties recur in both systems: modular tool integration, hard execution boundaries, and step-level observability. Modular tool integration means the harness can add or swap capabilities (a new API, a different sandbox) without rewriting the control loop. Hard execution boundaries, whether a container, an OS-level sandbox, or a permission gate, keep a single bad action from cascading into system-wide damage. Step-level observability (logging every planning decision, tool call, and validation result) is what makes these systems debuggable when something goes wrong. Check any custom system against these same properties before it goes into production.


![Screenshot or architecture diagram of Claude Code agent workflow](https://miro.medium.com/v2/resize:fit:1200/1*XIe0AzU8UNuX0Wqco7ouZg.png)
*Source: Level Up Coding - Gitconnected*


---

## 7. How to evaluate or build an agent harness for production

Here's what actually matters for a build-vs-buy decision, beyond the component checklist.

### 7.1 Key architectural requirements checklist

A production-ready agent harness needs: a control loop capable of multi-step planning, a well-defined tool/function-calling interface, solid state and memory management across sessions, an isolated or permission-gated environment for risky actions, and a feedback mechanism for error handling and self-correction. Miss any one of these and you usually get an agent that either stalls on complex tasks or takes unsafe actions. When evaluating a third-party harness, confirm each piece actually exists rather than assuming it does.

Turning this into a working checklist means asking specific questions of any harness under review, rather than taking a vendor's architecture diagram at face value. For the control loop: does it visibly replan after a failed step, or does it just retry the same action? For the tool interface: are permissions scoped per tool, or is it all-or-nothing? For memory: does the system have an explicit strategy for long sessions, such as summarization or external storage, or does it just truncate older context when the window fills up? For the execution environment: is the isolation boundary a container, an OS-level sandbox, or nothing at all, and does that match the risk profile of the actions the agent can take? For feedback: are outputs checked against an objective test, or only against the model's own assessment of its work? Running through these questions on any candidate system, whether bought or built in-house, surfaces gaps that a summary marketing page won't.

### 7.2 When to buy an existing harness vs. build your own

Adopting an existing harness like Claude Code or Codex makes sense when your task fits the vendor's target use case (coding, for example), you don't need custom tool integrations beyond what the vendor supports, and you want the sandboxing, memory, and feedback logic already hardened by another team's production traffic. Building your own is usually worth the cost when your domain has unusual tools or data sources, your compliance or security requirements dictate a specific execution environment, or you need tight control over the planning and replanning logic for a workflow the existing harnesses don't model well. A useful rule of thumb: the more your task resembles general-purpose coding or research assistance, the more an off-the-shelf harness will get you most of the way there; the more it resembles a specialized internal workflow, the faster you'll hit the limits of what a generic harness assumes.

Cost is also part of the calculation, though it's better treated as a heuristic than a settled rule. Buying typically means paying for usage on someone else's infrastructure and accepting their defaults on sandboxing and permissioning, while building means absorbing the engineering time to implement and maintain the same components yourself: the control loop, the tool interface, memory management, and the sandbox, plus the ongoing cost of keeping all of it working as the underlying models change. As a rough guide, teams with a small number of well-defined agentic workflows tend to find the economics favor buying, since the fixed cost of building and maintaining a harness is spread across less usage. Teams running many varied agent tasks across a large engineering org more often find that the long-run cost of adapting a vendor-shaped harness to each new workflow exceeds the cost of building a purpose-fit one, particularly once custom tool integrations and compliance requirements start to accumulate.

### 7.3 Common pitfalls in harness design

The most common failures: error handling that's too thin and lets failures cascade, memory leakage where irrelevant context degrades planning over long sessions, and unbounded tool permissions that expose systems to unnecessary risk. Watch also for systems that skip validation, letting outputs that look right but aren't pass through unchecked. Fixing these early is much cheaper than retrofitting safety into a system that's already in production.


![Checklist infographic for evaluating an AI agent harness](https://harness-engineering.ai/wp-content/uploads/2026/03/agent-harness-complete-guide-infographic-scaled.webp)
*Source: Harness Engineering*


---

## FAQ

### What is the difference between an agent harness and an agent SDK?

An SDK gives you the raw APIs, and sometimes some session or memory helpers, but it doesn't own task state at runtime. A harness is the assembled system that runs on top of an SDK, coordinating memory, sandboxing, and feedback so an agent can complete multi-step tasks on its own.

### What components make up an AI agent harness?

A control loop, tool/function-calling interfaces, state and memory management, an isolated or permission-gated execution environment, and feedback loops for error handling and self-correction.

### How does an agent harness enable autonomous task execution?

It cycles through planning, action, and observation, uses tool calls to interact with external systems, checks outputs against defined criteria, and adjusts its plan based on results instead of stopping after one response.

### Is a code harness the same as an agent harness?

No. A code harness is narrower, focused on running and validating code for coding agents. An agent harness covers a wider range of tools, memory, and feedback across different task types.

### Does an agent SDK manage state on its own?

Partially. Some SDKs offer session or memory primitives that make it easier to persist conversation history or basic facts, but they generally don't provide a runtime that owns task state across a long-running, multi-step process. That ownership is what a full agent harness adds on top.

### Can I build a custom agent harness without an SDK?

Yes. Most teams still build on top of an existing SDK so they're not reimplementing low-level API integrations, leaving more time for the orchestration and feedback logic that actually defines the harness.

---

## Conclusion

An **agent harness** is the orchestration layer that turns a single-turn LLM into a system that can plan, act, observe, and adapt across multi-step tasks, through control loops, tool interfaces, memory management, sandboxed or permission-gated execution, and feedback loops. This is what lets systems like Claude Code and Codex work through complex tasks without constant human direction. It sits in a different layer than the agent SDK, which supplies the underlying primitives, and the narrower code harness, which focuses on execution and validation for coding tasks specifically.

When evaluating or designing your own agentic systems, check them against the same list: control loop, tool interface, memory, sandbox, feedback mechanism. Missing pieces show up quickly in production, usually as stalled tasks, runaway tool calls, or plans that never adapt when something goes wrong. If you want to go deeper, see our guides on [what is an AI agent](#), agent SDK comparisons, [sandboxing techniques](#), building feedback loops, [Codex vs. Claude Code](#), and how to [build a custom agent harness](#) from these same components.
---

## Sources

1. Medium — [https://miro.medium.com/1*JO74jIwiGmZNZzDS_8K6AA.png](https://miro.medium.com/1*JO74jIwiGmZNZzDS_8K6AA.png) (accessed September 03, 2026)
2. Medium — [https://miro.medium.com/v2/resize:fit:1400/0*AqZXSZB2OmCmYW_6.png](https://miro.medium.com/v2/resize:fit:1400/0*AqZXSZB2OmCmYW_6.png) (accessed September 03, 2026)
3. Penligent — [https://www.penligent.ai/hackinglabs/wp-content/uploads/2026/04/Sandboxes-for-Coding-Agents.png](https://www.penligent.ai/hackinglabs/wp-content/uploads/2026/04/Sandboxes-for-Coding-Agents.png) (accessed September 03, 2026)
4. Medium — [https://miro.medium.com/v2/resize:fit:2000/1*T_cbAuHmCbyIvgfK78eWKA.png](https://miro.medium.com/v2/resize:fit:2000/1*T_cbAuHmCbyIvgfK78eWKA.png) (accessed September 03, 2026)
5. Medium — [https://miro.medium.com/v2/resize:fit:2000/1*OWn0y2HG8LzQwkcdnaljcQ.png](https://miro.medium.com/v2/resize:fit:2000/1*OWn0y2HG8LzQwkcdnaljcQ.png) (accessed September 03, 2026)
6. Medium — [https://miro.medium.com/v2/resize:fit:1200/1*XIe0AzU8UNuX0Wqco7ouZg.png](https://miro.medium.com/v2/resize:fit:1200/1*XIe0AzU8UNuX0Wqco7ouZg.png) (accessed September 03, 2026)
7. Harness Engineering — [https://harness-engineering.ai/wp-content/uploads/2026/03/agent-harness-complete-guide-infographic-scaled.webp](https://harness-engineering.ai/wp-content/uploads/2026/03/agent-harness-complete-guide-infographic-scaled.webp) (accessed September 03, 2026)
8. Anthropic — [https://www.anthropic.com/engineering/building-effective-agents](https://www.anthropic.com/engineering/building-effective-agents) (accessed September 03, 2026)
9. OpenAI — [https://platform.openai.com/docs/guides/agents](https://platform.openai.com/docs/guides/agents) (accessed September 03, 2026)
10. Anthropic — [https://docs.anthropic.com/en/docs/claude-code/overview](https://docs.anthropic.com/en/docs/claude-code/overview) (accessed September 03, 2026)
11. OpenAI — [https://developers.openai.com/codex/](https://developers.openai.com/codex/) (accessed September 03, 2026)

---

That's a wrap for this edition. If it gave you something useful, the best next step is to try one idea for real this week.

**Let's connect.** 👉 I share what I'm building and learning with AI agents on [LinkedIn](https://www.linkedin.com/in/imrantauqir/) — come say hi, and see my work in my portfolio at [imrantauqir.com](https://imrantauqir.com/).

Until next time,
**Imran**
