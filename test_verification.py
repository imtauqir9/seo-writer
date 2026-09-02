"""Stage 6.5 harness: drives the verify -> rebut -> judge -> fix loop with stubbed
model calls, so the arguing logic can be checked without spending a pipeline's
worth of API calls.

Run:  python test_verification.py
"""

import json
import os
import sys

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

import seo_writer as sw

ARTICLE = "# Heading\n\nThe original sentence.\n"
OUTLINE = "## One\n## Two\n"
TAKEAWAYS = "- takeaway one\n"
RESEARCH = {
    "keywords": {"primary_keyword": "kw", "secondary_keywords": ["a", "b"]},
    "search_intent": "informational",
    "target_audience": "devs",
    "article_goal": "explain",
    "serp_context": "(snippets)",
}

ISSUE_A = {"id": "i1", "category": "factual", "severity": "high",
           "quote": "The original sentence.", "problem": "uncited", "fix": "cite it"}
ISSUE_B = {"id": "i2", "category": "ai_tell", "severity": "medium",
           "quote": "The original sentence.", "problem": "padding", "fix": "cut it"}


class Stub:
    """Replaces call_claude/call_openai. Routes by which system prompt is passed."""

    def __init__(self, reports, rebuttals=(), rulings=(), openai_raises=False):
        self.reports = list(reports)       # one per verify round
        self.rebuttals = list(rebuttals)
        self.rulings = list(rulings)
        self.openai_raises = openai_raises
        self.calls = []                    # (role, model)

    def _next(self, queue, role):
        if not queue:
            raise AssertionError("stub ran out of %s responses" % role)
        return json.dumps(queue.pop(0))

    def claude(self, prompt, system="", max_tokens=16000, model=None, effort=None, **kw):
        if system == sw.VERIFY_SYSTEM:
            self.calls.append(("verify", model))
            return self._next(self.reports, "verify")
        if system == sw.WRITER_SYSTEM:
            self.calls.append(("rebut", model))
            return self._next(self.rebuttals, "rebut")
        if system == sw.JUDGE_SYSTEM:
            self.calls.append(("judge-claude", model))
            return self._next(self.rulings, "judge")
        self.calls.append(("fix", model))
        return "# Heading\n\nThe revised sentence.\n"

    def openai(self, prompt, system="", max_tokens=4000):
        self.calls.append(("judge-openai", sw.OPENAI_JUDGE_MODEL))
        if self.openai_raises:
            raise sw.ClaudeError("503 the judge is down")
        return self._next(self.rulings, "judge")

    def roles(self):
        return [c[0] for c in self.calls]


def install(stub):
    sw.call_claude = stub.claude
    sw.call_openai = stub.openai
    return stub


def loop(stub, rounds=2):
    install(stub)
    return sw.verification_loop(ARTICLE, OUTLINE, TAKEAWAYS, RESEARCH, max_rounds=rounds)


def spy_on_apply_fixes(record):
    """Wrap apply_fixes so a case can assert which findings survived."""
    real = sw.apply_fixes

    def spy(article, upheld):
        record["ids"] = sorted(i.get("id") for i in upheld)
        return real(article, upheld)

    sw.apply_fixes = spy
    return real


PASS_REPORT = {"verdict": "pass", "scores": {"factual_support": 9}, "issues": []}


def revise(*issues):
    return {"verdict": "revise", "scores": {"factual_support": 5}, "issues": list(issues)}


# --- cases -----------------------------------------------------------------

def test_pass_verdict_leaves_article_alone():
    stub = Stub([PASS_REPORT])
    out = loop(stub)
    assert out == ARTICLE, "a passing audit must not touch the article"
    assert stub.roles() == ["verify"], stub.roles()


def test_revise_with_no_issues_is_treated_as_a_pass():
    stub = Stub([{"verdict": "revise", "issues": []}])
    out = loop(stub)
    assert out == ARTICLE
    assert stub.roles() == ["verify"]


def test_accepted_findings_are_applied_then_reaudited():
    stub = Stub(
        reports=[revise(ISSUE_A), PASS_REPORT],
        rebuttals=[{"responses": [{"id": "i1", "stance": "accept", "reason": "fair"}]}],
    )
    out = loop(stub)
    assert "revised" in out, "an accepted finding must reach apply_fixes"
    assert stub.roles() == ["verify", "rebut", "fix", "verify"], stub.roles()


def test_dispute_upheld_by_judge_is_applied():
    stub = Stub(
        reports=[revise(ISSUE_A), PASS_REPORT],
        rebuttals=[{"responses": [{"id": "i1", "stance": "dispute", "reason": "it is cited"}]}],
        rulings=[{"rulings": [{"id": "i1", "ruling": "uphold", "reasoning": "it is not"}]}],
    )
    out = loop(stub)
    assert "revised" in out
    assert "fix" in stub.roles(), stub.roles()


def test_dispute_overruled_by_judge_leaves_article_alone():
    stub = Stub(
        reports=[revise(ISSUE_A)],
        rebuttals=[{"responses": [{"id": "i1", "stance": "dispute", "reason": "taste"}]}],
        rulings=[{"rulings": [{"id": "i1", "ruling": "overrule", "reasoning": "agreed"}]}],
    )
    out = loop(stub)
    assert out == ARTICLE, "an overruled finding must not rewrite the article"
    assert "fix" not in stub.roles(), stub.roles()


def test_silence_from_the_writer_counts_as_acceptance():
    # Two findings raised, the writer answers only one.
    stub = Stub(
        reports=[revise(ISSUE_A, ISSUE_B), PASS_REPORT],
        rebuttals=[{"responses": [{"id": "i1", "stance": "accept", "reason": "ok"}]}],
    )
    applied = {}
    real = spy_on_apply_fixes(applied)
    try:
        loop(stub)
    finally:
        sw.apply_fixes = real
    assert applied.get("ids") == ["i1", "i2"], "unanswered i2 should be applied: %s" % applied


def test_unruled_dispute_leaves_the_text_standing():
    # The judge rules on i1 only; i2 is disputed and never ruled on.
    stub = Stub(
        reports=[revise(ISSUE_A, ISSUE_B), PASS_REPORT],
        rebuttals=[{"responses": [
            {"id": "i1", "stance": "dispute", "reason": "cited"},
            {"id": "i2", "stance": "dispute", "reason": "deliberate"},
        ]}],
        rulings=[{"rulings": [{"id": "i1", "ruling": "uphold", "reasoning": "no"}]}],
    )
    applied = {}
    real = spy_on_apply_fixes(applied)
    try:
        loop(stub)
    finally:
        sw.apply_fixes = real
    assert applied.get("ids") == ["i1"], "unruled i2 must not be applied: %s" % applied


def test_openai_judge_failure_falls_back_to_claude():
    prev_key = os.environ.get("OPENAI_API_KEY")
    prev_provider = sw.JUDGE_PROVIDER
    os.environ["OPENAI_API_KEY"] = "test-key"
    sw.JUDGE_PROVIDER = "auto"
    try:
        assert sw.judge_provider() == "openai"
        stub = Stub(
            reports=[revise(ISSUE_A), PASS_REPORT],
            rebuttals=[{"responses": [{"id": "i1", "stance": "dispute", "reason": "no"}]}],
            rulings=[{"rulings": [{"id": "i1", "ruling": "uphold", "reasoning": "yes"}]}],
            openai_raises=True,
        )
        loop(stub)
        roles = stub.roles()
        assert "judge-openai" in roles and "judge-claude" in roles, roles
    finally:
        sw.JUDGE_PROVIDER = prev_provider
        if prev_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = prev_key


def test_round_limit_is_honoured():
    # The audit never passes; the loop must still stop after max_rounds.
    stub = Stub(
        reports=[revise(ISSUE_A), revise(ISSUE_A)],
        rebuttals=[{"responses": [{"id": "i1", "stance": "accept", "reason": "ok"}]}] * 2,
    )
    out = loop(stub, rounds=2)
    assert stub.roles().count("verify") == 2, stub.roles()
    assert stub.roles().count("fix") == 2, stub.roles()
    assert "revised" in out


def test_verifier_and_judge_use_their_own_models():
    stub = Stub(
        reports=[revise(ISSUE_A), PASS_REPORT],
        rebuttals=[{"responses": [{"id": "i1", "stance": "dispute", "reason": "no"}]}],
        rulings=[{"rulings": [{"id": "i1", "ruling": "uphold", "reasoning": "yes"}]}],
    )
    prev_provider = sw.JUDGE_PROVIDER
    sw.JUDGE_PROVIDER = "anthropic"
    try:
        loop(stub)
    finally:
        sw.JUDGE_PROVIDER = prev_provider
    models = dict(stub.calls)
    assert models["verify"] == sw.VERIFIER_MODEL, models
    assert models["judge-claude"] == sw.JUDGE_MODEL, models


def test_unnumbered_and_duplicate_findings_survive():
    # An auditor that forgets an id, or repeats one, must not lose a finding.
    nameless = {k: v for k, v in ISSUE_B.items() if k != "id"}
    twin = dict(ISSUE_B, id="i1")
    stub = Stub(
        reports=[revise(ISSUE_A, nameless, twin), PASS_REPORT],
        rebuttals=[{"responses": [{"id": "i1", "stance": "accept", "reason": "ok"}]}],
    )
    applied = {}
    real = spy_on_apply_fixes(applied)
    try:
        loop(stub)
    finally:
        sw.apply_fixes = real
    assert len(applied.get("ids", [])) == 3, "all three findings should reach the fix: %s" % applied


CASES = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main():
    real_claude, real_openai = sw.call_claude, sw.call_openai
    failures = []
    for case in CASES:
        try:
            case()
        except Exception as e:
            failures.append((case.__name__, "%s: %s" % (type(e).__name__, e)))
        finally:
            sw.call_claude, sw.call_openai = real_claude, real_openai
    print("\n" + "=" * 60)
    for name, err in failures:
        print("FAIL %s\n     %s" % (name, err))
    print("%d/%d passed" % (len(CASES) - len(failures), len(CASES)))
    print("=" * 60)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
