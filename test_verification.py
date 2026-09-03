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

    def __init__(self, reports, rebuttals=(), rulings=(), openai_raises=False,
                 gemini_raises=False):
        self.reports = list(reports)       # one per verify round
        self.rebuttals = list(rebuttals)
        self.rulings = list(rulings)
        self.openai_raises = openai_raises
        self.gemini_raises = gemini_raises
        self.calls = []                    # (role-vendor, model)

    def _next(self, queue, role):
        if not queue:
            raise AssertionError("stub ran out of %s responses" % role)
        return json.dumps(queue.pop(0))

    def _route(self, system, vendor, model):
        """Every role is identified by its system prompt, whichever vendor runs it."""
        if system == sw.VERIFY_SYSTEM:
            self.calls.append(("verify-" + vendor, model))
            return self._next(self.reports, "verify")
        if system == sw.WRITER_SYSTEM:
            self.calls.append(("rebut-" + vendor, model))
            return self._next(self.rebuttals, "rebut")
        if system == sw.JUDGE_SYSTEM:
            self.calls.append(("judge-" + vendor, model))
            return self._next(self.rulings, "judge")
        self.calls.append(("fix-" + vendor, model))
        return "# Heading\n\nThe revised sentence.\n"

    def claude(self, prompt, system="", max_tokens=16000, model=None, effort=None, **kw):
        return self._route(system, "claude", model)

    def openai(self, prompt, system="", max_tokens=4000):
        if self.openai_raises:
            self.calls.append(("attempt-openai", sw.OPENAI_JUDGE_MODEL))
            raise sw.ClaudeError("503 the vendor is down")
        return self._route(system, "openai", sw.OPENAI_JUDGE_MODEL)

    def gemini(self, prompt, system="", max_tokens=4000):
        if self.gemini_raises:
            self.calls.append(("attempt-gemini", sw.GEMINI_MODEL))
            raise sw.ClaudeError("503 the vendor is down")
        return self._route(system, "gemini", sw.GEMINI_MODEL)

    def roles(self):
        """Call roles with the vendor suffix stripped, for order assertions."""
        return [c[0].rsplit("-", 1)[0] for c in self.calls]

    def tagged(self):
        return [c[0] for c in self.calls]


def install(stub):
    sw.call_claude = stub.claude
    sw.call_openai = stub.openai
    sw.call_gemini = stub.gemini
    return stub


def loop(stub, rounds=2, auditor="anthropic", judge="anthropic"):
    """Run the loop with the roster pinned, so a case tests logic and not .env."""
    install(stub)
    prev = (sw.AUDITOR_PROVIDER, sw.JUDGE_PROVIDER)
    sw.AUDITOR_PROVIDER, sw.JUDGE_PROVIDER = auditor, judge
    try:
        return sw.verification_loop(ARTICLE, OUTLINE, TAKEAWAYS, RESEARCH,
                                    max_rounds=rounds)
    finally:
        sw.AUDITOR_PROVIDER, sw.JUDGE_PROVIDER = prev


class keys:
    """Pin exactly which vendor keys exist for the duration of a block."""

    ALL = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")

    def __init__(self, *present):
        self.present = present

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in self.ALL}
        for k in self.ALL:
            if k in self.present:
                os.environ[k] = "test-key"
            else:
                os.environ.pop(k, None)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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
    with keys("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        stub = Stub(
            reports=[revise(ISSUE_A), PASS_REPORT],
            rebuttals=[{"responses": [{"id": "i1", "stance": "dispute", "reason": "no"}]}],
            rulings=[{"rulings": [{"id": "i1", "ruling": "uphold", "reasoning": "yes"}]}],
            openai_raises=True,
        )
        loop(stub, judge="openai")
        tagged = stub.tagged()
        assert "attempt-openai" in tagged, tagged
        assert "judge-claude" in tagged, tagged


def test_gemini_judge_failure_falls_back_to_claude():
    with keys("ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        stub = Stub(
            reports=[revise(ISSUE_A), PASS_REPORT],
            rebuttals=[{"responses": [{"id": "i1", "stance": "dispute", "reason": "no"}]}],
            rulings=[{"rulings": [{"id": "i1", "ruling": "uphold", "reasoning": "yes"}]}],
            gemini_raises=True,
        )
        loop(stub, judge="gemini")
        tagged = stub.tagged()
        assert "attempt-gemini" in tagged, tagged
        assert "judge-claude" in tagged, tagged


def test_a_working_second_vendor_actually_runs_the_audit():
    with keys("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        stub = Stub(
            reports=[revise(ISSUE_A), PASS_REPORT],
            rebuttals=[{"responses": [{"id": "i1", "stance": "accept", "reason": "ok"}]}],
        )
        loop(stub, auditor="openai")
        tagged = stub.tagged()
        # The audit runs on OpenAI; the writer and the fix stay on Claude.
        assert "verify-openai" in tagged, tagged
        assert "rebut-claude" in tagged and "fix-claude" in tagged, tagged


# --- roster resolution -----------------------------------------------------

def resolved(auditor="auto", judge="auto"):
    prev = (sw.AUDITOR_PROVIDER, sw.JUDGE_PROVIDER)
    sw.AUDITOR_PROVIDER, sw.JUDGE_PROVIDER = auditor, judge
    try:
        return sw.resolve_agents()
    finally:
        sw.AUDITOR_PROVIDER, sw.JUDGE_PROVIDER = prev


def test_writer_always_stays_on_anthropic():
    for combo in (("ANTHROPIC_API_KEY",),
                  ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"),
                  ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")):
        with keys(*combo):
            assert resolved()["writer"]["provider"] == "anthropic", combo


def test_auditor_leaves_the_writers_vendor_first():
    # One extra key exists; it must be spent on the auditor, not the judge,
    # because a blind spot there means the finding is never raised at all.
    with keys("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        agents = resolved()
        assert agents["auditor"]["provider"] == "openai", agents
        assert agents["judge"]["provider"] == "anthropic", agents


def test_three_keys_give_three_vendors():
    with keys("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        agents = resolved()
        vendors = {a["provider"] for a in agents.values()}
        assert vendors == {"anthropic", "openai", "gemini"}, agents


def test_anthropic_only_still_uses_three_distinct_models():
    with keys("ANTHROPIC_API_KEY"):
        agents = resolved()
        assert {a["provider"] for a in agents.values()} == {"anthropic"}, agents
        # The writer must at least not audit itself.
        assert agents["auditor"]["model"] != agents["writer"]["model"], agents


def test_explicit_provider_overrides_the_auto_choice():
    with keys("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        agents = resolved(auditor="anthropic", judge="openai")
        assert agents["auditor"]["provider"] == "anthropic", agents
        assert agents["judge"]["provider"] == "openai", agents


def test_judge_provider_helper_matches_the_roster():
    # Both read the module's own settings, so they must never disagree - whatever
    # JUDGE_PROVIDER happens to be set to in the local .env.
    with keys("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        assert sw.judge_provider() == sw.resolve_agents()["judge"]["provider"]


def test_description_counts_models_and_vendors():
    with keys("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        text = sw.describe_agents(resolved())
        assert "3 distinct models across 3 vendor(s)" in text, text


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
    with keys("ANTHROPIC_API_KEY"):
        loop(stub)
    models = dict(stub.calls)
    assert models["verify-claude"] == sw.VERIFIER_MODEL, models
    assert models["judge-claude"] == sw.JUDGE_MODEL, models
    # The rebuttal is the writer speaking, so it runs on the writer's model.
    assert models["rebut-claude"] == sw.MODEL or models["rebut-claude"] is None, models


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


# --- the argument record ---------------------------------------------------

def recorded(stub, rounds=2, auditor="anthropic", judge="anthropic"):
    install(stub)
    record = {}
    prev = (sw.AUDITOR_PROVIDER, sw.JUDGE_PROVIDER)
    sw.AUDITOR_PROVIDER, sw.JUDGE_PROVIDER = auditor, judge
    try:
        sw.verification_loop(ARTICLE, OUTLINE, TAKEAWAYS, RESEARCH,
                             max_rounds=rounds, record=record)
    finally:
        sw.AUDITOR_PROVIDER, sw.JUDGE_PROVIDER = prev
    return record


def test_record_is_optional():
    # The loop must still work for callers that do not want the transcript.
    stub = Stub([PASS_REPORT])
    out = loop(stub)
    assert out == ARTICLE


def test_record_captures_the_full_argument():
    with keys("ANTHROPIC_API_KEY"):
        record = recorded(Stub(
            reports=[revise(ISSUE_A, ISSUE_B), PASS_REPORT],
            rebuttals=[{"responses": [
                {"id": "i1", "stance": "accept", "reason": "fair"},
                {"id": "i2", "stance": "dispute", "reason": "deliberate"},
            ]}],
            rulings=[{"rulings": [{"id": "i2", "ruling": "overrule", "reasoning": "taste"}]}],
        ))
    assert len(record["rounds"]) == 2, record["rounds"]
    first = record["rounds"][0]
    assert len(first["issues"]) == 2
    assert len(first["responses"]) == 2
    assert first["rulings"][0]["ruling"] == "overrule"
    assert first["applied"] == ["i1"], first["applied"]
    assert record["agents"]["writer"]["provider"] == "anthropic"
    assert "passed on round 2" in record["outcome"], record["outcome"]


def test_record_marks_unanswered_and_unruled_findings():
    with keys("ANTHROPIC_API_KEY"):
        record = recorded(Stub(
            reports=[revise(ISSUE_A, ISSUE_B), PASS_REPORT],
            rebuttals=[{"responses": [{"id": "i1", "stance": "dispute", "reason": "no"}]}],
            rulings=[{"rulings": []}],
        ))
    first = record["rounds"][0]
    assert first["unanswered"] == ["i2"], first["unanswered"]
    assert first.get("unruled") == ["i1"], first.get("unruled")
    # i2 was never answered so it is applied; i1 was disputed and never ruled on.
    assert first["applied"] == ["i2"], first["applied"]


def test_stats_add_up():
    with keys("ANTHROPIC_API_KEY"):
        record = recorded(Stub(
            reports=[revise(ISSUE_A, ISSUE_B), PASS_REPORT],
            rebuttals=[{"responses": [
                {"id": "i1", "stance": "accept", "reason": "ok"},
                {"id": "i2", "stance": "dispute", "reason": "no"},
            ]}],
            rulings=[{"rulings": [{"id": "i2", "ruling": "uphold", "reasoning": "yes"}]}],
        ))
    s = sw.review_stats(record)
    assert s == {"rounds": 2, "raised": 2, "accepted": 1, "disputed": 1,
                 "upheld": 1, "overruled": 0, "applied": 2}, s


def test_review_report_shows_every_side():
    with keys("ANTHROPIC_API_KEY"):
        record = recorded(Stub(
            reports=[revise(ISSUE_A, ISSUE_B), PASS_REPORT],
            rebuttals=[{"responses": [
                {"id": "i1", "stance": "accept", "reason": "the auditor is right"},
                {"id": "i2", "stance": "dispute", "reason": "a deliberate choice"},
            ]}],
            rulings=[{"rulings": [{"id": "i2", "ruling": "overrule",
                                   "reasoning": "taste, not accuracy"}]}],
        ))
    md = sw.format_review(record, "A Title")
    assert "# Verification review: A Title" in md, md[:200]
    assert "the auditor is right" in md
    assert "a deliberate choice" in md
    assert "taste, not accuracy" in md
    assert "**Result:** applied" in md
    assert "**Result:** not applied" in md
    assert "| writer | `claude-sonnet-5` | anthropic |" in md


def test_review_report_names_silence_explicitly():
    with keys("ANTHROPIC_API_KEY"):
        record = recorded(Stub(
            reports=[revise(ISSUE_A), PASS_REPORT],
            rebuttals=[{"responses": []}],
        ))
    md = sw.format_review(record)
    assert "did not respond - counted as accepted" in md, md


def test_review_report_survives_an_audit_that_never_ran():
    assert "did not run" in sw.format_review({})


def test_review_files_are_written():
    import tempfile
    with keys("ANTHROPIC_API_KEY"):
        record = recorded(Stub(
            reports=[revise(ISSUE_A), PASS_REPORT],
            rebuttals=[{"responses": [{"id": "i1", "stance": "accept", "reason": "ok"}]}],
        ))
    with tempfile.TemporaryDirectory() as d:
        md_path, json_path = sw.write_review("a-slug", record, sw.Path(d), "Title")
        assert md_path.exists() and json_path.exists()
        assert "Verification review" in md_path.read_text(encoding="utf-8")
        reloaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert reloaded["rounds"][0]["applied"] == ["i1"], reloaded["rounds"][0]


# --- parsing what three different vendors call JSON -------------------------
#
# Eight call sites depend on extract_json, and a ValueError escaping it crashed
# the pipeline with a bare traceback - which the web UI could only report as
# "exited with code 1".

FENCE = "```"


def test_plain_json_object():
    assert sw.extract_json('{"verdict": "pass"}') == {"verdict": "pass"}


def test_fenced_json_block():
    text = FENCE + 'json\n{"verdict": "pass", "issues": []}\n' + FENCE
    assert sw.extract_json(text) == {"verdict": "pass", "issues": []}


def test_unlabelled_fence():
    text = FENCE + '\n{"a": 1}\n' + FENCE
    assert sw.extract_json(text) == {"a": 1}


def test_preamble_before_the_object():
    text = 'Here is my audit:\n\n{"verdict": "revise", "issues": [{"id": "i1"}]}'
    assert sw.extract_json(text)["verdict"] == "revise"


def test_trailing_chatter_containing_braces():
    # The greedy pattern would swallow the closing brace of the sign-off.
    text = '{"verdict": "pass"}\n\nLet me know if you need anything {else}!'
    assert sw.extract_json(text) == {"verdict": "pass"}


def test_nested_objects_survive():
    text = '{"scores": {"factual_support": 8}, "issues": []}'
    assert sw.extract_json(text)["scores"]["factual_support"] == 8


def test_prose_raises_a_readable_error_not_a_traceback():
    try:
        sw.extract_json("I am not able to audit this article.")
    except sw.ClaudeError as e:
        assert "not JSON" in str(e)
        assert "I am not able" in str(e), str(e)
    else:
        raise AssertionError("prose should not parse")


def test_empty_response_raises_claude_error():
    try:
        sw.extract_json("")
    except sw.ClaudeError as e:
        assert "empty" in str(e)
    else:
        raise AssertionError("an empty response should not parse")


def test_a_json_list_is_not_accepted_as_an_object():
    # Callers all do .get() on the result; a list would fail far from here.
    try:
        sw.extract_json("[1, 2, 3]")
    except sw.ClaudeError:
        pass
    else:
        raise AssertionError("a list should not satisfy extract_json")


# --- a failed stage must not destroy the article ----------------------------
#
# The article reaching stage 6.5 has already cost a dozen calls. A stage whose
# job is to check it must never be the thing that throws it away.

class Boom(Stub):
    """A stub where one named role raises instead of answering."""

    def __init__(self, fails, **kw):
        super().__init__(**kw)
        self.fails = fails

    def _route(self, system, vendor, model):
        role = {sw.VERIFY_SYSTEM: "verify", sw.WRITER_SYSTEM: "rebut",
                sw.JUDGE_SYSTEM: "judge"}.get(system, "fix")
        if role in self.fails:
            self.calls.append(("attempt-" + role, model))
            raise sw.ClaudeError("the model hit its output cap")
        return super()._route(system, vendor, model)


def test_a_failed_audit_returns_the_article_untouched():
    with keys("ANTHROPIC_API_KEY"):
        stub = Boom({"verify"}, reports=[PASS_REPORT])
        install(stub)
        record = {}
        out = sw.verification_loop(ARTICLE, OUTLINE, TAKEAWAYS, RESEARCH,
                                   max_rounds=2, record=record)
    assert out == ARTICLE, "a broken audit must not change the article"
    assert "verification failed" in record["outcome"], record["outcome"]
    assert "output cap" in record["error"], record["error"]


def test_a_failed_rebuttal_applies_every_finding():
    # Silence already counts as acceptance; an unreachable writer is silence.
    with keys("ANTHROPIC_API_KEY"):
        stub = Boom({"rebut"}, reports=[revise(ISSUE_A, ISSUE_B), PASS_REPORT])
        install(stub)
        applied = {}
        real = sw.apply_fixes

        def spy(article, upheld):
            applied["ids"] = sorted(i.get("id") for i in upheld)
            return real(article, upheld)

        sw.apply_fixes = spy
        try:
            out = sw.verification_loop(ARTICLE, OUTLINE, TAKEAWAYS, RESEARCH,
                                       max_rounds=2)
        finally:
            sw.apply_fixes = real
    assert applied["ids"] == ["i1", "i2"], applied
    assert "revised" in out


def test_a_failed_judge_leaves_disputed_text_standing():
    with keys("ANTHROPIC_API_KEY"):
        stub = Boom(
            {"judge"},
            reports=[revise(ISSUE_A), PASS_REPORT],
            rebuttals=[{"responses": [{"id": "i1", "stance": "dispute",
                                       "reason": "deliberate"}]}],
        )
        install(stub)
        out = sw.verification_loop(ARTICLE, OUTLINE, TAKEAWAYS, RESEARCH,
                                   max_rounds=2)
    assert out == ARTICLE, "an unruled dispute leaves the text as written"


def test_a_failed_fix_pass_keeps_the_last_good_version():
    with keys("ANTHROPIC_API_KEY"):
        stub = Boom(
            {"fix"},
            reports=[revise(ISSUE_A), PASS_REPORT],
            rebuttals=[{"responses": [{"id": "i1", "stance": "accept",
                                       "reason": "ok"}]}],
        )
        install(stub)
        record = {}
        out = sw.verification_loop(ARTICLE, OUTLINE, TAKEAWAYS, RESEARCH,
                                   max_rounds=2, record=record)
    assert out == ARTICLE, "a failed rewrite must not lose the article"
    assert "fix pass failed" in record["outcome"], record["outcome"]


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
