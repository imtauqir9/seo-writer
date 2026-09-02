"""Tests for the deterministic AI-tell scanner behind step 6's third pass.

Nothing here calls a model - the scanner is pure text analysis, which is the
whole point of it.

Run:  python test_humanizer.py
"""

import os
import sys

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

import seo_writer as sw


def scan(text):
    return sw.scan_ai_tells(text)


def phrases(result):
    return {h["phrase"]: h["count"] for h in result["hits"]}


def kinds(result):
    return {s["kind"]: s["count"] for s in result["structural"]}


# --- vocabulary and phrases ------------------------------------------------

def test_counts_vocabulary_tells():
    r = scan("This is a pivotal shift. We delve into the intricate tapestry of it.")
    p = phrases(r)
    assert p.get("pivotal") == 1, p
    assert p.get("delve") == 1, p
    assert p.get("intricate") == 1, p
    assert p.get("tapestry") == 1, p


def test_vocabulary_matches_whole_words_only():
    # "keynote" must not register as "key", "underscores" is its own entry.
    r = scan("The keynote covered delegation and understanding.")
    p = phrases(r)
    assert "delve" not in p, p
    assert r["total"] == 0, p


def test_counts_multiword_phrases():
    r = scan("It stands as proof. Experts say it is important to note this.")
    p = phrases(r)
    assert p.get("stands as") == 1, p
    assert p.get("experts say") == 1, p
    assert p.get("it is important to note") == 1, p


def test_repeated_phrase_counts_each_time():
    r = scan("In order to win, in order to grow, in order to last.")
    assert phrases(r).get("in order to") == 3, phrases(r)


def test_clean_prose_scores_zero():
    text = ("The API returned a 429. We retried twice and it worked.\n\n"
            "Cost went from $40 a month to $6. That was the whole change.")
    r = scan(text)
    assert r["total"] == 0, r["hits"] + r["structural"]


# --- structural tells ------------------------------------------------------

def test_flags_title_case_headings():
    r = scan("## How To Build A Retrieval Pipeline\n\nBody text here.\n")
    assert kinds(r).get("title_case_heading") == 1, r["structural"]


def test_sentence_case_heading_is_clean():
    r = scan("## How to build a retrieval pipeline\n\nBody text here.\n")
    assert "title_case_heading" not in kinds(r), r["structural"]


def test_short_heading_is_not_judged_title_case():
    # Two words proves nothing either way; do not cry wolf.
    r = scan("## Getting Started\n\nBody.\n")
    assert "title_case_heading" not in kinds(r), r["structural"]


def test_flags_inline_header_bullets():
    r = scan("- **Speed:** it is faster\n- **Cost:** it is cheaper\n")
    assert kinds(r).get("inline_header_bullet") == 2, r["structural"]


def test_flags_em_dashes_and_emoji_headings():
    r = scan("## Results \U0001F680\n\nIt worked — mostly.\n")
    k = kinds(r)
    assert k.get("em_dash") == 1, k
    assert k.get("emoji_heading") == 1, k


# --- scope of the scan -----------------------------------------------------

def test_citations_and_image_markers_are_not_scanned():
    # A source URL containing a tell word is not the writer's voice.
    text = ("Real sentence here.\n"
            "Source: https://example.com/the-evolving-landscape-of-delve\n"
            "[IMAGE: a pivotal chart | Query: pivotal tapestry]\n")
    r = scan(text)
    assert r["total"] == 0, r["hits"]


def test_code_blocks_are_not_scanned():
    text = "Plain line.\n\n```python\nkey = enhance(delve)\n```\n\nAnother line.\n"
    r = scan(text)
    assert r["total"] == 0, r["hits"]


# --- sentence uniformity ---------------------------------------------------

def test_uniform_sentence_length_scores_low_variation():
    uniform = " ".join(["The system reads the file and writes the result again."] * 8)
    r = scan(uniform)
    assert r["sentence_count"] >= 5, r
    assert r["sentence_cv"] < 0.2, r["sentence_cv"]


def test_varied_sentence_length_scores_higher_variation():
    varied = ("It broke. " * 1 +
              "The retry logic waited four seconds, then eight, then gave up entirely "
              "because the upstream service had stopped answering at all. "
              "We shipped it. "
              "Nobody noticed for a week, which tells you something about how much "
              "anyone was actually reading the output in the first place. "
              "Then it broke again. ")
    r = scan(varied)
    assert r["sentence_cv"] > 0.45, r["sentence_cv"]


def test_too_few_sentences_reports_no_variation_signal():
    r = scan("One short line.")
    assert r["sentence_cv"] == 0.0, r


# --- reporting -------------------------------------------------------------

def test_report_lists_phrases_with_counts():
    r = scan("It stands as proof. It stands as fact. We delve deeper.")
    report = sw.format_tell_report(r)
    assert '"stands as" x2' in report, report
    assert '"delve" x1' in report, report


def test_report_calls_out_uniform_sentences():
    uniform = " ".join(["The system reads the file and writes the result again."] * 8)
    report = sw.format_tell_report(scan(uniform))
    assert "too uniform" in report, report


def test_report_includes_structural_examples():
    r = scan("## How To Build A Retrieval Pipeline\n\nBody.\n")
    report = sw.format_tell_report(r)
    assert "title case heading" in report, report
    assert "How To Build" in report, report


def test_per_1k_is_normalised_by_length():
    dense = "We delve into it."
    padded = dense + " " + " ".join(["a clean filler sentence here."] * 60)
    assert scan(dense)["per_1k"] > scan(padded)["per_1k"]


# --- em dash stripping -----------------------------------------------------

def test_body_em_dashes_become_commas_and_hyphens():
    out = sw._strip_em_dashes("It worked — mostly. A well—known case.\n")
    assert "—" not in out, out
    assert "It worked, mostly." in out, out
    assert "well-known" in out, out


def test_heading_em_dash_becomes_a_colon():
    out = sw._strip_em_dashes("## NVIDIA AI certification costs — full breakdown\n")
    assert out.strip() == "## NVIDIA AI certification costs: full breakdown", out


def test_heading_that_already_has_a_colon_gets_a_hyphen():
    out = sw._strip_em_dashes("### DLI: course catalog — what is available?\n")
    assert out.strip() == "### DLI: course catalog - what is available?", out


def test_heading_comma_rule_is_not_used():
    # A comma reads wrong in a heading; that was the point of the separate rule.
    out = sw._strip_em_dashes("## Two frameworks — DLI vs NCP\n")
    assert "frameworks," not in out, out


def test_image_markers_and_sources_keep_their_em_dashes():
    text = ("[IMAGE: two engineers — one presenting | Query: engineer — meeting]\n"
            "Source: https://example.com/a—b\n")
    assert sw._strip_em_dashes(text) == text


def test_stripping_headings_clears_the_scanner_finding():
    article = ("## Costs — a full breakdown\n\nPlain body sentence here.\n"
               "### Difficulty — beginner to advanced\n\nMore body text.\n")
    before = scan(article)
    after = scan(sw._strip_em_dashes(article))
    assert kinds(before).get("em_dash") == 2, before["structural"]
    assert "em_dash" not in kinds(after), after["structural"]


CASES = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main():
    failures = []
    for case in CASES:
        try:
            case()
        except Exception as e:
            failures.append((case.__name__, "%s: %s" % (type(e).__name__, e)))
    print("\n" + "=" * 60)
    for name, err in failures:
        print("FAIL %s\n     %s" % (name, err))
    print("%d/%d passed" % (len(CASES) - len(failures), len(CASES)))
    print("=" * 60)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
