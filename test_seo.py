"""Tests for the SEO/GEO output layer: FAQ extraction, JSON-LD, named sources,
the answer block, and the HTML head.

No model calls - all of this is deterministic text and markup work.

Run:  python test_seo.py
"""

import json
import os
import re
import sys
import tempfile

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

import seo_writer as sw

ARTICLE = """# What is semantic caching?

Semantic caching stores answers by meaning rather than by exact string, so a
reworded question still hits the cache.

## How it works

An embedding of the query is compared against stored queries.
Source: https://docs.example.com/semantic-cache

## Costs

| Tier | Price |
|---|---|
| Free | $0 |

Pricing starts at $40 a month. Source: https://arxiv.org/abs/1234.5678

## FAQ

### Does semantic caching work with streaming?

Yes. The cache is checked before the stream opens, so a hit skips the model
call entirely.

### How much does it cut latency?

On a p99 basis, teams report 30 to 50 percent, though the number depends
heavily on hit rate.

## Conclusion

Try it on one endpoint first.
"""

META = {"title": "What is semantic caching? A practical guide",
        "description": "Semantic caching stores answers by meaning, not exact match.",
        "slug": "what-is-semantic-caching"}

IMAGES = {"[IMAGE: a]": {"alt": "a", "url": "https://img.example.com/1.png",
                         "source": "example"}}


# --- publisher names -------------------------------------------------------

def test_known_publishers_get_their_real_name():
    assert sw.publisher_name("https://arxiv.org/abs/1234") == "arXiv"
    assert sw.publisher_name("https://www.nytimes.com/x") == "The New York Times"
    assert sw.publisher_name("https://developer.nvidia.com/dli") == "NVIDIA Developer"


def test_unknown_publisher_falls_back_to_the_domain_label():
    assert sw.publisher_name("https://blog.acme-corp.com/post") == "Acme Corp"


def test_three_letter_domains_are_treated_as_acronyms():
    # Real output cited idc.com, which title-cased to the meaningless "Idc".
    assert sw.publisher_name("https://www.idc.com") == "IDC"
    assert sw.publisher_name("https://www.ibm.com/x") == "IBM"


# --- sources ---------------------------------------------------------------

def test_sources_are_named_and_dated():
    section = sw.build_sources_section(ARTICLE, accessed="March 03, 2026")
    assert "arXiv" in section, section
    assert "(accessed March 03, 2026)" in section, section
    # The URL is still a working link, not just prose.
    assert "](https://arxiv.org/abs/1234.5678)" in section, section


def test_no_sources_means_no_section():
    assert sw.build_sources_section("# Title\n\nNo links here.\n") == ""


# --- FAQ extraction --------------------------------------------------------

def test_faq_questions_and_answers_are_extracted():
    faqs = sw.parse_faq(ARTICLE)
    assert len(faqs) == 2, faqs
    assert faqs[0]["question"] == "Does semantic caching work with streaming?"
    assert "skips the model call" in faqs[0]["answer"]


def test_faq_parsing_stops_at_the_next_h2():
    # "Try it on one endpoint first." belongs to Conclusion, not to the last FAQ.
    faqs = sw.parse_faq(ARTICLE)
    assert all("one endpoint" not in f["answer"] for f in faqs), faqs


def test_headings_that_are_not_questions_are_skipped():
    doc = "## FAQ\n\n### Background\n\nSome prose.\n\n### Is this real?\n\nYes.\n"
    faqs = sw.parse_faq(doc)
    assert [f["question"] for f in faqs] == ["Is this real?"], faqs


def test_q_prefixes_are_stripped_from_question_names():
    # Real output writes "### Q: Do NVIDIA certifications expire?" - the "Q:" is
    # presentation, and belongs nowhere near a FAQPage question name.
    doc = ("## FAQ\n\n### Q: Do certifications expire?\n\nDLI certificates do not.\n\n"
           "### Question - How long is prep?\n\nAbout six weeks.\n")
    assert [f["question"] for f in sw.parse_faq(doc)] == [
        "Do certifications expire?", "How long is prep?"]


def test_article_without_an_faq_yields_nothing():
    assert sw.parse_faq("# Title\n\n## Body\n\nText.\n") == []


def test_bold_questions_are_also_picked_up():
    doc = "## FAQ\n\n**Is this supported?**\n\nYes, since v2.\n"
    faqs = sw.parse_faq(doc)
    assert faqs == [{"question": "Is this supported?", "answer": "Yes, since v2."}], faqs


# --- JSON-LD ---------------------------------------------------------------

def jsonld(article=ARTICLE, meta=META, images=IMAGES, when="2026-03-03T00:00:00Z"):
    return json.loads(sw.build_jsonld("a-slug", article, meta, images, when)
                      .replace("<\\/", "</"))


def test_jsonld_is_valid_json_with_both_nodes():
    graph = jsonld()["@graph"]
    types = [n["@type"] for n in graph]
    assert types == ["Article", "FAQPage"], types


def test_article_node_carries_the_metadata():
    node = jsonld()["@graph"][0]
    assert node["description"] == META["description"]
    assert node["datePublished"] == "2026-03-03T00:00:00Z"
    assert node["author"]["name"] == sw.AUTHOR_NAME
    assert node["wordCount"] > 50
    assert node["image"] == ["https://img.example.com/1.png"]


def test_headline_is_capped_at_110_characters():
    long_meta = dict(META, title="x" * 300)
    assert len(jsonld(meta=long_meta)["@graph"][0]["headline"]) == 110


def test_citations_are_named():
    names = {c["name"] for c in jsonld()["@graph"][0]["citation"]}
    assert "arXiv" in names, names


def test_faqpage_mirrors_the_parsed_questions():
    faq_node = jsonld()["@graph"][1]
    questions = [q["name"] for q in faq_node["mainEntity"]]
    assert questions == [f["question"] for f in sw.parse_faq(ARTICLE)], questions
    assert all(q["acceptedAnswer"]["text"] for q in faq_node["mainEntity"])


def test_no_faq_means_no_faqpage_node():
    graph = jsonld(article="# T\n\n## Body\n\nText.\n")["@graph"]
    assert [n["@type"] for n in graph] == ["Article"], graph


def test_canonical_appears_only_when_site_url_is_set():
    prev = sw.SITE_URL
    try:
        sw.SITE_URL = ""
        assert "url" not in jsonld()["@graph"][0]
        sw.SITE_URL = "https://example.com"
        node = jsonld()["@graph"][0]
        assert node["url"] == "https://example.com/a-slug"
        assert node["mainEntityOfPage"]["@id"] == "https://example.com/a-slug"
    finally:
        sw.SITE_URL = prev


def test_script_tag_cannot_break_out_of_the_json_block():
    doc = ARTICLE + '\n\n### Is </script> handled?\n\nYes it is escaped.\n'
    raw = sw.build_jsonld("a-slug", doc, META, IMAGES, "2026-03-03T00:00:00Z")
    assert "</script>" not in raw, raw[:400]


# --- answer block ----------------------------------------------------------

def test_answer_block_lands_directly_under_the_h1():
    out = sw.insert_answer_block(ARTICLE, "Semantic caching is X.")
    lines = [l for l in out.split("\n")]
    assert lines[0].startswith("# ")
    assert lines[2] == "Semantic caching is X.", lines[:4]


def test_empty_answer_leaves_the_article_untouched():
    assert sw.insert_answer_block(ARTICLE, "") == ARTICLE


def test_answer_block_still_works_without_an_h1():
    out = sw.insert_answer_block("No heading here.\n", "The answer.")
    assert out.startswith("The answer.")


# --- HTML head -------------------------------------------------------------

def html_for(article=ARTICLE, meta=META, images=IMAGES):
    with tempfile.TemporaryDirectory() as d:
        path = sw._write_html("a-slug", article, sw.Path(d), meta=meta, images=images,
                              generated_at="2026-03-03T00:00:00Z")
        if path is None:
            return None
        return path.read_text(encoding="utf-8")


def test_head_carries_title_description_and_author():
    html = html_for()
    if html is None:
        return                      # markdown package absent; nothing to assert
    assert f"<title>{META['title']}</title>" in html, html[:600]
    assert f'<meta name="description" content="{META["description"]}">' in html
    assert f'<meta name="author" content="{sw.AUTHOR_NAME}">' in html
    assert '<html lang="en">' in html
    assert 'name="viewport"' in html


def test_head_carries_open_graph_and_twitter_cards():
    html = html_for()
    if html is None:
        return
    for tag in ('property="og:type" content="article"', 'property="og:title"',
                'property="og:image"', 'name="twitter:card"', 'name="twitter:image"'):
        assert tag in html, tag


def test_head_embeds_the_structured_data():
    html = html_for()
    if html is None:
        return
    assert '<script type="application/ld+json">' in html
    block = re.search(r'<script type="application/ld\+json">\n(.*?)\n</script>',
                      html, re.DOTALL)
    assert block, html[:800]
    parsed = json.loads(block.group(1).replace("<\\/", "</"))
    assert parsed["@context"] == "https://schema.org"
    assert [n["@type"] for n in parsed["@graph"]] == ["Article", "FAQPage"]


def test_quotes_in_metadata_cannot_break_an_attribute():
    meta = dict(META, description='He said "yes" & left')
    html = html_for(meta=meta)
    if html is None:
        return
    assert 'content="He said &quot;yes&quot; &amp; left"' in html, html[:900]


def test_canonical_is_omitted_when_site_url_is_unset():
    prev = sw.SITE_URL
    try:
        sw.SITE_URL = ""
        html = html_for()
        if html is None:
            return
        assert "rel=\"canonical\"" not in html
        sw.SITE_URL = "https://example.com"
        html = html_for()
        assert '<link rel="canonical" href="https://example.com/a-slug">' in html
    finally:
        sw.SITE_URL = prev


# --- article length ---------------------------------------------------------

def test_the_three_lengths_exist_and_differ():
    names = set(sw.LENGTH_PROFILES)
    assert names == {"default", "1000", "2000"}, names
    words = {n: sw.LENGTH_PROFILES[n]["words"] for n in names}
    assert len(set(words.values())) == 3, words


def test_unknown_or_missing_length_falls_back_to_default():
    for value in (None, "", "600", "enormous", "DEFAULT "):
        assert sw.length_profile(value)["words"] == \
               sw.LENGTH_PROFILES["default"]["words"], value


def test_a_shorter_article_asks_for_less_of_everything():
    # A short article must be short, not a full-length outline crammed down.
    short = sw.length_profile("1000")
    mid = sw.length_profile("2000")
    full = sw.length_profile("default")

    def low(profile, key):
        return int(profile[key].split("-")[0].replace(",", ""))

    for key in ("sections", "images", "faq", "intro", "conclusion", "words"):
        assert low(short, key) <= low(mid, key) <= low(full, key), key


def test_length_reaches_the_outline_and_the_writer():
    seen = {}

    def fake_call(prompt, system="", max_tokens=16000, model=None, effort=None, **kw):
        seen.setdefault("prompts", []).append(prompt)
        return "## An outline"

    real = sw.call_claude
    sw.call_claude = fake_call
    try:
        profile = sw.length_profile("1000")
        sw.generate_outline("T", "kw", {"keywords": {}}, "takeaways", profile=profile)
        sw.write_content("T", "kw", "outline", {"keywords": {}}, "takeaways",
                         profile=profile)
    finally:
        sw.call_claude = real

    outline_prompt, writer_prompt = seen["prompts"]
    assert "900-1,100" in outline_prompt, "the outline must know the target"
    assert "3-4 H2 main sections" in outline_prompt, outline_prompt[:400]
    assert "900-1,100 words total" in writer_prompt, "the writer must know the target"


def test_the_writer_defaults_when_given_no_profile():
    seen = {}

    def fake_call(prompt, system="", max_tokens=16000, model=None, effort=None, **kw):
        seen["prompt"] = prompt
        return "text"

    real = sw.call_claude
    sw.call_claude = fake_call
    try:
        sw.write_content("T", "kw", "outline", {"keywords": {}}, "takeaways")
    finally:
        sw.call_claude = real
    assert "2,500-3,500" in seen["prompt"]


# --- the LinkedIn post ------------------------------------------------------

def test_linkedin_post_is_built_from_the_article_not_the_topic():
    seen = {}

    def fake_call(prompt, system="", max_tokens=16000, model=None, effort=None, **kw):
        seen["prompt"] = prompt
        return "A post about semantic caching.\n\nRead the full piece."

    real = sw.call_claude
    sw.call_claude = fake_call
    try:
        post = sw.generate_linkedin_post("Semantic caching", ARTICLE,
                                         {"keywords": {"primary_keyword": "semantic caching"}})
    finally:
        sw.call_claude = real

    # The article body has to be in the prompt, or the post can invent claims.
    assert "Semantic caching stores answers by meaning" in seen["prompt"]
    assert "semantic caching" in seen["prompt"]
    assert "A post about semantic caching." in post


def test_linkedin_post_has_its_em_dashes_stripped():
    real = sw.call_claude
    sw.call_claude = lambda *a, **k: "One line — with an em dash.\n\nAnother line."
    try:
        post = sw.generate_linkedin_post("T", ARTICLE, {"keywords": {}})
    finally:
        sw.call_claude = real
    assert "—" not in post, post


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
