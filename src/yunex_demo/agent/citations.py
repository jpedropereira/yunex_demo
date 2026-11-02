"""The grounding check: does the page the agent cites really say what it claims?

A model can invent a page number as easily as anything else, and it can cite a real page
for something that page never says. So each citation is compared with the text of the
pages search_filing actually returned, by plain code, and shown next to the closest passage.

Matching words can't judge meaning: "none" in place of "all" still matches. So this never
says a claim is verified. It finds the passage, and a person reads it.
"""

import json
import re
from dataclasses import dataclass, field
from urllib.parse import quote

from yunex_demo.agent.hooks import COMPANY_ALIASES
from yunex_demo.agent.quotes import plain_quotes
from yunex_demo.agent.tools import RetrievedPage
from yunex_demo.sec import filing
from yunex_demo.workflow.validators import numbers_in

# Inside the brackets, so one pair can hold several: "(Apple 10-K, p. 12; NVIDIA 10-K, p. 3)".
BRACKETS = re.compile(r"\(([^()]*10-K[^()]*)\)")
# A citation can list several pages, in the model's own wording: "p. 10, p. 26" or "pp. 10, 26".
CITATION = re.compile(r"([\w.]+) 10-K,((?:\s*(?:pp?\.\s*)?\d+,?)+)")

PASSAGE_FOUND = "passage_found"
NO_CLOSE_PASSAGE = "no_close_passage"
NOT_RETRIEVED = "not_retrieved"

# Measured on 55 citations from real answers, across all five companies. One scored below
# this, and it really was the wrong page. The same claims scored at or above it on just 5 of
# 330 randomly chosen pages from the same filing, so it can still pass a wrong page.
MATCH_THRESHOLD = 0.4

# Not "U.S. and" or "Corp. a": a sentence ends at punctuation followed by a capital or bullet.
SENTENCE_END = re.compile(r"(?<![A-Z]\.)(?<=[.;!?])\s+(?=[A-Z•])")
MIN_WORDS_PER_SENTENCE = 4
# Two, because a claim can sum up neighbouring sentences.
MOST_SENTENCES_IN_A_PASSAGE = 2

WORD = re.compile(r"[a-z0-9][a-z0-9'-]*")
# Words that say little about what a claim means, and would make any two sentences look alike.
_COMMON_WORDS_TEXT = (
    "a also an and are as at be been by can could for from had has have in is it its may "
    "not of on or our such that the their they this to was we were which will with would"
)
COMMON_WORDS = set(_COMMON_WORDS_TEXT.split())

# A bold list label such as "**Supply chain:**", which is the model's wording and not its claim.
BOLD_LABEL = re.compile(r"\*\*[^*]*:\s*\*\*")
QUOTED_TEXT = re.compile(r'"([^"]+)"')

# How much more a missing claim word counts against a passage than an extra passage word.
# Filing sentences run long, such as a list of competitors the claim only draws on, and
# treating the two alike (F1) flagged 4 more of those 55 citations.
CLAIM_WORDS_WEIGHT = 9

# Enough to be found only once, and few enough to stay inside one paragraph of the filing.
LINK_WORDS = 10


@dataclass
class Evidence:
    """One citation, and what the cited page says about the claim it supports."""

    cite: str
    claim: str
    status: str
    passage: str = ""
    page_text: str = ""
    source_url: str = ""
    # Set when the wording is on another page the agent read.
    better_cite: str = ""
    better_passage: str = ""
    missing_numbers: list[str] = field(default_factory=list)
    misquotes: list[str] = field(default_factory=list)

    @property
    def needs_attention(self) -> bool:
        return self.status != PASSAGE_FOUND or bool(self.missing_numbers or self.misquotes)


def _words(text: str) -> set[str]:
    # Commas go first, so that "223,000" stays one word and only matches itself.
    return set(WORD.findall(plain_quotes(text).lower().replace(",", ""))) - COMMON_WORDS


def _overlap(claim_words: set[str], passage_words: set[str]) -> float:
    """An F-score of the shared words. Only the claim's share would favour the longest passage."""
    shared = len(claim_words & passage_words)
    weighted_size = CLAIM_WORDS_WEIGHT * len(claim_words) + len(passage_words)
    return (CLAIM_WORDS_WEIGHT + 1) * shared / weighted_size if shared else 0.0


def closest_passage(claim: str, page_text: str) -> tuple[float, str]:
    """The sentence or two on this page sharing the most words with the claim, and its score."""
    claim_words = _words(claim)
    sentences = [
        sentence
        for sentence in SENTENCE_END.split(page_text)
        if len(sentence.split()) >= MIN_WORDS_PER_SENTENCE
    ]

    best_score = 0.0
    best_passage = ""
    for start in range(len(sentences)):
        for count in range(1, MOST_SENTENCES_IN_A_PASSAGE + 1):
            passage = " ".join(sentences[start : start + count])
            score = _overlap(claim_words, _words(passage))
            if score > best_score:
                best_score = score
                best_passage = passage
    return best_score, best_passage


def _claim_before(text: str) -> str:
    """The last sentence of the text leading up to a citation, without its markdown."""
    text = BOLD_LABEL.sub("", text)
    text = re.sub(r"[*#|`]", "", text).strip(" \n.;,")
    last_line = text.splitlines()[-1] if text else ""
    return SENTENCE_END.split(last_line)[-1].strip()


def _missing_numbers(claim: str, page_text: str) -> list[str]:
    missing = numbers_in(claim) - numbers_in(page_text)
    return [f"{number:,}" for number in sorted(missing)]


def _misquotes(claim: str, page_text: str) -> list[str]:
    """Text in quotation marks is a promise of the exact words, so test it for exactly that."""
    page = " ".join(plain_quotes(page_text).lower().split())
    return [
        quoted
        for quoted in QUOTED_TEXT.findall(plain_quotes(claim))
        if " ".join(quoted.lower().split()) not in page
    ]


def _check_citation(
    cite: str,
    claim: str,
    ticker: str | None,
    page_number: int,
    retrieved_pages: dict[tuple[str, int], RetrievedPage],
) -> Evidence:
    evidence = Evidence(cite, claim, NOT_RETRIEVED)

    page = retrieved_pages.get((ticker, page_number))
    if page:
        score, evidence.passage = closest_passage(claim, page.text)
        evidence.status = PASSAGE_FOUND if score >= MATCH_THRESHOLD else NO_CLOSE_PASSAGE
        evidence.page_text = page.text
        evidence.source_url = page.source_url
        evidence.missing_numbers = _missing_numbers(claim, page.text)
        evidence.misquotes = _misquotes(claim, page.text)

    if evidence.status == PASSAGE_FOUND:
        return evidence

    # The page doesn't back the claim, so say which page the agent read does, if any.
    best_score = MATCH_THRESHOLD
    for (other_ticker, _), other in retrieved_pages.items():
        if other_ticker == ticker and other is not page:
            score, passage = closest_passage(claim, other.text)
            if score >= best_score:
                best_score = score
                evidence.better_cite = other.cite
                evidence.better_passage = passage
    return evidence


def check_answer(
    answer: str, retrieved_pages: dict[tuple[str, int], RetrievedPage]
) -> list[Evidence]:
    """One Evidence for each citation in the answer, the same claim and page only once."""
    results = []
    seen = set()
    text_starts_at = 0
    for brackets in BRACKETS.finditer(answer):
        claim = _claim_before(answer[text_starts_at : brackets.start()])
        text_starts_at = brackets.end()
        for company, pages in CITATION.findall(brackets[1]):
            ticker = COMPANY_ALIASES.get(company.lower())
            for page in re.findall(r"\d+", pages):
                cite = f"{company} 10-K, p. {page}"
                if (cite, claim) not in seen:
                    seen.add((cite, claim))
                    results.append(_check_citation(cite, claim, ticker, int(page), retrieved_pages))
    return results


def pages_read(
    steps: list, retrieved_pages: dict[tuple[str, int], RetrievedPage]
) -> list[RetrievedPage]:
    """The pages this turn's searches returned, for an answer that cites none of them."""
    page_by_cite = {page.cite: page for page in retrieved_pages.values()}
    cites = []
    for step in steps:
        if step.tool == "search_filing" and step.decision == "allow":
            for row in json.loads(step.result).get("results", []):
                cites.append(row["cite"])
    # Through a dict, which drops repeats and keeps the order the agent read them in.
    return [page_by_cite[cite] for cite in dict.fromkeys(cites)]


def filing_link(source_url: str, passage: str = "") -> str:
    """A link to the filing that scrolls to the passage, or "" if the URL isn't EDGAR's.

    A text fragment: browsers find these words on the page and highlight them, and open the
    filing at the top if they can't. It takes words from the passage's middle, because a
    fragment can't cross a block of the page, and the edges are where a heading can sit.
    """
    try:
        filing.parse_edgar_url(source_url)
    except ValueError:
        return ""
    if not passage:
        return source_url

    words = passage.split()
    start = max(0, (len(words) - LINK_WORDS) // 2)
    # quote() leaves "-" alone, but it has a meaning of its own inside a text fragment.
    target = quote(" ".join(words[start : start + LINK_WORDS]), safe="").replace("-", "%2D")
    return f"{source_url}#:~:text={target}"
