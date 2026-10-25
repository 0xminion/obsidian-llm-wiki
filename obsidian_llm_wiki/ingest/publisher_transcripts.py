"""Conservative acquisition of explicitly labelled episode transcript links.

This is not web discovery: only links in an already matched episode are used.
Retrieval, public-IP/redirect validation and normalization stay in the shared
transcript resolver. An HTTP 200 or a long article alone is never sufficient.
"""
from __future__ import annotations

import re
from dataclasses import replace
from html.parser import HTMLParser
from urllib.parse import urlparse

from obsidian_llm_wiki.ingest.transcript_resolver import TranscriptResult, fetch_public_transcript

_LABEL = r"(?:full\s+)?transcript"
_LABELLED_URL = re.compile(
    rf"(?:\b{_LABEL}\s*:\s*(?:\[[^\]\n]+\]\(\s*|<)?"
    rf"|\[{_LABEL}\]\(\s*)(https?://[^\s<>\[\]()]+)",
    re.I,
)
_TURN = re.compile(
    r"(?m)^[ \t]*(?P<speaker>[^\W\d_][\w .’'-]{0,60})[ \t]*:?[ \t]*\n"
    r"\s*\((?P<time>\d{1,2}:\d{2}(?::\d{2})?)\)[ \t]*",
)


class _NotesParser(HTMLParser):
    """Expose RSS anchors as Markdown without guessing their destination."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.href = ""
        self.hidden = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.hidden:
            return
        if tag in {"script", "style", "template", "noscript"}:
            self.hidden = tag
            return
        if tag == "a":
            self.href = dict(attrs).get("href") or ""
            self.parts.append("[")
        elif tag in {"br", "p", "div", "li"}:
            self.parts.append("\n")
        elif tag.startswith(("http:", "https:")):
            self.parts.append(self.get_starttag_text() or "")

    def handle_endtag(self, tag: str) -> None:
        if self.hidden:
            if tag == self.hidden:
                self.hidden = ""
            return
        if tag == "a":
            self.parts.append(f"]({self.href})")
            self.href = ""
        elif tag in {"p", "div", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def has_transcript_link(text: str) -> bool:
    """A linked transcript section is not inline speech, even if padded.

    Detect labels independently of retrieval eligibility: a blocked/unsupported
    destination or ambiguous links must not fall back to their surrounding prose.
    """
    parser = _NotesParser()
    parser.feed(text)
    parser.close()
    return bool(re.search(
        rf"\[{_LABEL}\]\s*\(|\b{_LABEL}\s*:\s*(?:\[|<?[a-z][a-z0-9+.-]*://)",
        "".join(parser.parts), re.I,
    ))


def _same_episode_heading(text: str, episode_title: str) -> bool:
    """Compare the full title, allowing an episode number in a show suffix."""
    heading = text.splitlines()[0].strip()
    if not re.match(r"(?i)^(?:full\s+)?transcript\s+(?:for\s+)?", heading):
        return False
    number = re.match(r"^(?:#|Episode\s+)(\d+)\s*[-–—:]?\s*", episode_title, re.I)
    if number:
        if set(re.findall(r"(?:#|\bEpisode\s+)(\d+)\b", heading, re.I)) != {number[1]}:
            return False
        episode_title = episode_title[number.end():]
    heading = re.sub(r"(?i)^(?:full\s+)?transcript\s+(?:for\s+)?", "", heading)
    heading = re.sub(r"^(?:#|Episode\s+)\d+\s*[-–—:]?\s*", "", heading, flags=re.I)
    heading = heading.split(" | ", 1)[0]

    def normalize(value: str) -> str:
        return " ".join(re.sub(r"[^\w\s]", " ", value.casefold()).split())

    return bool(episode_title) and normalize(heading) == normalize(episode_title)


def conflicting_episode_title(document_title: str, episode_title: str) -> bool:
    """Only explicit document headers can contradict the matched RSS episode.

    Generic publisher titles and absent headings are not conflicts. In
    particular, discussion of another episode in the speech is not identity.
    """
    number_pattern = r"(?:#|\bEpisode\s+)(\d+)\b"
    expected = set(re.findall(number_pattern, episode_title, re.I))
    for heading in document_title.splitlines():
        observed = set(re.findall(number_pattern, heading, re.I))
        if expected and observed and expected != observed:
            return True
        # A publisher may title the artifact "Transcript for Show #181" while
        # RSS uses an unnumbered episode title. That is an alias, not evidence
        # of a conflicting episode. Compare full titles only when both headers
        # explicitly identify the numbered episode; labelled links separately
        # require their stronger positive title/turn match.
        if (
            expected and observed
            and re.match(r"(?i)^(?:full\s+)?transcript\s+for\s+", heading)
            and not _same_episode_heading(heading, episode_title)
        ):
            return True
    return False


def fetch_labelled_transcript(
    notes: str, *, episode_title: str, language: str = "",
) -> TranscriptResult | None:
    """Fetch one unambiguous label; never crawl links found in the response.

    Duplicate labels are harmless, distinct targets are ambiguous. One candidate
    uses the shared retrieval cap of five redirects (at most six HTTP requests).
    Admission deliberately requires a matching transcript-title line and at
    least three substantive timestamped turns from two speakers; unsupported
    publisher layouts fail closed rather than becoming article/ASR lookalikes.
    """
    parser = _NotesParser()
    parser.feed(notes)
    parser.close()
    rendered = "".join(parser.parts)
    matches = list(_LABELLED_URL.finditer(rendered))
    # Unsupported nested link syntax must not become a request to a truncated URL.
    if any(rendered[match.end():match.end() + 1] in ("(", "[") for match in matches):
        return None
    if any(re.search(r"\[[^\]\n]*$", rendered[:match.start()]) for match in matches):
        return None  # A URL inside anchor text is not that anchor's destination.
    urls = list(dict.fromkeys(match[1] for match in matches))
    if len(urls) != 1:
        return None
    try:
        parsed = urlparse(urls[0])
        if parsed.username is not None or parsed.password is not None:
            return None
    except ValueError:
        return None
    result = fetch_public_transcript(urls[0], language=language)
    if result is None:
        return None
    if conflicting_episode_title(result.document_title, episode_title):
        return None
    if not _same_episode_heading(result.text, episode_title):
        return None
    turns = list(_TURN.finditer(result.text))
    if len(turns) < 3 or len({turn["speaker"].strip() for turn in turns}) < 2:
        return None
    if re.search(
        r"(?im)^\s*(?:(?:sign|log)\s*in\s+to\s+(?:read|view|access)|"
        r"subscribe\s+to\s+(?:read|view|access)|access denied|verify you are human)",
        result.text,
    ):
        return None
    # WordPress emits this footer outside its transcript body. Exclude it and
    # any following navigation/scripts, but preserve every intervening paragraph.
    footer = re.search(r"(?im)^\s*This entry was posted in\s", result.text[turns[-1].end():])
    end = turns[-1].end() + footer.start() if footer else len(result.text)
    utterances = [
        result.text[turn.end():turns[index + 1].start() if index + 1 < len(turns) else end]
        for index, turn in enumerate(turns)
    ]
    # Long page furniture and a timestamped contents list are not speech.
    if sum(len(utterance.strip()) >= 60 for utterance in utterances) < 3:
        return None
    return replace(
        result, text=result.text[turns[0].start():end].strip(),
        provider="publisher_transcript_link", timed=True, speaker_labeled=True,
    )
