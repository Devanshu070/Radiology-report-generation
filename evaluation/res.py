"""Local clinically weighted RES-style report comparison.

This implementation is a local evaluation proxy, not an official scorer or a
measure of clinical safety. Lower scores indicate greater text agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TypedDict
import unicodedata


TOP_HEADING = re.compile(r"^(FINDINGS|IMPRESSION)[ \t]*:[ \t]*", re.I | re.M)
FIELD_HEADING = re.compile(r"^([A-Za-z][A-Za-z0-9 /&()\-]{0,60}):[ \t]*", re.M)
STRUCTURE_HEADING = re.compile(r"^([A-Za-z][A-Za-z0-9 /(),.&+\-]{0,95}):[ \t]*", re.M)
CRITICAL_WORDS = set(
    "no not without absent negative left right bilateral both mild moderate severe acute chronic new stable increased decreased mm cm m ml percent".split()
)
FUNCTION_WORDS = set(
    "a an the and or of with to in on at by for from is are was were be been being as this that these those it its there".split()
)
UNIT_ALIASES = {
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "milliliter": "ml",
    "milliliters": "ml",
    "millilitre": "ml",
    "millilitres": "ml",
}


@dataclass(frozen=True)
class ParsedReport:
    """Parsed report content used by the local scoring heuristic."""

    fields: dict[str, str]
    unlabelled: str
    impression: str
    parents: set[str]


class ResScore(TypedDict):
    """Score components returned by :func:`res_proxy`."""

    res_proxy: float
    findings_proxy: float
    impression_proxy: float
    parse_error: str


def canonical_label(label: str) -> str:
    """Normalize a findings field label for comparison."""
    return " ".join(label.upper().split())


def parse_report(text: str, template_parents: set[str] | None = None) -> ParsedReport:
    """Parse a report containing exactly one FINDINGS and one IMPRESSION section.

    Empty nonterminal headings in the template are treated as parent headings.
    Supply ``template_parents`` when parsing a reference or prediction so the
    same hierarchy is used for all three reports.
    """
    headings = list(TOP_HEADING.finditer(text))
    if [heading.group(1).upper() for heading in headings] != ["FINDINGS", "IMPRESSION"]:
        raise ValueError("expected exactly one FINDINGS heading followed by one IMPRESSION heading")
    findings, impression = headings
    if text[:findings.start()].strip():
        raise ValueError("content before FINDINGS is not allowed")

    body = text[findings.end():impression.start()]
    matches = list(STRUCTURE_HEADING.finditer(body))
    raw_fields: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, match in enumerate(matches):
        label = canonical_label(match.group(1))
        if label in seen:
            raise ValueError(f"duplicate FINDINGS field: {label}")
        seen.add(label)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        raw_fields.append((label, body[match.end():end].strip()))

    parents = template_parents or {
        label for label, content in raw_fields[:-1]
        if not content and label != "OTHER FINDINGS"
    }
    scoring_fields: dict[str, list[str]] = {}
    loose: list[str] = []
    current: str | None = None
    awaiting_body = False
    for line in body.splitlines():
        match = FIELD_HEADING.match(line)
        if match:
            current = canonical_label(match.group(1))
            scoring_fields[current] = []
            tail = line[match.end():].strip()
            awaiting_body = not tail
            if tail:
                (loose if current in parents else scoring_fields[current]).append(tail)
        elif not line.strip():
            if not awaiting_body:
                current = None
        elif current is not None:
            (loose if current in parents else scoring_fields[current]).append(line.strip())
            awaiting_body = False
        else:
            loose.append(line.strip())

    fields = {
        label: " ".join(parts)
        for label, parts in scoring_fields.items()
        if label not in parents
    }
    return ParsedReport(fields, " ".join(loose), text[impression.end():].strip(), parents)


def score_tokens(text: str) -> list[str]:
    """Normalize report text into tokens, retaining clinical contrasts."""
    text = unicodedata.normalize("NFKC", text).lower().replace("−", "-")
    text = re.sub(r"(?<=\d)[ \t]*[-–—][ \t]*(?=\.?\d)", " to ", text)
    text = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s+", "", text, flags=re.M)
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", "", text)
    text = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", text)
    return [UNIT_ALIASES.get(token, token) for token in re.findall(r"[+-]?\d+(?:\.\d+)?|[a-z]+", text)]


def token_weight(token: str) -> float:
    """Return the heuristic token weight used by the edit-distance score."""
    if token in CRITICAL_WORDS or re.fullmatch(r"[+-]?\d+(?:\.\d+)?", token):
        return 4.0
    return 0.25 if token in FUNCTION_WORDS else 2.0


def weighted_word_edit(reference: str, prediction: str) -> float:
    """Return normalized weighted edit distance between two report fragments."""
    source, target = score_tokens(reference), score_tokens(prediction)
    if source == target:
        return 0.0
    source_weights = [token_weight(token) for token in source]
    target_weights = [token_weight(token) for token in target]
    denominator = max(sum(source_weights), sum(target_weights))
    if not denominator:
        return 0.0

    previous = [0.0]
    for weight in target_weights:
        previous.append(previous[-1] + weight)
    for source_index, source_token in enumerate(source):
        current = [previous[0] + source_weights[source_index]]
        for target_index, target_token in enumerate(target):
            substitution = (
                0.0 if source_token == target_token
                else max(source_weights[source_index], target_weights[target_index])
            )
            current.append(min(
                previous[target_index + 1] + source_weights[source_index],
                current[target_index] + target_weights[target_index],
                previous[target_index] + substitution,
            ))
        previous = current
    return min(1.0, previous[-1] / denominator)


def res_proxy(template_text: str, reference_text: str, prediction_text: str) -> ResScore:
    """Score a predicted report against a reference using the local RES proxy.

    Findings fields unchanged from the template receive weight 1; changed
    reference fields receive weight 3. Findings and impression scores are
    combined at 0.65 and 0.35 respectively. A malformed prediction receives
    the maximum score. Malformed templates or references raise ``ValueError``.
    """
    template = parse_report(template_text)
    reference = parse_report(reference_text, template.parents)
    try:
        prediction = parse_report(prediction_text, template.parents)
    except ValueError as error:
        return {
            "res_proxy": 1.0,
            "findings_proxy": 1.0,
            "impression_proxy": 1.0,
            "parse_error": str(error),
        }

    numerator = denominator = 0.0
    for label, reference_value in reference.fields.items():
        weight = 1 if score_tokens(reference_value) == score_tokens(template.fields.get(label, "")) else 3
        numerator += weight * weighted_word_edit(reference_value, prediction.fields.get(label, ""))
        denominator += weight
    if reference.unlabelled:
        numerator += weighted_word_edit(reference.unlabelled, prediction.unlabelled)
        denominator += 1

    unexpected = set(prediction.fields) - set(reference.fields)
    if prediction.unlabelled and not reference.unlabelled:
        unexpected.add("__UNLABELLED__")
    findings = min(1.0, (numerator + len(unexpected)) / max(denominator, 1.0))
    impression = weighted_word_edit(reference.impression, prediction.impression)
    return {
        "res_proxy": 0.65 * findings + 0.35 * impression,
        "findings_proxy": findings,
        "impression_proxy": impression,
        "parse_error": "",
    }
