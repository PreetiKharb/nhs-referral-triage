"""Mock clinical signal extraction for the referral-triage POC.

Simulates what a real LLM extraction step would produce: structured clinical
facts, each with a supporting evidence span from the original letter text.

Design decisions:
- Every ExtractedFact carries evidence_text quoted from the input — no invented
  evidence, no facts without a source.
- Red flags are detected via rules.detect_red_flags, which handles negation.
  "No frank rectal bleeding" (REF-007) and "no chest pain at rest" (REF-008)
  must not produce red flag facts — the negation window in rules.py covers this.
- Hedged weight loss ("may be related to reduced appetite") is intentionally
  NOT emitted as a red flag here. That distinction is clinical, not mechanical.
  See assumptions log for why REF-004 / REF-007 gold labels hold.
- Confidence is set by content, not by word count alone: short or non-clinical
  letters get 0.20; clear clinical letters with flags get 0.85; otherwise 0.75.
"""

from __future__ import annotations

import re

from triage.rules import detect_red_flags, is_non_referral, is_too_short
from triage.schemas import ClinicalSignals, ExtractedFact, Priority, ReferralInput

# Symptom phrases to look for — maps canonical name to search phrase.
# This is keyword matching, not NLP; the mock LLM interface is the schema contract.
_SYMPTOM_PHRASES: list[str] = [
    "skin tag", "rash", "changing mole", "bleeding mole", "mole",
    "rectal bleeding", "weight loss", "chest pain", "shortness of breath",
    "breathlessness", "knee pain", "joint pain", "abdominal pain",
    "difficulty swallowing", "nausea", "bloating", "loose stools",
    "change in bowel habit", "epigastric pain",
]

_SUSPECTED_CONDITION_PHRASES: list[str] = [
    "suspected melanoma", "melanoma", "suspected cancer",
    "colorectal cancer", "angina", "eczema",
]

# GP urgency phrases mapped to their canonical value.
_GP_URGENCY_MAP: dict[str, str] = {
    "two week wait": "two week wait",
    "two-week wait": "two week wait",
    "2ww": "two week wait",
    "2 week wait": "two week wait",
    "urgent": "urgent",
    "routine": "routine",
}


def _find_span(text: str, phrase: str) -> str:
    """Return a short evidence snippet around phrase in text."""
    idx = text.lower().find(phrase.lower())
    if idx == -1:
        return phrase
    start = max(0, idx - 20)
    end = min(len(text), idx + len(phrase) + 60)
    return text[start:end].strip()


def _extract_facts(text: str, phrases: list[str]) -> list[ExtractedFact]:
    """Return one ExtractedFact per phrase found in text (negation-unaware here;
    caller uses detect_red_flags for negation-sensitive extraction)."""
    facts = []
    lower = text.lower()
    for phrase in phrases:
        if phrase in lower:
            facts.append(ExtractedFact(
                value=phrase,
                evidence_text=_find_span(text, phrase),
                confidence=0.85,
            ))
    return facts


def extract_signals(referral: ReferralInput) -> ClinicalSignals:
    """Extract structured clinical signals from a referral letter.

    Returns low-confidence signals for unusable inputs (too short, non-referral)
    so the policy gate can route them to EXCEPTION without needing to re-check
    the raw text itself — though policy also runs its own pre-filter as
    defense-in-depth.
    """
    text = referral.text

    # Unusable inputs: return minimal signals with low confidence.
    if is_too_short(text) or is_non_referral(text):
        return ClinicalSignals(
            symptoms=[],
            red_flags=[],
            extraction_confidence=0.20,
        )

    # Red flags via rules layer (handles negation).
    flag_names = detect_red_flags(text)
    red_flag_facts = [
        ExtractedFact(
            value=name,
            evidence_text=_find_span(text, name),
            confidence=0.90,
        )
        for name in flag_names
    ]

    symptoms = _extract_facts(text, _SYMPTOM_PHRASES)

    # Suspected condition.
    suspected = None
    for phrase in _SUSPECTED_CONDITION_PHRASES:
        if phrase in text.lower():
            suspected = ExtractedFact(
                value=phrase,
                evidence_text=_find_span(text, phrase),
                confidence=0.88,
            )
            break

    # GP stated urgency — first match wins.
    gp_urgency = None
    lower = text.lower()
    for phrase, canonical in _GP_URGENCY_MAP.items():
        if phrase in lower:
            gp_urgency = ExtractedFact(
                value=canonical,
                evidence_text=_find_span(text, phrase),
                confidence=0.90,
            )
            break

    # Confidence: higher when clear clinical signals present.
    has_signal = bool(red_flag_facts or suspected or len(symptoms) >= 2)
    extraction_confidence = 0.85 if has_signal else 0.75

    return ClinicalSignals(
        symptoms=symptoms,
        red_flags=red_flag_facts,
        suspected_condition=suspected,
        gp_stated_urgency=gp_urgency,
        extraction_confidence=extraction_confidence,
    )
