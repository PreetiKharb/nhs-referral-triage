"""Clinical signal extraction for the referral-triage POC.

Two interchangeable backends behind one schema contract:

    extract_signals(referral, backend="mock")  -> deterministic keyword matching
    extract_signals(referral, backend="llm")   -> real Claude call, structured output

Both return the same ClinicalSignals model. This is the modularity claim made
concrete: the policy gate, rules, audit and eval downstream do not know or care
which backend produced the signals. Swapping mock for a real model changes
nothing past this function.

The LLM backend uses the Anthropic structured-outputs API (messages.parse with
a Pydantic output schema), so the model is *constrained* to return the same
shape the mock returns — it cannot free-text its way around the contract. If no
API key is present, the LLM backend falls back to the mock with a clear notice,
so the demo never depends on the network.

Design decisions (apply to both backends):
- Every ExtractedFact carries evidence_text quoted from the input — no invented
  evidence, no facts without a source.
- Negation must suppress red flags: "No frank rectal bleeding" (REF-007) and
  "no chest pain at rest" (REF-008) must not produce red flag facts.
- Hedged weight loss ("may be related to reduced appetite") is NOT a red flag.
  This is a clinical distinction; the keyword mock cannot make it, which is
  exactly why a real LLM extractor is the interesting upgrade (see README).

ASSERTED, NOT CALIBRATED: the mock's confidence numbers (0.85, 0.90, 0.20 etc.)
are hand-picked constants, not measured probabilities. A 0.90 here does NOT mean
"right 90% of the time" — only a real model with a calibrated output head can
claim that, and validating it needs a labelled holdout set (out of scope here).
The confidence-gating architecture is real; the specific numbers it gates on are
placeholders until calibrated. See README, "With more time".
"""

from __future__ import annotations

import os
import sys

from triage.rules import is_non_referral, is_too_short, red_flag_evidence_phrases
from triage.schemas import ClinicalSignals, ExtractedFact, ReferralInput

# Model used by the LLM backend. Structured outputs are GA on this model.
_LLM_MODEL = "claude-opus-4-8"


# ---------------------------------------------------------------------------
# Mock backend — deterministic keyword matching (no network, no API key)
# ---------------------------------------------------------------------------

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

_GP_URGENCY_MAP: dict[str, str] = {
    "two week wait": "two week wait",
    "two-week wait": "two week wait",
    "2ww": "two week wait",
    "2 week wait": "two week wait",
    "urgent": "urgent",
    "routine": "routine",
}


def _find_span(text: str, phrase: str) -> str:
    """Return a short evidence snippet quoted from text around phrase.

    Callers pass a phrase known to occur in the letter. If the exact phrase is
    not found as a substring (token-wise match across odd spacing/punctuation),
    anchor on its first word so the snippet is still a real span of the letter —
    never the bare phrase echoed back, which would be invented evidence.
    """
    lower = text.lower()
    idx = lower.find(phrase.lower())
    span_len = len(phrase)
    if idx == -1:
        first_word = phrase.split()[0]
        idx = lower.find(first_word)
        if idx == -1:
            return phrase  # unreachable for a matched phrase; defensive only
        span_len = len(first_word)
    start = max(0, idx - 20)
    end = min(len(text), idx + span_len + 60)
    return text[start:end].strip()


def _extract_facts(text: str, phrases: list[str]) -> list[ExtractedFact]:
    """Return one ExtractedFact per phrase found in text."""
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


def _extract_signals_mock(referral: ReferralInput) -> ClinicalSignals:
    """Deterministic keyword-based extraction. The POC default."""
    text = referral.text

    if is_too_short(text) or is_non_referral(text):
        return ClinicalSignals(symptoms=[], red_flags=[], extraction_confidence=0.20)

    # Quote the surface phrase that actually matched, not the canonical name —
    # "breathlessness" -> "shortness of breath" would otherwise put a phrase in
    # evidence_text that never appears in the letter. No fact without evidence.
    flag_phrases = red_flag_evidence_phrases(text)
    red_flag_facts = [
        ExtractedFact(value=name, evidence_text=_find_span(text, phrase), confidence=0.90)
        for name, phrase in sorted(flag_phrases.items())
    ]

    symptoms = _extract_facts(text, _SYMPTOM_PHRASES)

    suspected = None
    for phrase in _SUSPECTED_CONDITION_PHRASES:
        if phrase in text.lower():
            suspected = ExtractedFact(
                value=phrase, evidence_text=_find_span(text, phrase), confidence=0.88,
            )
            break

    gp_urgency = None
    lower = text.lower()
    for phrase, canonical in _GP_URGENCY_MAP.items():
        if phrase in lower:
            gp_urgency = ExtractedFact(
                value=canonical, evidence_text=_find_span(text, phrase), confidence=0.90,
            )
            break

    has_signal = bool(red_flag_facts or suspected or len(symptoms) >= 2)
    extraction_confidence = 0.85 if has_signal else 0.75

    return ClinicalSignals(
        symptoms=symptoms,
        red_flags=red_flag_facts,
        suspected_condition=suspected,
        gp_stated_urgency=gp_urgency,
        extraction_confidence=extraction_confidence,
    )


# ---------------------------------------------------------------------------
# LLM backend — real Claude call, constrained to the ClinicalSignals schema
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """You are a clinical signal extractor for an NHS GP referral triage system.

Extract structured clinical signals from the referral letter. You MUST follow these rules:

1. EVIDENCE GROUNDING: every fact you extract must include evidence_text quoted
   verbatim from the letter. Never invent a fact that is not supported by a span
   of the letter text. If you cannot quote it, do not extract it.

2. NEGATION: do NOT extract a red flag or symptom that the letter explicitly
   negates or rules out. "No rectal bleeding", "denies chest pain", "no
   breathlessness at rest" mean the sign is ABSENT — do not record it as present.

3. HEDGED FINDINGS: weight loss that the GP attributes to another cause (e.g.
   "weight loss likely due to reduced appetite") is NOT an unexplained-weight-loss
   red flag. Only record weight loss as a red flag when it is described as
   unexplained or unintentional.

4. RED FLAGS are the suspected-cancer / urgent signals: changing or bleeding mole,
   suspected melanoma or cancer, unexplained rectal bleeding, unexplained weight
   loss, chest pain, new shortness of breath, haemoptysis.

5. CONFIDENCE: set extraction_confidence to reflect how clearly the letter
   supports the signals. A short or non-clinical letter should score low (< 0.4).

Return only the structured signals. Do not add commentary."""


def _extract_signals_llm(referral: ReferralInput) -> ClinicalSignals:
    """Real Claude extraction via the structured-outputs API.

    Falls back to the mock backend (with a stderr notice) if the anthropic SDK
    is not installed or ANTHROPIC_API_KEY is not set, so a demo never fails live.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Load a local .env if present (gitignored). Optional dependency: if
        # python-dotenv is not installed we simply rely on the real environment.
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        except ImportError:
            pass

    if not api_key:
        print("[extract] No ANTHROPIC_API_KEY found — falling back to mock backend.",
              file=sys.stderr)
        return _extract_signals_mock(referral)

    try:
        from anthropic import Anthropic
    except ImportError:
        print("[extract] anthropic SDK not installed — falling back to mock backend.",
              file=sys.stderr)
        return _extract_signals_mock(referral)

    client = Anthropic(api_key=api_key)
    response = client.messages.parse(
        model=_LLM_MODEL,
        max_tokens=1024,
        system=_EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": referral.text}],
        output_format=ClinicalSignals,
    )
    # parsed_output is a validated ClinicalSignals instance — the schema contract
    # is enforced by the API, not by hopeful parsing of free text.
    return response.parsed_output


# ---------------------------------------------------------------------------
# Dispatcher — the only function the pipeline calls
# ---------------------------------------------------------------------------

def extract_signals(referral: ReferralInput, backend: str = "mock") -> ClinicalSignals:
    """Extract clinical signals using the chosen backend.

    backend="mock" (default) — deterministic, offline, used everywhere by default.
    backend="llm"            — real Claude call, falls back to mock if unavailable.

    Both return an identical ClinicalSignals shape; nothing downstream changes.
    """
    if backend == "llm":
        return _extract_signals_llm(referral)
    return _extract_signals_mock(referral)
