"""Tests for the controlled-vocabulary canonicalisation layer.

This layer is the fix for the extractor/classifier coupling bug: it maps the
richer phrasings a real LLM produces onto the fixed vocabulary the classifier
keys off. These tests pin that mapping.
"""

from triage.canonicalize import (
    canonicalize_signals,
    canonicalize_specialty,
)
from triage.schemas import ClinicalSignals, ExtractedFact


def _fact(value: str) -> ExtractedFact:
    return ExtractedFact(value=value, evidence_text=value, confidence=0.9)


def _signals(symptoms=None, red_flags=None) -> ClinicalSignals:
    return ClinicalSignals(
        symptoms=[_fact(s) for s in (symptoms or [])],
        red_flags=[_fact(f) for f in (red_flags or [])],
        extraction_confidence=0.9,
    )


# ---------------------------------------------------------------------------
# The core bug: richer LLM phrasings must map to canonical terms
# ---------------------------------------------------------------------------

def test_llm_phrasing_maps_to_canonical_skin_tag():
    out = canonicalize_signals(_signals(symptoms=["benign skin tag on the right forearm"]))
    assert out.symptoms[0].value == "skin tag"


def test_llm_phrasing_maps_to_canonical_rectal_bleeding():
    out = canonicalize_signals(_signals(red_flags=["blood per rectum for six weeks"]))
    assert out.red_flags[0].value == "rectal bleeding"


def test_synonym_dyspnoea_maps_to_shortness_of_breath():
    out = canonicalize_signals(_signals(symptoms=["dyspnoea on exertion"]))
    assert out.symptoms[0].value == "shortness of breath"


# ---------------------------------------------------------------------------
# Already-canonical input is unchanged (mock backend must still work)
# ---------------------------------------------------------------------------

def test_canonical_input_unchanged():
    out = canonicalize_signals(_signals(symptoms=["skin tag"], red_flags=["rectal bleeding"]))
    assert out.symptoms[0].value == "skin tag"
    assert out.red_flags[0].value == "rectal bleeding"


# ---------------------------------------------------------------------------
# Evidence text is preserved verbatim for the audit trail
# ---------------------------------------------------------------------------

def test_evidence_text_preserved_after_remap():
    original = "benign skin tag on the right forearm"
    out = canonicalize_signals(_signals(symptoms=[original]))
    assert out.symptoms[0].value == "skin tag"        # value canonicalised
    assert out.symptoms[0].evidence_text == original  # evidence untouched


# ---------------------------------------------------------------------------
# Unmappable terms are kept, not dropped (fail toward human review, not silence)
# ---------------------------------------------------------------------------

def test_unmappable_symptom_kept_not_dropped():
    out = canonicalize_signals(_signals(symptoms=["some unrecognised finding"]))
    assert len(out.symptoms) == 1
    assert out.symptoms[0].value == "some unrecognised finding"


# ---------------------------------------------------------------------------
# Specialty canonicalisation
# ---------------------------------------------------------------------------

def test_specialty_synonyms_map_to_canonical():
    assert canonicalize_specialty("derm") == "Dermatology"
    assert canonicalize_specialty("lower GI") == "Colorectal"
    assert canonicalize_specialty("cardiac clinic") == "Cardiology"


def test_unknown_specialty_returns_none():
    assert canonicalize_specialty("astrology") is None
