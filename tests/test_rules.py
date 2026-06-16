"""Tests for the deterministic safety rules — written before implementation.

The rules layer has two jobs:
  1. detect_red_flags: find concerning clinical signals in text, with negation
     awareness (REF-007 "No frank rectal bleeding", REF-008 "no chest pain at rest"
     are the regression cases for this).
  2. priority escalation: red flags can only raise priority, never lower it.

All priority ordering must use Priority.rank, never string comparison.
These tests fail until rules.py is implemented — that is expected.
"""

import pytest

from triage.rules import (
    detect_red_flags,
    is_non_referral,
    is_too_short,
    priority_max,
    red_flag_evidence_phrases,
    red_flag_priority_floor,
)
from triage.schemas import Priority


# ---------------------------------------------------------------------------
# is_too_short
# ---------------------------------------------------------------------------

def test_too_short_minimal_text():
    assert is_too_short("Please see patient.") is True


def test_too_short_empty():
    assert is_too_short("") is True


def test_too_short_false_for_real_referral():
    text = (
        "Dear Dermatology, I would be grateful for a routine review of Mr. Smith, "
        "a 54-year-old male with a benign skin tag on his right forearm. "
        "No red flags. Routine appointment please. Dr. Patel."
    )
    assert is_too_short(text) is False


# ---------------------------------------------------------------------------
# is_non_referral
# ---------------------------------------------------------------------------

def test_non_referral_admin_keywords():
    text = (
        "Hi Sarah, reminder that the staff rota for August needs to be submitted "
        "by Friday. Payroll deadline is the 28th. Holiday requests must be signed off."
    )
    assert is_non_referral(text) is True


def test_non_referral_invoice():
    assert is_non_referral("Please find attached the invoice for last month.") is True


def test_non_referral_false_for_clinical_letter():
    text = "Dear Cardiology, routine referral for Mrs. Fletcher, 63, with stable angina."
    assert is_non_referral(text) is False


# ---------------------------------------------------------------------------
# detect_red_flags — positive cases
# ---------------------------------------------------------------------------

def test_detects_changing_mole():
    flags = detect_red_flags("Patient has a changing mole on her shoulder.")
    assert any("mole" in f.lower() or "changing" in f.lower() for f in flags)


def test_detects_bleeding_mole():
    flags = detect_red_flags("The mole is bleeding and has developed irregular borders.")
    assert len(flags) > 0


def test_detects_rectal_bleeding():
    flags = detect_red_flags("He reports rectal bleeding over the past six weeks.")
    assert any("rectal bleeding" in f.lower() for f in flags)


def test_detects_weight_loss():
    flags = detect_red_flags("She has lost 6kg without trying over the past two months.")
    assert any("weight loss" in f.lower() for f in flags)


def test_detects_chest_pain():
    flags = detect_red_flags("Patient reports chest pain on exertion.")
    assert any("chest pain" in f.lower() for f in flags)


def test_detects_shortness_of_breath():
    flags = detect_red_flags("She has new shortness of breath on climbing stairs.")
    assert any("breath" in f.lower() for f in flags)


def test_detects_suspected_cancer():
    flags = detect_red_flags("I am concerned this may represent melanoma.")
    assert len(flags) > 0


# ---------------------------------------------------------------------------
# detect_red_flags — negation (regression tests for REF-007 and REF-008)
# ---------------------------------------------------------------------------

def test_negated_rectal_bleeding_not_detected():
    # REF-007: "No frank rectal bleeding" must not fire the rectal bleeding flag.
    flags = detect_red_flags("No frank rectal bleeding noted.")
    assert not any("rectal bleeding" in f.lower() for f in flags)


def test_negated_chest_pain_not_detected():
    # REF-008: "no chest pain at rest" must not fire the chest pain flag.
    flags = detect_red_flags("No chest pain at rest. No breathlessness at rest.")
    assert not any("chest pain" in f.lower() for f in flags)


def test_clean_letter_no_red_flags():
    # A routine skin tag referral should produce no flags at all.
    text = (
        "Routine referral for a skin tag. Asymptomatic. No concerning features. "
        "No change in size, no bleeding, no pigment change."
    )
    assert detect_red_flags(text) == []


# ---------------------------------------------------------------------------
# red_flag_evidence_phrases — the surface phrase that matched, so callers can
# quote real letter text as evidence (no invented evidence). The canonical flag
# name often differs from the surface form ("breathlessness" -> "shortness of
# breath"), so quoting the canonical name would fabricate a non-quote.
# ---------------------------------------------------------------------------

def test_evidence_phrase_is_present_in_text_for_synonym():
    text = "She has new breathlessness on climbing stairs."
    phrases = red_flag_evidence_phrases(text)
    assert phrases["shortness of breath"] == "breathlessness"
    assert phrases["shortness of breath"] in text.lower()


def test_evidence_phrase_keys_match_detect_red_flags():
    text = "He reports rectal bleeding and has lost weight without trying."
    assert sorted(red_flag_evidence_phrases(text)) == detect_red_flags(text)


def test_evidence_phrase_empty_when_no_flags():
    assert red_flag_evidence_phrases("Routine asymptomatic skin tag, no concerns.") == {}


# ---------------------------------------------------------------------------
# priority_max
# ---------------------------------------------------------------------------

def test_priority_max_picks_higher():
    assert priority_max(Priority.ROUTINE, Priority.URGENT) == Priority.URGENT


def test_priority_max_two_week_wait_beats_urgent():
    assert priority_max(Priority.URGENT, Priority.TWO_WEEK_WAIT) == Priority.TWO_WEEK_WAIT


def test_priority_max_same_returns_same():
    assert priority_max(Priority.ROUTINE, Priority.ROUTINE) == Priority.ROUTINE


def test_priority_max_uses_rank_not_string():
    # Prove ordering via .rank — if this test passes, we know rank was used.
    a, b = Priority.ROUTINE, Priority.TWO_WEEK_WAIT
    result = priority_max(a, b)
    assert result.rank >= a.rank
    assert result.rank >= b.rank


def test_priority_rank_ordering_is_correct():
    assert Priority.UNKNOWN.rank < Priority.ROUTINE.rank
    assert Priority.ROUTINE.rank < Priority.URGENT.rank
    assert Priority.URGENT.rank < Priority.TWO_WEEK_WAIT.rank


# ---------------------------------------------------------------------------
# red_flag_priority_floor
# ---------------------------------------------------------------------------

def test_cancer_pathway_flags_imply_two_week_wait():
    for flag in ["changing mole", "bleeding mole", "melanoma", "suspected cancer"]:
        assert red_flag_priority_floor([flag]) == Priority.TWO_WEEK_WAIT, flag


def test_chest_pain_implies_at_least_urgent():
    result = red_flag_priority_floor(["chest pain"])
    assert result.rank >= Priority.URGENT.rank


def test_shortness_of_breath_implies_at_least_urgent():
    result = red_flag_priority_floor(["shortness of breath"])
    assert result.rank >= Priority.URGENT.rank


def test_rectal_bleeding_and_weight_loss_together_imply_two_week_wait():
    result = red_flag_priority_floor(["rectal bleeding", "weight loss"])
    assert result == Priority.TWO_WEEK_WAIT


def test_empty_flags_return_routine():
    assert red_flag_priority_floor([]) == Priority.ROUTINE


def test_floor_takes_highest_across_mixed_flags():
    # chest pain (URGENT) + melanoma (TWO_WEEK_WAIT) → TWO_WEEK_WAIT
    result = red_flag_priority_floor(["chest pain", "melanoma"])
    assert result == Priority.TWO_WEEK_WAIT


def test_red_flag_with_routine_proposal_causes_escalation():
    # The safety guarantee: any red flag must produce a floor above ROUTINE.
    result = red_flag_priority_floor(["rectal bleeding"])
    assert result.rank > Priority.ROUTINE.rank
