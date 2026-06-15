"""Tests for the policy gate — written before policy.py is implemented.

The policy gate is the main safety boundary: it takes a model Proposal and
decides what action is actually allowed. Every routing decision must carry a
non-empty reason and have every override visible in rules_fired.

Decision tree, in evaluation order:
  1-2:   Text pre-filter — EXCEPTION before any signal/confidence check
  3:     Precedence test — pre-filter beats a high-confidence proposal
  4-8:   Quality gates — uncertain extraction or classification → HUMAN_REVIEW
  9:     Safety floor — red flag + Routine is a contradiction → HUMAN_REVIEW
  10:    Escalated recommended_priority when red flags present
  11-12: MVP scope — Urgent and Two-Week Wait always → HUMAN_REVIEW
  13:    The only AUTO_ROUTE path
  14:    AUTO_ROUTE has empty rules_fired
  15-16: Threshold boundary — pin >= semantics at exactly 0.80 / 0.70
  17:    Safety property — red-flag cases must never AUTO_ROUTE
  18:    Fail-closed — empty/whitespace text → EXCEPTION
  19-20: reason is always non-empty (EXCEPTION and HUMAN_REVIEW)
"""

import pytest

from triage.policy import apply_policy
from triage.schemas import (
    UNKNOWN_SPECIALTY,
    ClinicalSignals,
    ExtractedFact,
    Priority,
    Proposal,
    ReferralInput,
    Tier,
)

# ---------------------------------------------------------------------------
# Injected thresholds — tests must not depend on loading config from disk
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "version": "thresholds-v0.1",
    "min_extraction_confidence": 0.70,
    "min_specialty_confidence": 0.80,
    "min_priority_confidence": 0.80,
    "min_referral_words": 15,
}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

VALID_REFERRAL_TEXT = (
    "Dear Dermatology, please see this 54-year-old male with a benign skin tag "
    "on his right forearm. No red flags. Routine appointment please. Dr Patel."
)


def make_referral(text: str = VALID_REFERRAL_TEXT) -> ReferralInput:
    return ReferralInput(referral_id="TEST-001", text=text, source="test")


def make_signals(
    red_flags: list[str] | None = None,
    extraction_confidence: float = 0.9,
) -> ClinicalSignals:
    flags = [
        ExtractedFact(value=f, evidence_text=f"...{f}...", confidence=0.9)
        for f in (red_flags or [])
    ]
    return ClinicalSignals(
        symptoms=[],
        red_flags=flags,
        suspected_condition=None,
        gp_stated_urgency=None,
        extraction_confidence=extraction_confidence,
    )


def make_proposal(
    specialty: str = "Dermatology",
    priority: Priority = Priority.ROUTINE,
    specialty_confidence: float = 0.9,
    priority_confidence: float = 0.9,
) -> Proposal:
    return Proposal(
        specialty=specialty,
        specialty_confidence=specialty_confidence,
        priority=priority,
        priority_confidence=priority_confidence,
        alternatives=[],
        evidence=[],
    )


# ---------------------------------------------------------------------------
# 1-2: Text pre-filter — EXCEPTION before signals or confidence are checked.
# Fixtures use realistic signals (low confidence, empty) because that is what
# the extraction layer actually produces for unusable inputs.
# ---------------------------------------------------------------------------

def test_too_short_returns_exception():
    decision = apply_policy(
        make_referral("Please see patient."),
        make_signals(extraction_confidence=0.2),
        make_proposal(specialty=UNKNOWN_SPECIALTY, priority=Priority.UNKNOWN,
                      specialty_confidence=0.2, priority_confidence=0.2),
        THRESHOLDS,
    )
    assert decision.tier == Tier.EXCEPTION


def test_non_referral_returns_exception():
    decision = apply_policy(
        make_referral("Hi Sarah, staff rota for August due Friday. Payroll deadline 28th."),
        make_signals(extraction_confidence=0.2),
        make_proposal(specialty=UNKNOWN_SPECIALTY, priority=Priority.UNKNOWN,
                      specialty_confidence=0.2, priority_confidence=0.2),
        THRESHOLDS,
    )
    assert decision.tier == Tier.EXCEPTION


# ---------------------------------------------------------------------------
# 3: Precedence — text pre-filter fires before the confidence gate.
# This is the defense-in-depth check: even if upstream extraction returned an
# unrealistically high-confidence proposal for an HR email, policy catches it.
# ---------------------------------------------------------------------------

def test_non_referral_text_returns_exception_even_with_high_confidence_proposal():
    decision = apply_policy(
        make_referral("Hi Sarah, staff rota for August due Friday. Payroll deadline 28th."),
        make_signals(extraction_confidence=0.95),  # unrealistically high — tests ordering
        make_proposal(specialty_confidence=0.95, priority_confidence=0.95),
        THRESHOLDS,
    )
    assert decision.tier == Tier.EXCEPTION  # pre-filter beats confidence gate


# ---------------------------------------------------------------------------
# 4-8: Quality gates
# ---------------------------------------------------------------------------

def test_low_extraction_confidence_returns_human_review():
    decision = apply_policy(
        make_referral(),
        make_signals(extraction_confidence=0.50),
        make_proposal(),
        THRESHOLDS,
    )
    assert decision.tier == Tier.HUMAN_REVIEW


def test_unknown_specialty_returns_human_review():
    decision = apply_policy(
        make_referral(),
        make_signals(),
        make_proposal(specialty=UNKNOWN_SPECIALTY),
        THRESHOLDS,
    )
    assert decision.tier == Tier.HUMAN_REVIEW


def test_unknown_priority_returns_human_review():
    decision = apply_policy(
        make_referral(),
        make_signals(),
        make_proposal(priority=Priority.UNKNOWN),
        THRESHOLDS,
    )
    assert decision.tier == Tier.HUMAN_REVIEW


def test_low_specialty_confidence_returns_human_review():
    decision = apply_policy(
        make_referral(),
        make_signals(),
        make_proposal(specialty_confidence=0.60),
        THRESHOLDS,
    )
    assert decision.tier == Tier.HUMAN_REVIEW


def test_low_priority_confidence_returns_human_review():
    decision = apply_policy(
        make_referral(),
        make_signals(),
        make_proposal(priority_confidence=0.60),
        THRESHOLDS,
    )
    assert decision.tier == Tier.HUMAN_REVIEW


# ---------------------------------------------------------------------------
# 9-10: Safety floor — red flags + Routine is a contradiction
# ---------------------------------------------------------------------------

def test_red_flags_with_routine_proposal_returns_human_review():
    decision = apply_policy(
        make_referral(),
        make_signals(red_flags=["rectal bleeding"]),
        make_proposal(priority=Priority.ROUTINE),
        THRESHOLDS,
    )
    assert decision.tier == Tier.HUMAN_REVIEW
    assert any("red_flag" in r.lower() for r in decision.rules_fired)


def test_red_flags_escalate_recommended_priority_above_routine():
    decision = apply_policy(
        make_referral(),
        make_signals(red_flags=["rectal bleeding"]),
        make_proposal(priority=Priority.ROUTINE),
        THRESHOLDS,
    )
    assert decision.recommended_priority.rank > Priority.ROUTINE.rank


# ---------------------------------------------------------------------------
# 11-12: MVP scope — Urgent and Two-Week Wait always go to human review
# ---------------------------------------------------------------------------

def test_urgent_returns_human_review_in_mvp():
    decision = apply_policy(
        make_referral(),
        make_signals(),
        make_proposal(priority=Priority.URGENT),
        THRESHOLDS,
    )
    assert decision.tier == Tier.HUMAN_REVIEW


def test_two_week_wait_returns_human_review_in_mvp():
    decision = apply_policy(
        make_referral(),
        make_signals(),
        make_proposal(priority=Priority.TWO_WEEK_WAIT),
        THRESHOLDS,
    )
    assert decision.tier == Tier.HUMAN_REVIEW


# ---------------------------------------------------------------------------
# 13-14: The only AUTO_ROUTE path
# ---------------------------------------------------------------------------

def test_clean_routine_high_confidence_no_flags_returns_auto_route():
    decision = apply_policy(
        make_referral(),
        make_signals(extraction_confidence=0.95),
        make_proposal(
            specialty="Dermatology",
            priority=Priority.ROUTINE,
            specialty_confidence=0.92,
            priority_confidence=0.91,
        ),
        THRESHOLDS,
    )
    assert decision.tier == Tier.AUTO_ROUTE
    assert decision.safe_to_auto_route is True


def test_auto_route_has_empty_rules_fired():
    # A clean auto-route must have no overrides — nothing changed the proposal.
    decision = apply_policy(
        make_referral(),
        make_signals(extraction_confidence=0.95),
        make_proposal(
            specialty="Dermatology",
            priority=Priority.ROUTINE,
            specialty_confidence=0.92,
            priority_confidence=0.91,
        ),
        THRESHOLDS,
    )
    assert decision.rules_fired == []


# ---------------------------------------------------------------------------
# 15-16: Threshold boundary — pin >= semantics
# Exactly at threshold must PASS the gate; one unit below must FAIL.
# ---------------------------------------------------------------------------

def test_specialty_confidence_exactly_at_threshold_passes():
    decision = apply_policy(
        make_referral(),
        make_signals(),
        make_proposal(specialty_confidence=0.80, priority_confidence=0.90),
        THRESHOLDS,
    )
    # Should not be rejected for specialty confidence — 0.80 meets the floor.
    assert "specialty_confidence" not in " ".join(decision.rules_fired).lower()


def test_specialty_confidence_one_below_threshold_fails():
    decision = apply_policy(
        make_referral(),
        make_signals(),
        make_proposal(specialty_confidence=0.79, priority_confidence=0.90),
        THRESHOLDS,
    )
    assert decision.tier == Tier.HUMAN_REVIEW


def test_extraction_confidence_exactly_at_threshold_passes():
    decision = apply_policy(
        make_referral(),
        make_signals(extraction_confidence=0.70),
        make_proposal(),
        THRESHOLDS,
    )
    assert "extraction_confidence" not in " ".join(decision.rules_fired).lower()


def test_extraction_confidence_one_below_threshold_fails():
    decision = apply_policy(
        make_referral(),
        make_signals(extraction_confidence=0.69),
        make_proposal(),
        THRESHOLDS,
    )
    assert decision.tier == Tier.HUMAN_REVIEW


# ---------------------------------------------------------------------------
# 17: Safety property — red-flag cases must never AUTO_ROUTE
# ---------------------------------------------------------------------------

def test_red_flag_case_never_auto_routes():
    for flag in ["rectal bleeding", "melanoma", "chest pain", "changing mole"]:
        decision = apply_policy(
            make_referral(),
            make_signals(red_flags=[flag], extraction_confidence=0.95),
            make_proposal(
                specialty="Dermatology",
                priority=Priority.ROUTINE,
                specialty_confidence=0.95,
                priority_confidence=0.95,
            ),
            THRESHOLDS,
        )
        assert decision.tier != Tier.AUTO_ROUTE, f"'{flag}' must never AUTO_ROUTE"
        assert decision.safe_to_auto_route is False


# ---------------------------------------------------------------------------
# 18: Fail-closed — empty or whitespace-only text → EXCEPTION
# ---------------------------------------------------------------------------

def test_empty_text_returns_exception():
    decision = apply_policy(
        make_referral(""),
        make_signals(extraction_confidence=0.0),
        make_proposal(specialty=UNKNOWN_SPECIALTY, priority=Priority.UNKNOWN,
                      specialty_confidence=0.0, priority_confidence=0.0),
        THRESHOLDS,
    )
    assert decision.tier == Tier.EXCEPTION


def test_whitespace_only_text_returns_exception():
    decision = apply_policy(
        make_referral("     "),
        make_signals(extraction_confidence=0.0),
        make_proposal(specialty=UNKNOWN_SPECIALTY, priority=Priority.UNKNOWN,
                      specialty_confidence=0.0, priority_confidence=0.0),
        THRESHOLDS,
    )
    assert decision.tier == Tier.EXCEPTION


# ---------------------------------------------------------------------------
# 19-20: reason is always non-empty — the audit trail property
# ---------------------------------------------------------------------------

def test_exception_decision_has_non_empty_reason():
    decision = apply_policy(
        make_referral("Please see patient."),
        make_signals(extraction_confidence=0.2),
        make_proposal(specialty=UNKNOWN_SPECIALTY, priority=Priority.UNKNOWN,
                      specialty_confidence=0.2, priority_confidence=0.2),
        THRESHOLDS,
    )
    assert decision.tier == Tier.EXCEPTION
    assert decision.reason.strip() != ""


def test_human_review_decision_has_non_empty_reason():
    decision = apply_policy(
        make_referral(),
        make_signals(extraction_confidence=0.50),
        make_proposal(),
        THRESHOLDS,
    )
    assert decision.tier == Tier.HUMAN_REVIEW
    assert decision.reason.strip() != ""
