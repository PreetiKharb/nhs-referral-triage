"""Policy gate for the referral-triage POC.

The model proposes; this layer governs. Takes extraction signals and a
classification proposal and returns an auditable Decision.

Evaluation order — every step can only send a case to HUMAN_REVIEW or EXCEPTION,
never downgrade from a higher-severity outcome already reached:

  1. Text pre-filter     (EXCEPTION) — runs before any signal/confidence check.
                          Defense-in-depth: even if upstream returned high-confidence
                          signals for junk input, the text gate catches it.
  2. Extraction quality  (HUMAN_REVIEW) — low extraction confidence.
  3. Unknown outputs     (HUMAN_REVIEW) — specialty or priority could not be determined.
  4. Confidence gates    (HUMAN_REVIEW) — specialty or priority confidence below threshold.
  5. Red flag floor      (HUMAN_REVIEW) — any red flag present; recommended_priority
                          is escalated to max(proposal, red_flag_floor). A Routine
                          proposal with red flags is a contradiction the model made;
                          the rule fires and the case goes to human review.
  6. MVP scope           (HUMAN_REVIEW) — Urgent and Two-Week Wait are not auto-routed
                          in this release; they always go to human review.
  7. AUTO_ROUTE          — clean, Routine, above all thresholds, no red flags.

Threshold semantics: comparison is strict less-than (<), so a confidence score
exactly equal to the threshold passes. `0.80 >= 0.80` passes the gate.

Every non-AUTO_ROUTE decision carries a non-empty reason. Every rule that
changed routing appears in rules_fired.
"""

from __future__ import annotations

import yaml

from triage.rules import is_non_referral, is_too_short, priority_max, red_flag_priority_floor
from triage.schemas import (
    UNKNOWN_SPECIALTY,
    ClinicalSignals,
    Decision,
    Priority,
    Proposal,
    ReferralInput,
    Tier,
)

RULESET_VERSION = "rules-v0.1"
PIPELINE_VERSION = "poc-v0.1"


def load_thresholds(path: str = "config/thresholds.yaml") -> dict:
    """Load routing thresholds from config. Version string is returned in the dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def apply_policy(
    referral: ReferralInput,
    signals: ClinicalSignals,
    proposal: Proposal,
    thresholds: dict | None = None,
) -> Decision:
    """Apply deterministic routing policy and return an auditable Decision.

    thresholds is injectable for testing — callers that do not pass it get the
    config loaded from disk. Tests must always inject thresholds explicitly.
    """
    if thresholds is None:
        thresholds = load_thresholds()

    min_words: int = thresholds.get("min_referral_words", 15)
    min_extraction: float = thresholds["min_extraction_confidence"]
    min_specialty: float = thresholds["min_specialty_confidence"]
    min_priority: float = thresholds["min_priority_confidence"]

    # ------------------------------------------------------------------
    # 1. Text pre-filter — EXCEPTION before any confidence check
    # ------------------------------------------------------------------
    if len(referral.text.split()) < min_words:
        return Decision(
            tier=Tier.EXCEPTION,
            recommended_specialty=None,
            recommended_priority=Priority.UNKNOWN,
            reason="Input too short to contain a routable referral.",
            rules_fired=["too_short"],
            safe_to_auto_route=False,
        )

    if is_non_referral(referral.text):
        return Decision(
            tier=Tier.EXCEPTION,
            recommended_specialty=None,
            recommended_priority=Priority.UNKNOWN,
            reason="Input does not appear to be a clinical referral.",
            rules_fired=["non_referral"],
            safe_to_auto_route=False,
        )

    # ------------------------------------------------------------------
    # 2. Extraction quality gate
    # ------------------------------------------------------------------
    if signals.extraction_confidence < min_extraction:
        return Decision(
            tier=Tier.HUMAN_REVIEW,
            recommended_specialty=proposal.specialty,
            recommended_priority=proposal.priority,
            reason=f"Extraction confidence {signals.extraction_confidence:.2f} below threshold {min_extraction}.",
            rules_fired=["low_extraction_confidence"],
            safe_to_auto_route=False,
        )

    # ------------------------------------------------------------------
    # 3. Unknown outputs
    # ------------------------------------------------------------------
    if proposal.specialty == UNKNOWN_SPECIALTY:
        return Decision(
            tier=Tier.HUMAN_REVIEW,
            recommended_specialty=None,
            recommended_priority=proposal.priority,
            reason="Specialty could not be determined from the referral.",
            rules_fired=["unknown_specialty"],
            safe_to_auto_route=False,
        )

    if proposal.priority == Priority.UNKNOWN:
        return Decision(
            tier=Tier.HUMAN_REVIEW,
            recommended_specialty=proposal.specialty,
            recommended_priority=Priority.UNKNOWN,
            reason="Priority could not be determined from the referral.",
            rules_fired=["unknown_priority"],
            safe_to_auto_route=False,
        )

    # ------------------------------------------------------------------
    # 4. Confidence gates
    # ------------------------------------------------------------------
    if proposal.specialty_confidence < min_specialty:
        return Decision(
            tier=Tier.HUMAN_REVIEW,
            recommended_specialty=proposal.specialty,
            recommended_priority=proposal.priority,
            reason=f"Specialty confidence {proposal.specialty_confidence:.2f} below threshold {min_specialty}.",
            rules_fired=["low_specialty_confidence"],
            safe_to_auto_route=False,
        )

    if proposal.priority_confidence < min_priority:
        return Decision(
            tier=Tier.HUMAN_REVIEW,
            recommended_specialty=proposal.specialty,
            recommended_priority=proposal.priority,
            reason=f"Priority confidence {proposal.priority_confidence:.2f} below threshold {min_priority}.",
            rules_fired=["low_priority_confidence"],
            safe_to_auto_route=False,
        )

    # ------------------------------------------------------------------
    # 5. Red flag safety floor — model cannot downgrade past red flags
    # ------------------------------------------------------------------
    flag_values = [f.value for f in signals.red_flags]
    if flag_values:
        floor = red_flag_priority_floor(flag_values)
        final_priority = priority_max(proposal.priority, floor)
        rules_fired = [f"red_flag_detected:{v}" for v in flag_values]
        if proposal.priority == Priority.ROUTINE:
            rules_fired.append("red_flag_escalated_from_routine")
        return Decision(
            tier=Tier.HUMAN_REVIEW,
            recommended_specialty=proposal.specialty,
            recommended_priority=final_priority,
            reason=f"Red flags detected: {', '.join(flag_values)}. Requires human review.",
            rules_fired=rules_fired,
            safe_to_auto_route=False,
        )

    # ------------------------------------------------------------------
    # 6. MVP scope — only Routine may auto-route in this release
    # ------------------------------------------------------------------
    if proposal.priority == Priority.URGENT:
        return Decision(
            tier=Tier.HUMAN_REVIEW,
            recommended_specialty=proposal.specialty,
            recommended_priority=Priority.URGENT,
            reason="Urgent cases require human review in MVP scope.",
            rules_fired=["mvp_urgent_requires_review"],
            safe_to_auto_route=False,
        )

    if proposal.priority == Priority.TWO_WEEK_WAIT:
        return Decision(
            tier=Tier.HUMAN_REVIEW,
            recommended_specialty=proposal.specialty,
            recommended_priority=Priority.TWO_WEEK_WAIT,
            reason="Two-Week Wait cases require human review in MVP scope.",
            rules_fired=["mvp_two_week_wait_requires_review"],
            safe_to_auto_route=False,
        )

    # ------------------------------------------------------------------
    # 7. AUTO_ROUTE — clean, Routine, above all thresholds, no red flags
    # ------------------------------------------------------------------
    return Decision(
        tier=Tier.AUTO_ROUTE,
        recommended_specialty=proposal.specialty,
        recommended_priority=Priority.ROUTINE,
        reason="Clean routine referral above all confidence thresholds with no red flags.",
        rules_fired=[],
        safe_to_auto_route=True,
    )
