"""Policy gate for the referral-triage POC.

The model proposes; this layer governs. Takes extraction signals and a
classification proposal and returns an auditable Decision.

Evaluation order — every step can only send a case to HUMAN_REVIEW or EXCEPTION,
never downgrade from a higher-severity outcome already reached. The safety floor
runs BEFORE the confidence gates on purpose: the floor asks "is this dangerous
regardless of how sure the model is?" and must not be skipped just because a
confidence gate would also have fired. Otherwise a red flag with sub-threshold
confidence reports only "low confidence" and the flag is invisible in
rules_fired — hidden magic the ordering exists to prevent.

  1. Text pre-filter     (EXCEPTION) — runs before any signal/confidence check.
                          Defense-in-depth: even if upstream returned high-confidence
                          signals for junk input, the text gate catches it.
  2. Red flag floor      (HUMAN_REVIEW) — any red flag present; recommended_priority
                          is escalated to max(proposal, red_flag_floor). A Routine
                          proposal with red flags is a contradiction the model made;
                          the rule fires and the case goes to human review. Runs
                          before the confidence gates so the flag is always audited.
  3. GP-stated floor     (HUMAN_REVIEW) — the GP's stated priority is a lower bound
                          the model may never drop below. If the GP marked it more
                          urgent than the model proposed, escalate and hand to a human.
                          A separate source from red flags, so a separate audit reason.
  4. Extraction quality  (HUMAN_REVIEW) — low extraction confidence.
  5. Unknown outputs     (HUMAN_REVIEW) — specialty or priority could not be determined.
  6. Confidence gates    (HUMAN_REVIEW) — specialty or priority confidence below threshold.
  7. MVP scope           (HUMAN_REVIEW) — Urgent and Two-Week Wait are not auto-routed
                          in this release; they always go to human review.
  8. AUTO_ROUTE          — clean, Routine, above all thresholds, no red flags.

Threshold semantics: comparison is strict less-than (<), so a confidence score
exactly equal to the threshold passes. `0.80 >= 0.80` passes the gate.

Every non-AUTO_ROUTE decision carries a non-empty reason. Every rule that
changed routing appears in rules_fired.
"""

from __future__ import annotations

import yaml

from triage.rules import is_non_referral, priority_max, red_flag_priority_floor
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
    # 2. Red flag safety floor — model cannot downgrade past red flags.
    #    Runs before the confidence gates so a red flag is always escalated
    #    and always visible in rules_fired, never masked by a confidence gate
    #    that would also have fired (that masking would be hidden magic).
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
    # 3. GP-stated priority floor — never auto-downgrade below the GP.
    #    A distinct floor from red flags: the GP's stated urgency, not the
    #    letter's clinical content. Given its own rule name and reason so the
    #    audit story is honest ("GP said urgent" != "red flag detected").
    #    Priority.UNKNOWN ranks 0, so a missing gp_stated_priority never fires.
    #    It triggers only when the model actually proposed a priority AND the GP
    #    outranks it — if the model abstained (UNKNOWN), that is the unknown gate's
    #    job, not a downgrade to guard against.
    # ------------------------------------------------------------------
    gp_floor = referral.gp_stated_priority or Priority.UNKNOWN
    if proposal.priority != Priority.UNKNOWN and gp_floor.rank > proposal.priority.rank:
        return Decision(
            tier=Tier.HUMAN_REVIEW,
            recommended_specialty=proposal.specialty,
            recommended_priority=priority_max(proposal.priority, gp_floor),
            reason=(
                f"GP stated priority {gp_floor.value} exceeds model proposal "
                f"{proposal.priority.value}; cannot auto-downgrade."
            ),
            rules_fired=["gp_stated_priority_floor"],
            safe_to_auto_route=False,
        )

    # ------------------------------------------------------------------
    # 4. Extraction quality gate
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
    # 5. Unknown outputs
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
    # 6. Confidence gates
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
    # 7. MVP scope — only Routine may auto-route in this release
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
    # 8. AUTO_ROUTE — clean, Routine, above all thresholds, no red flags
    # ------------------------------------------------------------------
    return Decision(
        tier=Tier.AUTO_ROUTE,
        recommended_specialty=proposal.specialty,
        recommended_priority=Priority.ROUTINE,
        reason="Clean routine referral above all confidence thresholds with no red flags.",
        rules_fired=[],
        safe_to_auto_route=True,
    )
