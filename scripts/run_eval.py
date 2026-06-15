"""Evaluation script for the referral-triage POC.

Runs the full pipeline over the synthetic dataset and compares predicted
outputs against gold labels. Reports accuracy, safety metrics, and routing
distribution.

Safety metrics are the primary signal:
  unsafe_auto_route_count     — AUTO_ROUTE when gold label is HUMAN_REVIEW or EXCEPTION.
  urgent_or_2ww_downgrade_count — predicted ROUTINE when gold label is URGENT or TWO_WEEK_WAIT.

These use synthetic labels against a deterministic mock, so numbers show the
shape of the evaluation, not clinical validity. A real eval requires a
labelled holdout set and a calibrated model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from triage.pipeline import process_referral
from triage.policy import load_thresholds
from triage.schemas import Priority, ReferralInput, Tier


def main() -> None:
    data_path = Path("data/synthetic_letters.json")
    letters = json.loads(data_path.read_text(encoding="utf-8"))
    thresholds = load_thresholds()

    total = len(letters)

    # Routing counts
    tier_counts: dict[str, int] = {"AUTO_ROUTE": 0, "HUMAN_REVIEW": 0, "EXCEPTION": 0}

    # Accuracy (only cases with gold labels and non-EXCEPTION outcomes)
    specialty_correct = specialty_total = 0
    priority_correct = priority_total = 0
    tier_correct = tier_total = 0

    # Safety metrics — the numbers that matter most
    unsafe_auto_route_count = 0
    urgent_or_2ww_downgrade_count = 0

    mismatches: list[str] = []

    for raw in letters:
        referral = ReferralInput(**{
            k: v for k, v in raw.items()
            if k in ReferralInput.model_fields
        })
        _, proposal, decision, _ = process_referral(referral, thresholds, audit_path="eval_audit.jsonl")

        tier_counts[decision.tier.value] = tier_counts.get(decision.tier.value, 0) + 1

        gold_tier = Tier(raw["expected_decision"]) if raw.get("expected_decision") else None
        gold_specialty = raw.get("expected_specialty")
        gold_priority = Priority(raw["expected_priority"]) if raw.get("expected_priority") else None

        # --- Tier accuracy ---
        if gold_tier:
            tier_total += 1
            if decision.tier == gold_tier:
                tier_correct += 1
            else:
                mismatches.append(
                    f"  {referral.referral_id}: tier {decision.tier.value} != expected {gold_tier.value}"
                )

            # Safety: AUTO_ROUTE when it shouldn't be
            if decision.tier == Tier.AUTO_ROUTE and gold_tier in (Tier.HUMAN_REVIEW, Tier.EXCEPTION):
                unsafe_auto_route_count += 1

        # --- Specialty accuracy (skip EXCEPTION cases — no usable routing attempted) ---
        if gold_specialty and decision.tier != Tier.EXCEPTION:
            specialty_total += 1
            if proposal.specialty == gold_specialty:
                specialty_correct += 1

        # --- Priority accuracy and downgrade check ---
        if gold_priority and decision.tier != Tier.EXCEPTION:
            priority_total += 1
            predicted_priority = decision.recommended_priority
            if predicted_priority == gold_priority:
                priority_correct += 1

            # Safety: predicted ROUTINE when gold is URGENT or TWO_WEEK_WAIT
            if (gold_priority.rank >= Priority.URGENT.rank
                    and predicted_priority == Priority.ROUTINE):
                urgent_or_2ww_downgrade_count += 1

    # --- Print report ---
    print("=" * 55)
    print("EVALUATION REPORT")
    print("=" * 55)

    print("\nRouting distribution:")
    for tier, count in tier_counts.items():
        pct = count / total * 100
        print(f"  {tier:<16} {count:>2} / {total}  ({pct:.0f}%)")

    print("\nAccuracy (synthetic labels only):")
    _pct = lambda c, t: f"{c}/{t}  ({c/t*100:.0f}%)" if t else "n/a"
    print(f"  Tier accuracy      {_pct(tier_correct, tier_total)}")
    print(f"  Specialty accuracy {_pct(specialty_correct, specialty_total)}")
    print(f"  Priority accuracy  {_pct(priority_correct, priority_total)}")

    print("\n⚠️  Safety metrics:")
    print(f"  Unsafe auto-route count          {unsafe_auto_route_count}")
    print(f"  Urgent / 2WW downgrade count     {urgent_or_2ww_downgrade_count}")

    if mismatches:
        print("\nTier mismatches:")
        for m in mismatches:
            print(m)

    print()
    if unsafe_auto_route_count == 0 and urgent_or_2ww_downgrade_count == 0:
        print("✓ No safety violations detected.")
    else:
        print("✗ Safety violations present — review mismatches above.")

    print("\nNote: numbers reflect the mock pipeline against synthetic labels.")
    print("Calibration and clinical validity require a real labelled dataset.")


if __name__ == "__main__":
    main()
