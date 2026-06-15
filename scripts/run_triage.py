"""Demo script — runs all synthetic letters through the full pipeline.

Usage:
    python scripts/run_triage.py

Writes audit_log.jsonl and prints a summary table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from triage.pipeline import process_referral
from triage.policy import load_thresholds
from triage.schemas import ReferralInput


def main() -> None:
    data_path = Path("data/synthetic_letters.json")
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run from the project root.")
        sys.exit(1)

    letters = json.loads(data_path.read_text(encoding="utf-8"))
    thresholds = load_thresholds()

    col = "{:<12} {:<20} {:<16} {:<14} {}"
    header = col.format("ID", "Specialty", "Priority", "Tier", "Reason")
    print(header)
    print("-" * 100)

    counts: dict[str, int] = {"AUTO_ROUTE": 0, "HUMAN_REVIEW": 0, "EXCEPTION": 0}

    for raw in letters:
        referral = ReferralInput(**{
            k: v for k, v in raw.items()
            if k in ReferralInput.model_fields
        })
        _, _, decision, _ = process_referral(referral, thresholds)

        specialty = decision.recommended_specialty or "—"
        priority = decision.recommended_priority.value if decision.recommended_priority else "—"
        reason = decision.reason[:55] + "…" if len(decision.reason) > 55 else decision.reason

        print(col.format(
            referral.referral_id,
            specialty,
            priority,
            decision.tier.value,
            reason,
        ))
        counts[decision.tier.value] = counts.get(decision.tier.value, 0) + 1

    total = len(letters)
    print()
    print(f"Total: {total}  |  "
          f"AUTO_ROUTE: {counts['AUTO_ROUTE']}  |  "
          f"HUMAN_REVIEW: {counts['HUMAN_REVIEW']}  |  "
          f"EXCEPTION: {counts['EXCEPTION']}")
    print("Audit log → audit_log.jsonl")


if __name__ == "__main__":
    main()
