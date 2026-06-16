"""Demo script — runs all synthetic letters through the full pipeline.

Usage:
    python scripts/run_triage.py            # mock extraction (default, offline)
    python scripts/run_triage.py --real     # real Claude extraction (needs ANTHROPIC_API_KEY)

The --real flag swaps only the extraction backend. Everything downstream —
rules, policy gate, audit, routing — is byte-for-byte the same code path. That
is the modularity claim made demonstrable: the model is a dependency behind a
schema contract, not the architecture.

Writes audit_log.jsonl and prints a summary table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from triage.pipeline import process_referral
from triage.policy import load_thresholds
from triage.schemas import ReferralInput


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic referrals through the triage pipeline.")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use the real Claude extraction backend (falls back to mock if no API key).",
    )
    args = parser.parse_args()
    backend = "llm" if args.real else "mock"

    data_path = Path("data/synthetic_letters.json")
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run from the project root.")
        sys.exit(1)

    letters = json.loads(data_path.read_text(encoding="utf-8"))
    thresholds = load_thresholds()

    print(f"Extraction backend: {backend}")
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
        _, _, decision, _ = process_referral(referral, thresholds, backend=backend)

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
