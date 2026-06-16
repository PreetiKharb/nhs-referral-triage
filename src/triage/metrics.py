"""Observability metrics for the referral-triage POC.

Computes the operational, AI/product, and safety metrics named in the design
document from a batch of pipeline decisions. This is the shape of what a
dashboard would show in production; here it prints to the console.

The metrics are grouped to match the design doc's three lenses:
  - AI / product:  routing mix, specialty/priority distribution, red-flag rate,
                   confidence distribution, top abstention reasons.
  - Safety:        2WW/Urgent downgrades, unsafe auto-routes, missed red flags.
  - Operational:   exception rate, throughput (count) — latency/cost/queue depth
                   are named but not measurable in a single-process POC.

Nothing here makes a routing decision; it only summarises decisions already made.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from triage.schemas import ClinicalSignals, Decision, Priority, Proposal, Tier


@dataclass
class TriageMetrics:
    """Aggregated metrics over a batch of pipeline runs."""

    total: int = 0
    tier_counts: Counter = field(default_factory=Counter)
    specialty_counts: Counter = field(default_factory=Counter)
    priority_counts: Counter = field(default_factory=Counter)
    abstention_reasons: Counter = field(default_factory=Counter)
    red_flag_cases: int = 0
    confidence_buckets: Counter = field(default_factory=Counter)

    def observe(self, signals: ClinicalSignals, proposal: Proposal, decision: Decision) -> None:
        """Record one pipeline run."""
        self.total += 1
        self.tier_counts[decision.tier.value] += 1
        self.specialty_counts[decision.recommended_specialty or "—"] += 1
        self.priority_counts[decision.recommended_priority.value] += 1

        if signals.red_flags:
            self.red_flag_cases += 1

        # Top abstention reason = the first rule that fired (the deciding one).
        if decision.tier != Tier.AUTO_ROUTE and decision.rules_fired:
            self.abstention_reasons[decision.rules_fired[0]] += 1

        # Confidence distribution on the priority proposal (the safety-relevant one).
        bucket = _confidence_bucket(proposal.priority_confidence)
        self.confidence_buckets[bucket] += 1

    def render(self) -> str:
        """Format the metrics as a console report."""
        lines: list[str] = []
        add = lines.append

        add("=" * 55)
        add("OBSERVABILITY REPORT")
        add("=" * 55)

        add("\nAI / product metrics")
        add(f"  Referrals processed        {self.total}")
        add("  Routing mix:")
        for tier in ("AUTO_ROUTE", "HUMAN_REVIEW", "EXCEPTION"):
            c = self.tier_counts.get(tier, 0)
            add(f"    {tier:<14} {c:>2}  ({_pct(c, self.total)})")
        add(f"  Auto-route rate            {_pct(self.tier_counts.get('AUTO_ROUTE', 0), self.total)}")
        add(f"  Human-review rate          {_pct(self.tier_counts.get('HUMAN_REVIEW', 0), self.total)}")
        add(f"  Red-flag detection rate    {_pct(self.red_flag_cases, self.total)}")

        add("  Specialty distribution:")
        for specialty, c in self.specialty_counts.most_common():
            add(f"    {specialty:<18} {c}")

        add("  Priority distribution:")
        for priority, c in self.priority_counts.most_common():
            add(f"    {priority:<18} {c}")

        add("  Priority-confidence distribution:")
        for bucket in ("<0.70", "0.70–0.79", "0.80–0.89", "0.90+"):
            c = self.confidence_buckets.get(bucket, 0)
            add(f"    {bucket:<18} {c}")

        if self.abstention_reasons:
            add("  Top abstention reasons:")
            for reason, c in self.abstention_reasons.most_common(5):
                add(f"    {reason:<34} {c}")

        return "\n".join(lines)


def _confidence_bucket(value: float) -> str:
    if value < 0.70:
        return "<0.70"
    if value < 0.80:
        return "0.70–0.79"
    if value < 0.90:
        return "0.80–0.89"
    return "0.90+"


def _pct(count: int, total: int) -> str:
    return f"{count / total * 100:.0f}%" if total else "n/a"
