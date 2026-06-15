"""Audit logging for the referral-triage POC.

Append-only JSONL: one line per referral, containing the full decision
journey (signals -> proposal -> decision) plus version stamps for replay.

Raw letter text is never written to the log — only referral_id is stored.
Version fields are passed in explicitly by the caller; no defaults here.
"""

from __future__ import annotations

from datetime import datetime, timezone

from triage.schemas import AuditRecord, ClinicalSignals, Decision, Proposal, ReferralInput


def create_audit_record(
    referral: ReferralInput,
    signals: ClinicalSignals,
    proposal: Proposal,
    decision: Decision,
    *,
    pipeline_version: str,
    ruleset_version: str,
    threshold_version: str,
) -> AuditRecord:
    """Build an AuditRecord for one referral pipeline run."""
    return AuditRecord(
        referral_id=referral.referral_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        synthetic_input_reference=referral.referral_id,
        signals=signals,
        proposal=proposal,
        decision=decision,
        pipeline_version=pipeline_version,
        ruleset_version=ruleset_version,
        threshold_version=threshold_version,
    )


def append_audit_record(record: AuditRecord, path: str = "audit_log.jsonl") -> None:
    """Append a single audit record to the JSONL log. Never overwrites."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
