"""End-to-end pipeline for the referral-triage POC.

Wires the four stages in order:
    extract_signals -> classify -> apply_policy -> create_audit_record

Version strings are read from the loaded thresholds config and from constants
in policy.py so the audit stamp always reflects what actually ran.
"""

from __future__ import annotations

from triage.audit import append_audit_record, create_audit_record
from triage.canonicalize import canonicalize_signals
from triage.classify import classify
from triage.extract import extract_signals
from triage.policy import PIPELINE_VERSION, RULESET_VERSION, apply_policy, load_thresholds
from triage.schemas import AuditRecord, ClinicalSignals, Decision, Proposal, ReferralInput


def process_referral(
    referral: ReferralInput,
    thresholds: dict | None = None,
    audit_path: str = "audit_log.jsonl",
    backend: str = "mock",
) -> tuple[ClinicalSignals, Proposal, Decision, AuditRecord]:
    """Run one referral through the full pipeline and write an audit record.

    backend selects the extraction implementation ("mock" or "llm"). Everything
    downstream of extraction is identical regardless of backend — that is the
    point of the schema contract.
    """
    if thresholds is None:
        thresholds = load_thresholds()

    signals = extract_signals(referral, backend=backend)
    signals = canonicalize_signals(signals)  # controlled-vocab contract: see canonicalize.py
    proposal = classify(signals)
    decision = apply_policy(referral, signals, proposal, thresholds)

    record = create_audit_record(
        referral, signals, proposal, decision,
        pipeline_version=PIPELINE_VERSION,
        ruleset_version=RULESET_VERSION,
        threshold_version=thresholds.get("version", "unknown"),
    )
    append_audit_record(record, path=audit_path)

    return signals, proposal, decision, record
