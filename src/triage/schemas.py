"""Data contracts for the referral-triage pipeline.

One model per pipeline stage:

    ReferralInput   -> raw synthetic letter entering the pipeline.
    ClinicalSignals -> extraction stage: structured clinical facts.
    Proposal        -> classification stage: model-suggested specialty + priority.
    Decision        -> safety/policy stage: final auditable routing outcome.
    AuditRecord     -> one letter's full journey, persisted as a JSONL line.

Priority carries an explicit numeric rank so every comparison uses ordinal
severity, never enum names or strings. All priority ordering must use .rank.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Priority(str, Enum):
    """Clinical priority bands. Severity order: UNKNOWN < ROUTINE < URGENT < TWO_WEEK_WAIT."""

    UNKNOWN = "UNKNOWN"
    ROUTINE = "ROUTINE"
    URGENT = "URGENT"
    TWO_WEEK_WAIT = "TWO_WEEK_WAIT"

    @property
    def rank(self) -> int:
        """Numeric severity ordinal. Use this for all priority comparisons."""
        return {
            "UNKNOWN": 0,
            "ROUTINE": 1,
            "URGENT": 2,
            "TWO_WEEK_WAIT": 3,
        }[self.value]


class Tier(str, Enum):
    """Routing tier: how much autonomy the system takes over a letter."""

    AUTO_ROUTE = "AUTO_ROUTE"       # Tier 1: confident, routine, clean — no human touch
    HUMAN_REVIEW = "HUMAN_REVIEW"   # Tier 2: uncertain, escalated, or ambiguous
    EXCEPTION = "EXCEPTION"         # Not a usable referral — fail closed to a human


class ExtractedFact(BaseModel):
    """A single clinical fact with its supporting quote. No fact without evidence."""

    value: str
    evidence_text: str
    confidence: float = Field(ge=0.0, le=1.0)


class ReferralInput(BaseModel):
    """Raw letter entering the pipeline. expected_* hold gold labels for synthetic eval."""

    referral_id: str
    text: str
    source: str
    gp_stated_priority: Priority | None = None
    expected_specialty: str | None = None
    expected_priority: Priority | None = None
    expected_decision: Tier | None = None


class ClinicalSignals(BaseModel):
    """Extraction-stage output: structured signals lifted from the GP letter."""

    symptoms: list[ExtractedFact]
    red_flags: list[ExtractedFact]
    suspected_condition: ExtractedFact | None = None
    gp_stated_urgency: ExtractedFact | None = None
    extraction_confidence: float = Field(ge=0.0, le=1.0)


class Proposal(BaseModel):
    """Classification-stage output: model-suggested specialty and priority with rationale."""

    specialty: str
    specialty_confidence: float = Field(ge=0.0, le=1.0)
    priority: Priority
    priority_confidence: float = Field(ge=0.0, le=1.0)
    alternatives: list[str]
    evidence: list[str]
    model_version: str = "mock-classifier-v0.1"


class Decision(BaseModel):
    """Policy-stage output: the final, auditable routing outcome.

    The model proposes; this layer governs. Every rule that affected
    routing must appear in rules_fired — no hidden magic.
    """

    tier: Tier
    recommended_specialty: str | None
    recommended_priority: Priority
    reason: str
    rules_fired: list[str]
    safe_to_auto_route: bool


class AuditRecord(BaseModel):
    """One letter's full journey (signals -> proposal -> decision), persisted as JSONL.

    Version fields have no defaults — audit.py must pass the versions that were
    actually loaded at runtime. A default here would let a stale stamp silently
    disagree with the config that produced the decision, breaking replay.
    """

    referral_id: str
    timestamp: str
    synthetic_input_reference: str
    signals: ClinicalSignals
    proposal: Proposal
    decision: Decision
    pipeline_version: str
    ruleset_version: str
    threshold_version: str
