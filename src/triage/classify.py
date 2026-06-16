"""Mock specialty and priority classifier for the referral-triage POC.

Deterministic rule-based logic that simulates what a real classifier would
return: separate specialty and priority confidence scores, alternatives, and
evidence drawn from the extracted signals.

NOT clinical logic — these mappings are illustrative placeholders for the POC.
A real classifier would be a fine-tuned model or an LLM with a structured
output schema; this mock sits behind the same Proposal interface, so swapping
it changes nothing downstream.

Design note: specialty_confidence and priority_confidence are kept separate
because you can know the right team (high specialty confidence) while still
being unsure about urgency (low priority confidence), and vice versa. Conflating
them into one score would obscure which dimension drove a HUMAN_REVIEW decision.

Production path (deliberately not built — the brief said model choice is out of
scope, and a rule-based mock keeps the POC's behaviour fully inspectable):
  - Replace this rule table with a classifier behind the SAME Proposal contract,
    so nothing downstream changes — the same swap already demonstrated for
    extraction via extract.py's --real backend.
  - MVP-grade option: an LLM (e.g. Claude via the structured-outputs API, the
    same messages.parse + Pydantic-schema approach used in extract.py) prompted
    to return specialty, priority, alternatives and per-axis confidence. Cheap
    to stand up, no training data needed, good for cold-start.
  - Mature option: a supervised classifier (e.g. a fine-tuned transformer or a
    gradient-boosted model over extracted features) trained on the labelled
    golden dataset, once shadow mode has produced enough SME-reviewed examples.
    Preferred long-term because it is cheaper per call, calibratable, and its
    confidence scores can be validated against held-out accuracy.
  - Either way the safety floor (rules.py) stays deterministic and OUTSIDE the
    model: the classifier proposes, it never gets the final say on priority.
"""

from __future__ import annotations

from triage.schemas import UNKNOWN_SPECIALTY, ClinicalSignals, Priority, Proposal

# ---------------------------------------------------------------------------
# Signal-to-routing maps (illustrative, not clinically validated)
#
# Two layers of assumption live in this file, both unvalidated:
#   1. The signal->specialty/priority mappings (e.g. chest pain -> Cardiology/
#      Urgent). These are engineering placeholders; in production they are
#      either SME-authored config or a model trained on labelled referrals.
#   2. The confidence numbers below (0.92, 0.88, 0.30 ...) are hand-picked
#      constants, NOT calibrated probabilities. They exist so the confidence
#      gate has something to gate on; their specific values are not meaningful
#      until a real model produces calibrated scores against a labelled set.
# ---------------------------------------------------------------------------

_TWO_WEEK_WAIT_FLAGS = {"melanoma", "suspected cancer", "changing mole",
                        "bleeding mole", "irregular borders"}

_CARDIAC_FLAGS = {"chest pain"}
_RESPIRATORY_FLAGS = {"shortness of breath", "haemoptysis", "breathlessness"}
_GI_FLAGS = {"rectal bleeding", "weight loss"}

_DERMATOLOGY_SYMPTOMS = {"skin tag", "rash", "mole", "eczema"}
_ORTHOPAEDIC_SYMPTOMS = {"knee pain", "joint pain"}
_GI_SYMPTOMS = {"abdominal pain", "loose stools", "nausea", "bloating",
                "change in bowel habit", "epigastric pain", "difficulty swallowing"}
_CARDIAC_SYMPTOMS = {"angina"}


def classify(signals: ClinicalSignals) -> Proposal:
    """Propose specialty and priority from extracted clinical signals.

    Returns UNKNOWN_SPECIALTY / Priority.UNKNOWN with low confidence when
    signals are insufficient or contradictory — the policy gate handles these.
    """
    flag_set = {f.value.lower() for f in signals.red_flags}
    symptom_set = {s.value.lower() for s in signals.symptoms}
    evidence = [f.value for f in signals.red_flags] + [s.value for s in signals.symptoms]

    # --- Two-Week Wait pathway (suspected cancer) ---
    tww_flags = flag_set & _TWO_WEEK_WAIT_FLAGS
    if tww_flags:
        # Skin/lesion 2WW → Dermatology
        if tww_flags & {"melanoma", "suspected melanoma", "changing mole",
                        "bleeding mole", "irregular borders"}:
            return Proposal(
                specialty="Dermatology",
                specialty_confidence=0.92,
                priority=Priority.TWO_WEEK_WAIT,
                priority_confidence=0.95,
                alternatives=[],
                evidence=evidence,
            )

    # --- Rectal bleeding + weight loss → Colorectal 2WW ---
    if "rectal bleeding" in flag_set and "weight loss" in flag_set:
        return Proposal(
            specialty="Colorectal",
            specialty_confidence=0.85,
            priority=Priority.TWO_WEEK_WAIT,
            priority_confidence=0.88,
            alternatives=["Gastroenterology"],
            evidence=evidence,
        )

    # --- Cardiac urgent ---
    if flag_set & _CARDIAC_FLAGS or symptom_set & _CARDIAC_SYMPTOMS:
        return Proposal(
            specialty="Cardiology",
            specialty_confidence=0.88,
            priority=Priority.URGENT,
            priority_confidence=0.85,
            alternatives=[],
            evidence=evidence,
        )

    # --- Respiratory urgent ---
    if flag_set & _RESPIRATORY_FLAGS:
        return Proposal(
            specialty="Respiratory",
            specialty_confidence=0.87,
            priority=Priority.URGENT,
            priority_confidence=0.85,
            alternatives=[],
            evidence=evidence,
        )

    # --- GI urgent (single red flag without combination) ---
    if flag_set & _GI_FLAGS:
        return Proposal(
            specialty="Gastroenterology",
            specialty_confidence=0.80,
            priority=Priority.URGENT,
            priority_confidence=0.78,
            alternatives=["Colorectal"],
            evidence=evidence,
        )

    # --- Routine by symptom ---
    if symptom_set & _DERMATOLOGY_SYMPTOMS:
        return Proposal(
            specialty="Dermatology",
            specialty_confidence=0.88,
            priority=Priority.ROUTINE,
            priority_confidence=0.90,
            alternatives=[],
            evidence=evidence,
        )

    if symptom_set & _ORTHOPAEDIC_SYMPTOMS:
        return Proposal(
            specialty="Orthopaedics",
            specialty_confidence=0.87,
            priority=Priority.ROUTINE,
            priority_confidence=0.89,
            alternatives=[],
            evidence=evidence,
        )

    if symptom_set & _GI_SYMPTOMS:
        return Proposal(
            specialty="Gastroenterology",
            specialty_confidence=0.80,
            priority=Priority.ROUTINE,
            priority_confidence=0.82,
            alternatives=["Colorectal"],
            evidence=evidence,
        )

    # --- Fallback: cannot classify ---
    return Proposal(
        specialty=UNKNOWN_SPECIALTY,
        specialty_confidence=0.30,
        priority=Priority.UNKNOWN,
        priority_confidence=0.30,
        alternatives=[],
        evidence=evidence,
    )
