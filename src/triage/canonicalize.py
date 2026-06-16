"""Controlled-vocabulary canonicalisation between extraction and classification.

This layer exists because of a coupling bug surfaced by running the real LLM
extractor (see README, "A real LLM backend, the coupling it exposed, and the
fix"). The mock extractor emits tidy tokens ("skin tag"); a real LLM emits richer
phrasings ("benign skin tag on the right forearm"). The classifier did
exact-string matching, so real-LLM output fell through to Unknown.

A typed schema (ClinicalSignals) guarantees *shape*. It does not guarantee
*value semantics* — that both producers use the same vocabulary. This module is
that missing contract: it maps whatever free text an extractor produces onto a
fixed controlled vocabulary the classifier can rely on.

It is deliberately a thin, transparent substring map, not an ontology service.
In production this is where SNOMED CT / a clinical terminology service would
sit; the seam is here so that swap is local.

Two vocabularies, not one: symptoms map against the symptom vocab (open routing
set), red flags against a separate red-flag vocab whose canonical names are
exactly the ones rules.py owns. Keeping them apart stops a red flag being
collapsed onto a symptom token the safety rules don't recognise — which would
silently drop the priority floor. See _RED_FLAG_VOCAB.
"""

from __future__ import annotations

from triage.schemas import ClinicalSignals, ExtractedFact

# Canonical symptom vocabulary. Key = canonical term the classifier keys off;
# value = substrings that map to it. First match wins; order does not matter
# because each canonical term owns disjoint substrings.
_SYMPTOM_VOCAB: dict[str, tuple[str, ...]] = {
    "skin tag": ("skin tag",),
    "rash": ("rash",),
    "mole": ("mole", "naevus", "pigmented lesion"),
    "eczema": ("eczema", "dermatitis"),
    "rectal bleeding": ("rectal bleeding", "pr bleeding", "blood in stool",
                        "blood per rectum", "bleeding from the back passage"),
    "weight loss": ("weight loss", "lost weight", "losing weight"),
    "chest pain": ("chest pain", "chest tightness", "anginal pain"),
    "shortness of breath": ("shortness of breath", "breathless", "sob",
                            "dyspnoea", "dyspnea"),
    "knee pain": ("knee pain", "knee joint pain"),
    "joint pain": ("joint pain", "arthralgia"),
    "abdominal pain": ("abdominal pain", "tummy pain", "stomach pain"),
    "difficulty swallowing": ("difficulty swallowing", "dysphagia"),
    "nausea": ("nausea", "feeling sick"),
    "bloating": ("bloating", "abdominal distension"),
    "loose stools": ("loose stools", "diarrhoea", "diarrhea"),
    "change in bowel habit": ("change in bowel habit", "bowel habit change",
                              "altered bowel habit"),
    "epigastric pain": ("epigastric pain", "upper abdominal pain"),
}

# Canonical RED-FLAG vocabulary. Separate from the symptom vocab on purpose:
# its canonical keys are exactly the flag names rules.py owns, so a red flag is
# never collapsed onto a non-flag token. (The symptom vocab maps "changing mole"
# -> "mole" for routing; "mole" is not a 2WW flag name, so routing red flags
# through it silently dropped the priority floor from TWO_WEEK_WAIT to ROUTINE.
# Red flags get their own value-set whose names the safety rules recognise.)
# Full-phrase variants only — never a bare "mole"/"bleeding" that could shadow
# a neighbouring flag.
_RED_FLAG_VOCAB: dict[str, tuple[str, ...]] = {
    "changing mole": ("changing mole", "evolving mole", "mole that has changed",
                      "changing naevus"),
    "bleeding mole": ("bleeding mole",),
    "irregular borders": ("irregular borders", "irregular border", "irregular edges"),
    "melanoma": ("melanoma",),
    "suspected cancer": ("suspected cancer", "suspected malignancy", "?cancer"),
    "rectal bleeding": ("rectal bleeding", "pr bleeding", "blood in stool",
                        "blood per rectum", "bleeding from the back passage"),
    "weight loss": ("weight loss", "lost weight", "losing weight",
                    "unintentional weight loss"),
    "chest pain": ("chest pain", "chest tightness", "anginal pain"),
    "shortness of breath": ("shortness of breath", "breathlessness", "breathless",
                            "dyspnoea", "dyspnea"),
    "haemoptysis": ("haemoptysis", "hemoptysis", "coughing up blood",
                    "blood in sputum"),
}

# Canonical specialty vocabulary. Same structure; maps near-synonyms used by
# different extractors to one routing label.
#
# NOTE: order matters and variants must be specific. "gi" as a bare substring
# is unsafe (it matches "surgical", "logic", "lower gi"...), so Colorectal's
# "lower gi" is listed and Gastroenterology uses word-bounded forms only. The
# matcher checks each canonical term's variants; the first canonical term with
# any matching variant wins, so more-specific terms are placed first.
_SPECIALTY_VOCAB: dict[str, tuple[str, ...]] = {
    "Colorectal": ("colorectal", "lower gi", "bowel"),
    "Dermatology": ("dermatology", "skin", "derm"),
    "Cardiology": ("cardiology", "cardiac", "heart"),
    "Respiratory": ("respiratory", "chest medicine", "pulmonary"),
    "Gastroenterology": ("gastroenterology", "gastro", "upper gi"),
    "Orthopaedics": ("orthopaedics", "orthopedics", "ortho", "msk"),
}


def _canonical_symptom(value: str) -> str | None:
    """Map a free-text symptom string to its canonical term, or None if unknown."""
    v = value.lower()
    for canonical, variants in _SYMPTOM_VOCAB.items():
        if any(variant in v for variant in variants):
            return canonical
    return None


def _canonical_red_flag(value: str) -> str | None:
    """Map a free-text red-flag string to a canonical flag name rules.py owns, or None.

    Uses the red-flag vocabulary, never the symptom vocabulary, so a flag is
    never remapped to a token the safety rules don't recognise as a red flag.
    """
    v = value.lower()
    for canonical, variants in _RED_FLAG_VOCAB.items():
        if any(variant in v for variant in variants):
            return canonical
    return None


def canonicalize_specialty(value: str) -> str | None:
    """Map a free-text specialty string to a canonical specialty, or None."""
    v = value.lower()
    for canonical, variants in _SPECIALTY_VOCAB.items():
        if any(variant in v for variant in variants):
            return canonical
    return None


def canonicalize_signals(signals: ClinicalSignals) -> ClinicalSignals:
    """Return a copy of signals with symptom and red-flag values canonicalised.

    Symptoms map against the symptom vocabulary; red flags map against the
    red-flag vocabulary (their canonical names are the ones rules.py owns).
    Unmappable terms are kept as-is rather than dropped — losing a signal is
    less safe than carrying an unrecognised one, which simply won't match a
    routing rule and will fall to human review. Evidence_text is preserved
    verbatim so the audit trail still shows what the extractor actually said.
    """
    def _remap(fact: ExtractedFact, to_canonical) -> ExtractedFact:
        canon = to_canonical(fact.value)
        if canon is None:
            return fact
        return ExtractedFact(
            value=canon,
            evidence_text=fact.evidence_text,  # keep original wording for audit
            confidence=fact.confidence,
        )

    return signals.model_copy(update={
        "symptoms": [_remap(s, _canonical_symptom) for s in signals.symptoms],
        "red_flags": [_remap(f, _canonical_red_flag) for f in signals.red_flags],
    })
