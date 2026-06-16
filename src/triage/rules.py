"""Deterministic safety rules for the referral-triage POC.

Five functions; no mutable state; no ML:

    is_too_short(text)              -> bool
    is_non_referral(text)           -> bool
    detect_red_flags(text)          -> list[str]  canonical flag names
    priority_max(a, b)              -> Priority   uses .rank, never strings
    red_flag_priority_floor(flags)  -> Priority   minimum priority implied by flags

These run before and after the model. The pre-filtering gate (is_too_short,
is_non_referral) runs in policy.py before extraction. detect_red_flags and
the floor/max functions run after classification to enforce the safety floor.

All priority ordering uses Priority.rank — the ordinal defined in schemas.py.
NEVER compare priority by name or value string.
"""

from __future__ import annotations

import re

from triage.schemas import Priority

# ---------------------------------------------------------------------------
# Configuration — clinical thresholds and keyword lists
#
# CLINICAL ASSUMPTION — NOT VALIDATED. Every mapping below (which red flags
# exist, and whether each implies Urgent or Two-Week Wait) is an engineering
# placeholder authored WITHOUT clinical sign-off. They are illustrative
# simplifications of NICE-style criteria, not a clinical ruleset.
#
# Known gaps a clinician would flag immediately, for example:
#   - isolated rectal bleeding in an older patient is itself a NICE lower-GI
#     2WW criterion, but here it only reaches Urgent unless paired with weight
#     loss. That is a recall gap in these rules, not in the architecture.
#
# In production these tables are NOT owned by engineering. They belong to a
# clinical safety officer, are encoded as versioned, reviewed config (not code),
# and are validated against current NICE guidance and a labelled referral set
# with red-flag-recall as the gating metric. The rules live in this isolated,
# tested module precisely so that ownership handoff is a config-and-review
# process, not a code change.
# ---------------------------------------------------------------------------

# Referrals below this word count cannot carry enough clinical detail to route.
_MIN_WORD_COUNT = 15

# Tokens that mark an input as admin rather than clinical.
_ADMIN_KEYWORDS: frozenset[str] = frozenset({
    "rota", "payroll", "invoice",
    "holiday request", "holiday requests",
    "staff meeting", "annual leave", "sick leave",
})

# Negation words that cancel a red-flag match when found in the lookback window.
_NEGATION_WORDS: frozenset[str] = frozenset({
    "no", "not", "without", "nil",
    "denies", "denied", "negative", "absence", "absent",
})

# How many tokens before a phrase match to check for negation.
_NEGATION_WINDOW = 4

# (phrase to match as token sequence, canonical flag name returned to caller).
# More specific phrases listed before their substrings to avoid shadowing.
# NOTE: "without trying" maps to "weight loss" — GP phrasing for unintentional
# weight loss. "without" appears in _NEGATION_WORDS but is the START of this
# phrase, not a token before it, so the negation window never fires on itself.
_FLAG_PATTERNS: list[tuple[str, str]] = [
    ("suspected melanoma",      "melanoma"),
    ("suspected cancer",        "suspected cancer"),
    ("changing mole",           "changing mole"),
    ("bleeding mole",           "bleeding mole"),
    ("irregular borders",       "irregular borders"),  # morphological skin-lesion flag
    ("rectal bleeding",         "rectal bleeding"),
    ("shortness of breath",     "shortness of breath"),
    ("breathlessness",          "shortness of breath"),
    ("haemoptysis",             "haemoptysis"),
    ("chest pain",              "chest pain"),
    ("melanoma",                "melanoma"),
    ("weight loss",             "weight loss"),
    ("lost weight",             "weight loss"),
    ("without trying",          "weight loss"),        # unintentional weight loss
]

# Flags that individually imply the Two-Week Wait (suspected-cancer) pathway.
_TWO_WEEK_WAIT_FLAGS: frozenset[str] = frozenset({
    "changing mole", "bleeding mole", "irregular borders",
    "melanoma", "suspected cancer",
})

# Flags that individually imply Urgent (serious but not on the 2WW pathway alone).
_URGENT_FLAGS: frozenset[str] = frozenset({
    "chest pain", "shortness of breath", "haemoptysis",
    "rectal bleeding", "weight loss",
})

# Combinations that escalate to Two-Week Wait even if neither flag alone would.
# Reflects NICE lower-GI cancer guidance: rectal bleeding + weight loss together
# is a 2WW criterion even when each feature is individually non-specific.
_TWO_WEEK_WAIT_COMBINATIONS: list[frozenset[str]] = [
    frozenset({"rectal bleeding", "weight loss"}),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Lowercase alphabetic tokens; strips punctuation and digits."""
    return re.findall(r"[a-z']+", text.lower())


def _is_negated(tokens: list[str], phrase_start: int) -> bool:
    """True if a negation word appears in the window immediately before phrase_start."""
    window = tokens[max(0, phrase_start - _NEGATION_WINDOW): phrase_start]
    return bool(_NEGATION_WORDS & set(window))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_too_short(text: str) -> bool:
    """True if the text has too few words to carry a routable clinical referral."""
    return len(text.split()) < _MIN_WORD_COUNT


def is_non_referral(text: str) -> bool:
    """True if the text looks like admin content rather than a GP referral."""
    lower = text.lower()
    return any(kw in lower for kw in _ADMIN_KEYWORDS)


def detect_red_flags(text: str) -> list[str]:
    """Return sorted canonical red-flag names found in text, with negation suppression.

    Iterates token-by-token so negation is checked by position, not regex lookahead.
    REF-007 ("No frank rectal bleeding") and REF-008 ("no chest pain at rest")
    are the regression cases: both must return no flags despite containing the
    keyword phrases.

    Hedged weight loss ("may be related to reduced appetite") is NOT detected here
    by design — extract.py is responsible for that clinical distinction.
    """
    tokens = _tokenise(text)
    n = len(tokens)
    found: set[str] = set()

    for phrase, canonical in _FLAG_PATTERNS:
        if canonical in found:
            continue  # already have this flag; skip cheaper patterns for it
        phrase_tokens = phrase.split()
        phrase_len = len(phrase_tokens)

        for i in range(n - phrase_len + 1):
            if tokens[i: i + phrase_len] == phrase_tokens:
                if not _is_negated(tokens, i):
                    found.add(canonical)
                break  # stop scanning for this phrase after first match

    return sorted(found)


def priority_max(a: Priority, b: Priority) -> Priority:
    """Return the more clinically severe priority. Uses .rank — never string comparison."""
    return a if a.rank >= b.rank else b


def red_flag_priority_floor(red_flags: list[str]) -> Priority:
    """Minimum Priority implied by the detected flags.

    Individual flags map to TWO_WEEK_WAIT or URGENT. Combinations can escalate
    further: rectal bleeding + weight loss → TWO_WEEK_WAIT even if neither alone
    would reach it (NICE lower-GI cancer pathway).
    """
    if not red_flags:
        return Priority.ROUTINE

    flag_set = {f.lower() for f in red_flags}
    floor = Priority.ROUTINE

    for flag in flag_set:
        if flag in _TWO_WEEK_WAIT_FLAGS:
            floor = priority_max(floor, Priority.TWO_WEEK_WAIT)
        elif flag in _URGENT_FLAGS:
            floor = priority_max(floor, Priority.URGENT)

    for combination in _TWO_WEEK_WAIT_COMBINATIONS:
        if combination.issubset(flag_set):
            floor = priority_max(floor, Priority.TWO_WEEK_WAIT)

    return floor
