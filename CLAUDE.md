# CLAUDE.md — agent instructions for this repo

This is a **deliberately minimal proof-of-concept** for an NHS GP-referral triage system. It accompanies a system design document and exists to demonstrate the design's core decisions in working code — not to be a production system.

## Prime directive
I (the human) make the design decisions; you write the syntax. When a design choice is ambiguous, ASK — do not invent architecture. Every line you write, I must be able to defend out loud without notes. If a line can't be explained simply, simplify it.

## The model proposes; policy governs
Extraction and classification behave like a model and produce a *proposal* — never an action. The policy gate decides whether that proposal is safe to act on. Anything uncertain, high-risk, unsupported, or out of scope goes to human review or exception, never straight to auto-route.

## Scope discipline (do not over-engineer)
- This is a local CLI proof-of-concept. Do not introduce distributed infrastructure, services, databases, async workers, queues, or web APIs. A CLI process reading JSON is the intended scope.
- Keep the implementation small: a few hundred lines total across the repo. If a file grows past roughly 80 lines, stop and flag it.
- No frameworks beyond Pydantic and pyyaml. No FastAPI, no LangChain, no cloud SDKs.
- Prefer the boring, readable solution over the clever one.

## Non-negotiables (safety — never violate these)
Priority order is: **Routine < Urgent < Two-Week Wait.** When comparing priorities, always use this explicit ordering — never string comparison.

1. **Never auto-downgrade priority.** Rules may escalate priority or hold it; they may NEVER lower it below the GP-stated priority or below what deterministic red-flag rules imply. Compare priorities using the explicit ordering above.
2. **No fact without evidence.** Every extracted clinical signal carries `evidence_text`. An extraction with no supporting quote is discarded, never kept. No evidence, no fact. Evidence text must be copied from or clearly traceable to the synthetic referral text — do not invent evidence phrases.
3. **Red-flag rules are deterministic and override the model.** Red flags may be detected during extraction, but deterministic safety rules are enforced after classification and before routing. They can only escalate or hold a decision, never downgrade it. Only clean, routine, high-confidence cases with no red flag are eligible for auto-route; urgent / suspected-cancer signals go to human review even when the classifier looks confident.
4. **Fail closed.** Empty, malformed, or unparseable input routes to human review or exceptions — never a silent guess, never a default-to-routine.
5. **No hidden magic.** Anything that affects routing must be visible in the returned decision — it appears in `rules_fired` and in the `reason`. A rule that changes the outcome silently is a bug.
6. **Every decision is auditable.** Each decision logs the referral ID, synthetic input reference, extracted signals, proposal, rules fired, confidence, final decision + reason, and version stamps: pipeline version, mock model version, ruleset version, and threshold version. Append-only JSONL. Do not log real patient-identifiable information.

## Test-first for the safety floor
`rules.py` and `policy.py` get their tests written BEFORE their implementation. The tests define what "safe" means; the code is built to pass them.

## Evaluate behaviour, not just output
`run_eval.py` reports safety- and workflow-shaped metrics, not accuracy alone: specialty / priority / decision accuracy, auto-route rate, human-review rate, exception rate, unsafe auto-route count, and Urgent/2WW downgrade count. Labels here are synthetic — these show the *shape* of evaluation, not clinical validity.

## Conventions
- Python 3.11, type hints everywhere, `snake_case`.
- Pydantic v2 for all data contracts (see `schemas.py` — the source of truth for shapes).
- Fixed-value fields are enums (Priority, Tier); open/growing sets (specialty) are strings or config.
- Thresholds (min extraction / specialty / priority confidence) live in `config/thresholds.yaml`, never hardcoded — the automation boundary stays inspectable.
- Names explain the domain decision: `rules_fired`, `safe_to_auto_route`, `red_flags`, `evidence_text`, `priority_confidence`. Avoid `result`, `data`, `output`, `thing`.
- Priority enum values carry an explicit numeric rank: Routine = 1, Urgent = 2, Two-Week Wait = 3. All priority comparisons must use that rank, never enum names or strings.
- Catch specific exceptions, never bare `except:`, never swallow errors silently.
- The model lives behind a stable interface: a `--mock` deterministic implementation and a real-LLM one must be swappable without changing anything downstream. Make the mock's keyword logic explicit and comment that it is not clinical logic.

## What NOT to do
- Don't add features I didn't ask for, such as retries, caching, a web UI, authentication, databases, or deployment code.
- Do not expand the POC into production infrastructure. Production concerns should be documented as future extensions, not implemented here.
- Don't write code whose behaviour would be hard to explain simply.
- Don't invent clinical knowledge — red-flag examples are illustrative; flag where a real clinical decision or governance input would be needed.

## Before committing
- Can each function be explained in one sentence?
- Does it fail closed on bad input?
- Are safety rules visible in `rules_fired`, and does every decision carry a reason?
- Does the audit record hold enough to replay the decision?
- Are thresholds in config, and synthetic data / labels separated from logic?
- Any line that's clever but hard to explain? Simplify it.
