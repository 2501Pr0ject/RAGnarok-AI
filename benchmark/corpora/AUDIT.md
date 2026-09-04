# Corpus Audit — pre-freeze

Audit of the assembled corpora (PROTOCOL.md §2), performed after
assembly and before writing any question. Date: 2026-09-04.
Auditor: maintainer.

## Inventory

| Corpus | Files* | Size | Format | Mean / median file | Files >10K | Sections |
|---|---:|---:|---|---|---:|---|
| docker | 174 | 1.6 MB | markdown | 9.0K / 6.5K | ~55 | get-started, manuals/engine, reference/compose-file |
| python | 40 | 0.9 MB | reStructuredText | 23.6K / 18.9K | 27 | tutorial, howto |
| fastapi | 85 | 0.5 MB | markdown | 5.7K / 4.5K | 11 | tutorial, advanced |
| kubernetes | 186 | 2.3 MB | markdown | 12.4K / 8.2K | 80 | concepts (13 subsections) |

\* current-version files; each corpus additionally carries 2 old-version
pair files (8 total). Grand total: 529 files, ~6.4 MB.

Two formats are present (markdown + reStructuredText for Python); the
chunking configuration must handle both, as recorded in each manifest's
`format` field.

## Question-type potential

- **direct / contextual**: abundant everywhere. Compose-file reference
  (option semantics, defaults) and K8s concepts (feature states,
  defaults, conditions) are particularly strong.
- **multi_chunk**: strong. Deep documents (python howto median ~19K,
  80 K8s files >10K) and natural cross-concept comparisons
  (CMD-in-compose vs Dockerfile, Deployment vs StatefulSet, Query vs
  Path params, venv vs system installs).
- **trap**: good support — docs state explicit negative conditions
  (immutable selectors, unsupported option combinations) to build false
  premises against.
- **unanswerable**: by construction (subsets exclude reference/API and
  task sections), plausible questions outside the subset boundary are
  easy to pose — e.g. CLI flags (docker), stdlib reference details
  (python), deployment/production topics (fastapi reference pages),
  kubectl tasks (kubernetes).
- **contradiction**: 8 verified pairs (2 per corpus), all confirmed
  materially different at assembly time; natures documented in each
  manifest. Highlights: venv Windows activation script rename (python),
  Query/Annotated style change (fastapi), Endpoints deprecation
  (kubernetes), Compose `command` semantics (docker).

## Audit findings and actions

1. **docker: release notes excluded** (`manuals/engine/release-notes/**`,
   ~400K across 28 files, including the 3 largest files of the corpus).
   Reason: changelogs are weak question material and their dated version
   claims risk creating *unintentional* contradictions that would
   pollute the deliberate contradiction-pair design. Exclusion recorded
   with rationale in `corpora.yaml` and the docker manifest.
2. **Pair natures characterized** from actual diffs (all 8 replaced
   their `TO_VERIFY` placeholder before freeze).
3. No other anomalies: no empty files of concern (<1K files are section
   indexes), no non-text assets copied, licenses extracted for all four
   corpora (Apache-2.0, PSF-2.0, MIT, CC-BY-4.0 with attribution).

## Freeze declaration

With this audit complete, the four corpus directories
(`files/`, `LICENSE`, `MANIFEST.yaml`) are **immutable experimental
data**. Corpus-level SHA-256:

| Corpus | corpus_sha256 (prefix) |
|---|---|
| docker | `f77720152a12` |
| python | `6c03cb43b5c4` |
| fastapi | `8e4265cfd43b` |
| kubernetes | `ccb5479b114b` |

Any change requires a PROTOCOL.md amendment and full re-assembly with a
new audit. Question authoring (Phase 1, 30 pilot cases) may now begin
against these exact files.
