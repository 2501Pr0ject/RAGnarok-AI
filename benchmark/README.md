# RAGnarok Evaluation Study

A reproducible benchmark asking one question with data instead of faith:

> **Can a local-first RAG evaluation framework produce reliable,
> reproducible quality measurements — and how closely do local LLM
> judges reproduce human judgment?**

Built from public technical documentation (Docker, Python, FastAPI,
Kubernetes), with controlled-difficulty questions, human reference
labels, and deliberately degraded RAG configurations that the evaluation
must be able to tell apart.

- **[PROTOCOL.md](PROTOCOL.md)** — the frozen study protocol: corpora,
  question taxonomy, RAG configurations, judges, analyses,
  reproducibility requirements. Read this first.
- **[ANNOTATION.md](ANNOTATION.md)** — the frozen human annotation
  rules.
- **[schemas/](schemas/)** — machine-validatable formats for benchmark
  cases and human annotations.

Status: **protocol phase**. No questions are written until the protocol
is frozen; results land in `results/` as the phases described in
PROTOCOL.md §3.2 complete.

The study runs on a tagged ragnarok-ai release and uses the framework's
own `calibration` module (v1.10.0) as the measurement instrument for
judge-human agreement.
