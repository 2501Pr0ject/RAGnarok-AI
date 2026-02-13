# LLM-as-Judge

Evaluate answer quality using Prometheus 2, a specialized evaluation model.

---

## Overview

LLM-as-Judge provides multi-criteria evaluation:

| Criterion | Description |
|-----------|-------------|
| Faithfulness | Is the answer grounded in the provided context? |
| Relevance | Does the answer address the question? |
| Hallucination | Does the answer contain fabricated information? |
| Completeness | Are all aspects of the question covered? |

---

## Setup

Install Prometheus 2 via Ollama:

```bash
ollama pull hf.co/RichardErkhov/prometheus-eval_-_prometheus-7b-v2.0-gguf:Q5_K_M
```

Requirements:

- ~5GB disk space
- 16GB RAM recommended
- Ollama running (`ollama serve`)

---

## Basic Usage

### Python API

```python
from ragnarok_ai.evaluators.judge import LLMJudge

judge = LLMJudge()

# Evaluate faithfulness
result = await judge.evaluate_faithfulness(
    context="Paris is the capital of France. It has a population of 2.1 million.",
    question="What is the capital of France?",
    answer="Paris is the capital of France.",
)

print(f"Verdict: {result.verdict}")      # PASS
print(f"Score: {result.score:.2f}")       # 0.85
print(f"Explanation: {result.explanation}")
```

### CLI

```bash
ragnarok judge \
  --context "Paris is the capital of France." \
  --question "What is the capital of France?" \
  --answer "Paris is the capital of France."
```

---

## Evaluation Criteria

### Faithfulness

Checks if the answer is grounded in the provided context.

```python
result = await judge.evaluate_faithfulness(
    context="Python was created by Guido van Rossum in 1991.",
    question="Who created Python?",
    answer="Python was created by Guido van Rossum.",
)
# PASS - answer is supported by context
```

### Relevance

Checks if the answer addresses the question.

```python
result = await judge.evaluate_relevance(
    question="What is the capital of France?",
    answer="Paris is the capital of France.",
)
# PASS - directly answers the question
```

### Hallucination Detection

Checks for fabricated information not in the context.

```python
result = await judge.detect_hallucination(
    context="Python was created by Guido van Rossum.",
    answer="Python was created by Guido van Rossum in the Netherlands in 1991.",
)
# PARTIAL - "1991" and "Netherlands" not in context
```

### Completeness

Checks if the answer covers all aspects of the question.

```python
result = await judge.evaluate_completeness(
    question="What is Python and who created it?",
    answer="Python is a programming language.",
    context="Python is a programming language created by Guido van Rossum.",
)
# PARTIAL - missing creator information
```

---

## Batch Evaluation

Evaluate multiple items from a file:

```bash
# items.json
# [
#   {"context": "...", "question": "...", "answer": "..."},
#   {"context": "...", "question": "...", "answer": "..."}
# ]

ragnarok judge --file items.json --criteria faithfulness,relevance
```

---

## Select Criteria

Evaluate specific criteria only:

```bash
# Single criterion
ragnarok judge --file items.json --criteria faithfulness

# Multiple criteria
ragnarok judge --file items.json --criteria faithfulness,relevance

# All criteria (default)
ragnarok judge --file items.json --criteria all
```

---

## Medical Mode

Reduce false positives in healthcare RAG evaluation:

```python
judge = LLMJudge(medical_mode=True)

result = await judge.evaluate_faithfulness(
    context="Patient diagnosed with CHF.",
    question="What condition does the patient have?",
    answer="Patient has congestive heart failure.",
)
# PASS - "CHF" and "congestive heart failure" are equivalent
```

Features:

- 350+ medical abbreviations (CHF, MI, COPD, DVT...)
- Context-aware disambiguation
- Multiple formats: dotted (q.d.), slash (s/p), mixed-case (SpO2)

---

## Scoring

Prometheus 2 uses a 1-5 rubric, normalized to 0-1:

| Raw Score | Normalized | Verdict |
|-----------|------------|---------|
| 5 | 1.0 | PASS |
| 4 | 0.75 | PASS |
| 3 | 0.5 | PARTIAL |
| 2 | 0.25 | FAIL |
| 1 | 0.0 | FAIL |

Verdict thresholds:

- **PASS**: score >= 0.7
- **PARTIAL**: 0.4 <= score < 0.7
- **FAIL**: score < 0.4

---

## Performance

On Apple M2 16GB:

| Criterion | Avg Time |
|-----------|----------|
| Faithfulness | ~25s |
| Relevance | ~22s |
| Hallucination | ~28s |
| Completeness | ~24s |

!!! tip "Keep Alive"
    RAGnarok-AI uses `keep_alive` by default to prevent Ollama from unloading the model between requests.

---

## Configuration

### Custom Model

```python
judge = LLMJudge(
    model="llama3",  # Use different model
    base_url="http://localhost:11434",
)
```

### CLI Options

```bash
ragnarok judge \
  --file items.json \
  --model llama3 \
  --ollama-url http://localhost:11434 \
  --fail-under 0.7 \
  --output results.json
```

---

## Output Formats

### Console Output

```
  RAGnarok-AI LLM-as-Judge
  ========================================

  Items to evaluate: 1
  Criteria: faithfulness, relevance

  [1/1] Evaluating: What is the capital of France?

  ----------------------------------------
  Results
  ----------------------------------------

  Item 1:
    Question: What is the capital of France?
    [+] faithfulness: 0.85 (PASS)
    [+] relevance: 0.90 (PASS)
    Average: 0.88

  ----------------------------------------
  Overall Average: 0.8750
```

### JSON Output

```bash
ragnarok judge --file items.json --json
```

```json
{
  "command": "judge",
  "status": "pass",
  "version": "1.4.0",
  "data": {
    "items_evaluated": 1,
    "criteria": ["faithfulness", "relevance"],
    "results": [
      {
        "question": "What is the capital of France?",
        "criteria": {
          "faithfulness": {"verdict": "PASS", "score": 0.85, "explanation": "..."},
          "relevance": {"verdict": "PASS", "score": 0.90, "explanation": "..."}
        },
        "average_score": 0.875
      }
    ],
    "overall_average": 0.875
  }
}
```

---

## CI/CD Integration

Use `--fail-under` for quality gates:

```bash
ragnarok judge --file items.json --fail-under 0.7

# Exit code 0 if average >= 0.7
# Exit code 1 if average < 0.7
```

!!! warning "Advisory Scores"
    LLM-as-Judge scores are advisory. For CI/CD, consider using `fail-on-threshold: false` in the GitHub Action.

---

## Next Steps

- [Evaluation Guide](evaluation.md) — Retrieval metrics
- [GitHub Action](../ci-cd/github-action.md) — CI/CD integration
- [CLI Reference](../ci-cd/cli-reference.md) — Full command reference
