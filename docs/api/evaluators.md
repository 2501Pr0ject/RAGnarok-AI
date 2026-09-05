# Evaluators

Reference for RAGnarok-AI evaluators.

---

## Retrieval Metrics

### Import

```python
from ragnarok_ai.evaluators import (
    precision_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
)
```

### precision_at_k

```python
def precision_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int = 10,
) -> float
```

Fraction of retrieved documents that are relevant.

**Parameters:**

- `retrieved` — List of retrieved document IDs
- `relevant` — List of relevant document IDs (ground truth)
- `k` — Number of top results to consider

**Returns:** Float between 0.0 and 1.0

**Example:**

```python
p = precision_at_k(
    retrieved=["doc_1", "doc_2", "doc_3"],
    relevant=["doc_1", "doc_3"],
    k=3,
)
# p = 2/3 = 0.67
```

---

### recall_at_k

```python
def recall_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int = 10,
) -> float
```

Fraction of relevant documents that were retrieved.

**Example:**

```python
r = recall_at_k(
    retrieved=["doc_1", "doc_2"],
    relevant=["doc_1", "doc_3", "doc_4"],
    k=10,
)
# r = 1/3 = 0.33 (found 1 of 3 relevant docs)
```

---

### mrr

```python
def mrr(
    retrieved: list[str],
    relevant: list[str],
) -> float
```

Mean Reciprocal Rank — inverse of the position of the first relevant document.

**Example:**

```python
m = mrr(
    retrieved=["doc_2", "doc_1", "doc_3"],
    relevant=["doc_1", "doc_3"],
)
# m = 1/2 = 0.5 (first relevant doc at position 2)
```

---

### ndcg_at_k

```python
def ndcg_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int = 10,
) -> float
```

Normalized Discounted Cumulative Gain — measures ranking quality.

**Example:**

```python
n = ndcg_at_k(
    retrieved=["doc_1", "doc_2", "doc_3"],
    relevant=["doc_1", "doc_3"],
    k=3,
)
```

---

## RetrievalMetrics

Combined retrieval metrics.

```python
from ragnarok_ai.evaluators.retrieval import RetrievalMetrics, evaluate_retrieval

result = RetrievalResult(
    query=query,
    retrieved_docs=docs,
    scores=[0.9, 0.8, 0.7],
)

metrics: RetrievalMetrics = evaluate_retrieval(result, k=10)

print(metrics.precision)  # 0.67
print(metrics.recall)     # 0.50
print(metrics.mrr)        # 1.0
print(metrics.ndcg)       # 0.85
print(metrics.k)          # 10
```

---

## LLM-as-Judge

### LLMJudge

```python
from ragnarok_ai.evaluators.judge import LLMJudge

judge = LLMJudge(
    model: str | None = None,      # Default: Prometheus 2
    base_url: str = "http://localhost:11434",
    medical_mode: bool = False,
)
```

**Methods:**

### evaluate_faithfulness

```python
async def evaluate_faithfulness(
    context: str,
    question: str,
    answer: str,
) -> JudgeResult
```

Check if answer is grounded in context.

### evaluate_relevance

```python
async def evaluate_relevance(
    question: str,
    answer: str,
) -> JudgeResult
```

Check if answer addresses the question.

### detect_hallucination

```python
async def detect_hallucination(
    context: str,
    answer: str,
) -> JudgeResult
```

Check for fabricated information.

### evaluate_completeness

```python
async def evaluate_completeness(
    question: str,
    answer: str,
    context: str,
) -> JudgeResult
```

Check if all aspects are covered.

---

### JudgeResult

```python
from ragnarok_ai.evaluators.judge import JudgeResult

result = await judge.evaluate_faithfulness(context, question, answer)

print(result.verdict)      # "PASS", "PARTIAL", or "FAIL"
print(result.score)        # Float 0.0-1.0
print(result.explanation)  # Detailed explanation
```

| Field | Type | Description |
|-------|------|-------------|
| `verdict` | `str` | PASS, PARTIAL, or FAIL |
| `score` | `float` | Normalized score (0.0-1.0) |
| `explanation` | `str` | Detailed reasoning |

---

## FaithfulnessEvaluator

Lower-level faithfulness evaluator.

```python
from ragnarok_ai.evaluators import FaithfulnessEvaluator
from ragnarok_ai.adapters.llm import OllamaLLM

async with OllamaLLM() as llm:
    evaluator = FaithfulnessEvaluator(llm, medical_mode=True)
    result = await evaluator.evaluate(context, question, answer)
```

---

## RelevanceEvaluator

Measures whether the generated answer addresses the original question. Implements `EvaluatorProtocol` for use in evaluation pipelines.

```python
from ragnarok_ai.evaluators import RelevanceEvaluator
from ragnarok_ai.adapters.llm import OllamaLLM

async with OllamaLLM() as llm:
    evaluator = RelevanceEvaluator(llm)
    score = await evaluator.evaluate(
        response="Paris is the capital of France.",
        query="What is the capital of France?",
    )
```

---

## HallucinationDetector

Extracts the claims made by an answer and checks each one against the provided context. The score is `hallucinated_claims / total_claims`: 0.0 means no hallucination, 1.0 means every claim is unsupported. Implements `EvaluatorProtocol`.

```python
from ragnarok_ai.evaluators import HallucinationDetector
from ragnarok_ai.adapters.llm import OllamaLLM

async with OllamaLLM() as llm:
    detector = HallucinationDetector(llm)
    score = await detector.evaluate(
        response="Paris, founded in 500 BC, is the capital of France.",
        context="France is a country in Europe. Its capital is Paris.",
    )
```

---

## Medical Evaluation

Medical-domain utilities for evaluating RAG over clinical text, where ambiguous abbreviations (is "MS" multiple sclerosis or mitral stenosis?) can silently distort scores.

### MedicalAbbreviationNormalizer

Expands medical abbreviations before evaluation, using a built-in dictionary plus context keywords for ambiguous cases.

```python
from ragnarok_ai.evaluators.medical import MedicalAbbreviationNormalizer

normalizer = MedicalAbbreviationNormalizer()
text, expansions = normalizer.normalize_text("CHF with EF 30%")
# text: "congestive heart failure with ejection fraction 30%"
# expansions: ["CHF → congestive heart failure", "EF → ejection fraction"]
```

Options: `custom_abbreviations` (extra abbreviation → full-form pairs), `context_window` (words considered around an ambiguous abbreviation), and `disambiguator` (escalation strategy, below).

### SLMDisambiguator

Resolves ambiguous abbreviations with a small local language model (e.g. `qwen2.5:0.5b`, `phi3:mini`), consulted by `normalize_text_async` only when context keywords are inconclusive. Decisions are memoized.

```python
from ragnarok_ai.evaluators.medical import MedicalAbbreviationNormalizer, SLMDisambiguator
from ragnarok_ai.adapters.llm import OllamaLLM

async with OllamaLLM(model="qwen2.5:0.5b") as llm:
    normalizer = MedicalAbbreviationNormalizer(disambiguator=SLMDisambiguator(llm))
    text, expansions = await normalizer.normalize_text_async(clinical_note)
```

Any object implementing the `DisambiguationStrategy` protocol (a `resolve` method) can be plugged in instead.

Related: `LLMJudge(medical_mode=True)` and `FaithfulnessEvaluator(medical_mode=True)` apply medical normalization during judging. See the [LLM-as-Judge guide](../user-guide/judge.md) for the full medical workflow.

---

## Next Steps

- [Core Types](types.md) — Type reference
- [Adapters](adapters.md) — LLM and vector store adapters
