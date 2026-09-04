# Judge Calibration

The first objection to LLM-as-judge — especially with local models — is *"can I trust the judge?"*. Calibration answers it with data instead of faith: label a small set of examples yourself, run the judge on them, and measure how often it agrees with you.

## Step 1 — Label a sample set

Label 20-50 (question, context, answer) triples. Real examples from your own evaluations are best — include both good and bad answers:

```python
from ragnarok_ai import CalibrationSample, CalibrationSet

calset = CalibrationSet(
    name="medical-v1",
    samples=[
        CalibrationSample(
            question="What is the first-line treatment for CHF?",
            context="ACE inhibitors are first-line therapy for CHF...",
            answer="ACE inhibitors.",
            human_pass=True,
        ),
        CalibrationSample(
            question="What is the first-line treatment for CHF?",
            context="ACE inhibitors are first-line therapy for CHF...",
            answer="Beta-blockers are the only first-line option.",
            human_pass=False,
            note="Contradicts the context",
        ),
        # An answer can be grounded but off-topic — override per criterion:
        CalibrationSample(
            question="What is the CHF dosage protocol?",
            context="...",
            answer="CHF stands for congestive heart failure.",
            human_pass=True,
            human_labels={"relevance": False},
        ),
    ],
)
calset.save("calibration-set.json")  # version it with your repo
```

## Step 2 — Calibrate

```python
from ragnarok_ai import JudgeCalibrator, LLMJudge

calibrator = JudgeCalibrator(LLMJudge(medical_mode=True))
report = await calibrator.calibrate(calset, criteria=["faithfulness", "relevance"])
print(report.summary())
```

```
Judge calibration on 40 labeled samples

faithfulness:
  agreement 88% (95% CI 74%-95%), kappa 0.72 (substantial)
  false accepts 8%, false rejects 14%
  recommended threshold 0.65 (kappa 0.78, current 0.70)
```

## Reading the report

- **Kappa** is chance-corrected agreement ([Landis & Koch](https://doi.org/10.2307/2529310) bands: 0.6+ substantial, 0.8+ almost perfect). Raw accuracy can look great on an imbalanced set; kappa cannot be gamed by always saying PASS.
- **False accepts vs. false rejects** split disagreement by direction. False accepts (the judge passes what you rejected) are the dangerous kind — bad answers slipping into production unnoticed. False rejects just add noise.
- **Recommended threshold**: the pass cutoff (default 0.7) is swept and the value maximizing kappa on *your* labels is suggested — local judges often run systematically strict or lenient, and shifting the cutoff is free accuracy. Ties keep the current threshold, so the recommendation only moves when the data supports it.
- **Confidence interval**: a 95% Wilson interval on agreement — with 20 labels the interval is wide, which is exactly the point of showing it.
- **`report.criteria[c].disagreements`** lists the sample indices where you and the judge differ: review them — sometimes the judge is wrong, sometimes your label is.

## Practical advice

- **Cost**: one judge call per sample per criterion — 40 samples × 2 criteria ≈ 80 calls (~30 min with local Prometheus 2; run it once per judge model, not per evaluation).
- Fewer than 20 labels flags the report `insufficient_data` — numbers are shown but should be treated as indicative.
- Re-calibrate when you **change the judge model** or meaningfully change your domain; the labeled set is reusable.
- A kappa below 0.4 on a criterion means the judge's scores on that criterion should not gate your CI — fix the judge (better model, tuned threshold) before trusting the metric.
