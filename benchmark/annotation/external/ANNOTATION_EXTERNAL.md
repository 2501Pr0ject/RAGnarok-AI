# Annotator Guide — RAGnarok Human Evaluation Study

Thank you for participating! This guide is everything you need. Reading
it takes ~5 minutes; annotating a batch of 12-15 cases takes ~30-45
minutes. You can stop at any time.

## What this study is

We are studying how reliably **RAG systems** (retrieval-augmented
generation: a search step feeding an AI answer generator) can be
evaluated. Several RAG systems answered questions about public technical
documentation (Docker, Python, FastAPI, Kubernetes). Your job is to
judge the quality of individual answers. Your judgments — together with
other annotators' — form the human reference that automated evaluation
methods will later be compared against.

The full methodology, corpus, and protocol are public in this
repository. **No personal data is collected**: you are identified only
by a generated pseudonymous ID (e.g. `ANN-7F42`).

## What you will see for each case

- **Question** — what a user asked.
- **Retrieved context** — the documentation excerpts the system found
  and used. This is all the system had; it may be good, partial,
  or completely off-topic.
- **Generated answer** — what the system produced.
- **Reference information** — background gathered by the study authors
  from the same documentation, describing what the documentation
  actually says on the topic. **It is context for your judgment, not
  "the correct answer" to compare word-for-word.** An answer can be
  worded completely differently and still be excellent.

There is no expected verdict for any case. Some answers are good, some
are bad, some are in between — judge each on its own.

## The four questions to answer (Yes = 1 / No = 0)

Answer them in this order:

### 1. Retrieval relevance — *could a competent person answer the question from the retrieved context?*

- **Yes** if the retrieved excerpts contain enough information to answer
  (extra irrelevant excerpts don't matter).
- **No** if essential information is missing from the excerpts — even if
  the answer happens to be good anyway.

### 2. Faithfulness — *is every claim in the answer supported by the retrieved context?*

- Judge against the **retrieved context only** — not against your own
  knowledge of the technology. An answer can be true in the real world
  and still unfaithful here, if the context does not support it.
- Paraphrase and summarizing are fine. Added specifics (version numbers,
  defaults, flags, commands) that are **not** in the context → **No**.
- An answer that only says the information is not available makes no
  claims → **Yes**.

### 3. Answer relevance — *does it address the question that was asked?*

- **No** if it is off-topic, answers a different question, or restates
  the question without answering.
- If you believe the context genuinely does not contain the needed
  information, then an answer that **says so explicitly** is the right
  behavior → **Yes**. Merely echoing a link or heading that mentions
  the topic is not an answer and not an abstention → **No**.

### 4. Completeness — *is anything essential missing?*

- Judge against the essentials of the **reference information** (not
  against everything the documentation could say). Less detail is fine.
- A comparison question answered on only one side, or an important
  condition/exception omitted → **No**.
- If the question rests on an assumption that the retrieved context
  contradicts, a complete answer **corrects that assumption**. An answer
  that accepts a false premise and builds on it is not faithful to the
  context (see question 2).
- If the retrieved context contains **conflicting information** (e.g.
  two versions of the same page), a complete answer acknowledges or
  scopes the conflict; silently picking one value as if it were
  unambiguous → **No**.

## Then, for each case

- **Confidence**: high (obvious) / medium (had to think) / low (another
  annotator might disagree — please add a comment).
- **Ambiguity flag**: tick it if the case itself felt ambiguous or
  badly posed (unclear question, rules hard to apply). This is
  valuable signal — flag it, don't try to compensate for it.
- **Comment** (optional, but required when you answer No somewhere or
  your confidence is low): one sentence on why.

## Ground rules

- Judge **only from what is shown**. Please don't look up the live
  documentation or the study repository while annotating — the excerpts
  shown are the study's fixed reference.
- Don't try to guess what the system "meant" or which system produced
  the answer. Judge the text in front of you.
- Empty answer: relevance No, completeness No, faithfulness Yes (no
  claims made).
- Work case by case, in the order presented; there is no time limit,
  but if a case takes you more than ~4-5 minutes, flag it as ambiguous
  and move on.

## About blindness (honest note)

You are not told which system produced each answer, and cases are
shuffled. However, blindness has limits we openly acknowledge: the
shape of the context (one tiny excerpt vs several long ones) may hint
at the kind of system. That's fine — just judge the answer against its
context using the rules above, and resist the temptation to guess.

## Questions or problems

Use the comment field, or the contact given where you received your
participation link. Thank you — every carefully annotated case directly
improves how honestly AI systems can be evaluated.
