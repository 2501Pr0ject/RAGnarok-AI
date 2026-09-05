"use client";
import { useEffect, useState } from "react";

export default function About() {
  const [search, setSearch] = useState("");
  useEffect(() => setSearch(window.location.search), []);

  return (
    <>
      <div className="topbar">
        <h1>About this study</h1>
        <a href={`/${search}`}>&larr; Back to the study</a>
      </div>

      <div className="card">
        <h2 style={{ marginTop: 0 }}>What are we studying?</h2>
        <p>
          Several RAG systems (retrieval-augmented generation: a search step feeding an AI answer
          generator) answered questions about public technical documentation. Automated evaluation
          methods — including local AI judges — can score those answers at scale, cheaply and
          consistently. But an important question remains open: <b>can we trust those automated
          evaluations?</b> This study measures how well they agree with human judgment.
        </p>

        <h2>Why human evaluation?</h2>
        <p>
          Automated judges are useful because they can evaluate large numbers of answers consistently
          and cheaply. But they are not ground truth. Human judgment provides an independent reference
          against which automated evaluations can be compared. Your annotations — together with other
          annotators&rsquo; — form the human reference used to measure how closely automated evaluation
          agrees with people who actually understand the technology.
        </p>

        <h2>What is RAGnarok?</h2>
        <p>
          RAGnarok is an open-source framework for evaluating RAG systems locally and reproducibly. It
          can evaluate different aspects of a RAG pipeline and use local AI judges to assess generated
          answers. This study is not about confirming that the tool works — it is designed to
          investigate whether automated evaluations like the ones it produces can be trusted at all.
        </p>

        <h2>How were the documents and answers selected?</h2>
        <p>
          The corpus is frozen snapshots of public technical documentation (Docker, Python, FastAPI,
          Kubernetes), so the study is reproducible. The systems&rsquo; answers are shown anonymized and
          shuffled: you are never told which system produced an answer, and the item&ndash;system mapping
          never reaches the browser. There is no expected verdict for any case — some answers are good,
          some are bad, some are in between.
        </p>

        <h2>How is anonymity handled?</h2>
        <p>
          No account, email, or personal information is collected. You are identified only by a
          generated pseudonymous ID (e.g. <b>ANN-7F42</b>), shown to you when you finish. Please do not
          include personal information in the free-text comment fields.
        </p>

        <h2>What will happen to my annotations?</h2>
        <p>
          Your anonymous annotations may be included in the publicly released research dataset and study
          results, alongside the full methodology. If you keep your annotator ID, you can reference or
          withdraw your contribution later through the contact given where you received your
          participation link.
        </p>

        <p>
          <a href="https://github.com/2501Pr0ject/RAGnarok-AI/tree/main/benchmark" target="_blank" rel="noreferrer">
            Open methodology on GitHub &rarr;
          </a>
        </p>
      </div>
    </>
  );
}
