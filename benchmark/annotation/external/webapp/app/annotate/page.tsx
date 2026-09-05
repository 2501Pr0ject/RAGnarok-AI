"use client";
import { Suspense, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";

type Item = {
  item_key: string;
  question: string;
  reference_information: string;
  retrieved_context: { id: string; text: string }[];
  answer: string;
};

const CRITERIA: [string, string, string, string][] = [
  [
    "retrieval_relevance",
    "Retrieval relevance",
    "Could a competent person answer the question from the retrieved context?",
    "Ignore the answer for a moment and look only at the excerpts: do they contain enough to answer the question? Extra irrelevant excerpts don't matter — missing essential information does.",
  ],
  [
    "faithfulness",
    "Faithfulness",
    "Is every claim in the answer supported by the retrieved context (not by your own knowledge)?",
    "Think: could I point to the provided excerpts and show where this claim comes from? An answer can be true in the real world and still unsupported here.",
  ],
  [
    "answer_relevance",
    "Answer relevance",
    "Does it address the question asked? (Saying the info is missing counts as Yes if the context really lacks it.)",
    "Think: does the response actually answer what was asked, rather than discussing something related? If the context genuinely lacks the information, saying so explicitly is the right behavior.",
  ],
  [
    "completeness",
    "Completeness",
    "Given the reference information, is anything essential missing?",
    "Think: would leaving this information out materially change the usefulness of the answer? Less detail than the reference is fine — a missing essential is not.",
  ],
];

function AnnotatePage() {
  const token = useSearchParams().get("token") ?? "";
  const [items, setItems] = useState<Item[]>([]);
  const [doneKeys, setDoneKeys] = useState<Set<string>>(new Set());
  const [idx, setIdx] = useState(0);
  const [labels, setLabels] = useState<Record<string, number | null>>({});
  const [confidence, setConfidence] = useState("");
  const [ambiguity, setAmbiguity] = useState(false);
  const [note, setNote] = useState("");
  const [startedAt, setStartedAt] = useState("");
  const [finished, setFinished] = useState<{ annotator_id: string; cases: number } | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [justSaved, setJustSaved] = useState(false);

  useEffect(() => {
    fetch(`/api/batch?token=${token}`)
      .then((r) => r.json())
      .then((d) => {
        if (d.error) {
          try {
            if (localStorage.getItem("ragnarok_token") === token) localStorage.removeItem("ragnarok_token");
          } catch {}
          return setError(d.error);
        }
        try {
          localStorage.setItem("ragnarok_token", token);
        } catch {}
        setItems(d.items);
        const done = new Set<string>(d.done);
        setDoneKeys(done);
        const first = d.items.findIndex((i: Item) => !done.has(i.item_key));
        setIdx(first === -1 ? 0 : first);
      });
  }, [token]);

  const item = items[idx];
  useEffect(() => {
    setLabels({});
    setConfidence("");
    setAmbiguity(false);
    setNote("");
    setStartedAt(new Date().toISOString());
  }, [idx, items.length]);

  const needsNote = useMemo(
    () => Object.values(labels).some((v) => v === 0) || confidence === "low",
    [labels, confidence],
  );
  const complete =
    CRITERIA.every(([k]) => labels[k] === 0 || labels[k] === 1) &&
    ["low", "medium", "high"].includes(confidence) &&
    (!needsNote || note.trim().length > 0);

  async function submit() {
    setBusy(true);
    setError("");
    const res = await fetch("/api/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token, item_key: item.item_key, labels, confidence, ambiguity, note, started_at: startedAt }),
    });
    const d = await res.json();
    setBusy(false);
    if (!res.ok) return setError(d.error ?? "submit failed");
    const done = new Set(doneKeys).add(item.item_key);
    setDoneKeys(done);
    if (done.size >= items.length) {
      const fin = await fetch("/api/complete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token }),
      });
      const fd = await fin.json();
      if (fin.ok) {
        try {
          localStorage.removeItem("ragnarok_token");
        } catch {}
        return setFinished(fd);
      }
      return setError(fd.error ?? "completion failed");
    }
    setJustSaved(true);
    setTimeout(() => setJustSaved(false), 2500);
    const next = items.findIndex((i) => !done.has(i.item_key));
    setIdx(next);
  }

  if (finished)
    return (
      <div className="card">
        <h1>Thank you!</h1>
        <p>
          <b>
            {finished.cases} / {finished.cases} cases completed.
          </b>{" "}
          Your annotations are now part of the study&rsquo;s human reference.
        </p>
        <p>
          Your anonymous annotation ID: <b>{finished.annotator_id}</b>
        </p>
        <p className="muted">You can keep this ID if you ever want to reference or withdraw your contribution.</p>
      </div>
    );
  if (error && !item) return <div className="card">{error}</div>;
  if (!item) return <div className="card">Loading…</div>;

  return (
    <>
      <div className="progress">
        <div style={{ width: `${(doneKeys.size / items.length) * 100}%` }} />
      </div>
      <p className="muted">
        <b>Case {doneKeys.size + 1} of {items.length}</b> &middot; {doneKeys.size} completed &middot; about
        3 minutes per case {justSaved && <span className="saved">Saved</span>}
      </p>
      <p className="muted">
        Your progress is saved automatically after each case. To stop and resume later, just keep this
        page&rsquo;s link (or come back to the start page on this device).
      </p>

      <div className="card">
        <h2 style={{ marginTop: 0 }}>Question</h2>
        <p>{item.question}</p>
      </div>

      <div className="card">
        <h2 style={{ marginTop: 0 }}>Retrieved context — what the system had</h2>
        {item.retrieved_context.map((c, i) => (
          <div className="context" key={c.id}>
            <span className="pill">excerpt {i + 1}</span>
            {"\n"}
            {c.text}
          </div>
        ))}
      </div>

      <div className="card">
        <h2 style={{ marginTop: 0 }}>Generated answer — what you are judging</h2>
        <div className="context">{item.answer}</div>
      </div>

      <div className="card">
        <h2 style={{ marginTop: 0 }}>Reference information</h2>
        <p className="muted">
          Background from the documentation, for your judgment — not a text to compare word-for-word.
        </p>
        <div className="context">{item.reference_information}</div>
      </div>

      <div className="card">
        <h2 style={{ marginTop: 0 }}>Your evaluation</h2>
        <p className="muted">
          There are no trick questions — judge only what is shown on this page, and do not use outside
          knowledge when judging faithfulness or retrieval relevance. When unsure, choose the option that
          best matches the evidence and use the confidence field to indicate uncertainty.
        </p>
        {CRITERIA.map(([key, title, help, hint]) => (
          <div className="crit" key={key}>
            <p>
              <b>{title}</b>
              <br />
              <span className="muted">{help}</span>
              <details className="hint">
                <summary>How to think about it</summary>
                <span>{hint}</span>
              </details>
            </p>
            <label>
              <input type="radio" name={key} checked={labels[key] === 1} onChange={() => setLabels({ ...labels, [key]: 1 })} /> Yes
            </label>
            <label>
              <input type="radio" name={key} checked={labels[key] === 0} onChange={() => setLabels({ ...labels, [key]: 0 })} /> No
            </label>
          </div>
        ))}
        <div className="crit">
          <p>
            <b>Confidence</b>
          </p>
          {["low", "medium", "high"].map((v) => (
            <label key={v}>
              <input type="radio" name="conf" checked={confidence === v} onChange={() => setConfidence(v)} /> {v}
            </label>
          ))}
        </div>
        <div className="crit">
          <p>
            <b>This case felt ambiguous or badly posed</b>
          </p>
          <label>
            <input type="checkbox" checked={ambiguity} onChange={(e) => setAmbiguity(e.target.checked)} /> flag
          </label>
        </div>
        <p>
          <b>Comment</b> {needsNote ? <span className="muted">(required: you answered No somewhere or confidence is low)</span> : <span className="muted">(optional)</span>}
          <br />
          <span className="muted">Please do not include personal information — comments may be published with the dataset.</span>
        </p>
        <textarea rows={2} value={note} onChange={(e) => setNote(e.target.value)} />
        <p style={{ marginTop: "1rem" }}>
          <button onClick={submit} disabled={!complete || busy}>
            Submit &amp; next
          </button>
        </p>
        {error && <p style={{ color: "#c0392b" }}>{error}</p>}
      </div>
    </>
  );
}

export default function Page() {
  return (
    <Suspense fallback={<div className="card">Loading…</div>}>
      <AnnotatePage />
    </Suspense>
  );
}
