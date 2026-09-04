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

const CRITERIA: [string, string, string][] = [
  ["retrieval_relevance", "Retrieval relevance", "Could a competent person answer the question from the retrieved context?"],
  ["faithfulness", "Faithfulness", "Is every claim in the answer supported by the retrieved context (not by your own knowledge)?"],
  ["answer_relevance", "Answer relevance", "Does it address the question asked? (Saying the info is missing counts as Yes if the context really lacks it.)"],
  ["completeness", "Completeness", "Given the reference information, is anything essential missing?"],
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

  useEffect(() => {
    fetch(`/api/batch?token=${token}`)
      .then((r) => r.json())
      .then((d) => {
        if (d.error) return setError(d.error);
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
      if (fin.ok) return setFinished(fd);
      return setError(fd.error ?? "completion failed");
    }
    const next = items.findIndex((i) => !done.has(i.item_key));
    setIdx(next);
  }

  if (finished)
    return (
      <div className="card">
        <h1>Thank you!</h1>
        <p>
          {finished.cases} / {finished.cases} cases completed.
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
        Case {doneKeys.size + 1} of {items.length}
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
        {CRITERIA.map(([key, title, help]) => (
          <div className="crit" key={key}>
            <p>
              <b>{title}</b>
              <br />
              <span className="muted">{help}</span>
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
