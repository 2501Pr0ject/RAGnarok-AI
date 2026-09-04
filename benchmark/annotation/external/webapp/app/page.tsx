"use client";
import { useState } from "react";

const DOMAINS = [
  ["docker", "Docker"],
  ["python", "Python"],
  ["fastapi", "FastAPI"],
  ["kubernetes", "Kubernetes"],
];
const LEVELS = ["beginner", "intermediate", "advanced", "expert", "undisclosed"];

export default function Landing() {
  const [domain, setDomain] = useState("");
  const [level, setLevel] = useState("undisclosed");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function start() {
    setBusy(true);
    setError("");
    const params = new URLSearchParams(window.location.search);
    const res = await fetch("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ domain, level, source: params.get("src") ?? "direct", smoke: params.get("smoke") === "1" }),
    });
    const data = await res.json();
    if (!res.ok) {
      setError(data.error ?? "something went wrong");
      setBusy(false);
      return;
    }
    window.location.href = `/annotate?token=${data.token}`;
  }

  const smoke = typeof window !== "undefined" && new URLSearchParams(window.location.search).get("smoke") === "1";

  return (
    <>
      <h1>RAGnarok Human Evaluation</h1>
      <div className="card">
        <p>
          Help an <b>open-source research study</b> measure how reliably AI answers over technical
          documentation can be evaluated. You will review 10&ndash;15 anonymized answers: for each, the
          question, the documentation excerpts the system used, and its answer.
        </p>
        <p className="muted">
          ~30&ndash;45 minutes &middot; no account, no email, no personal data &middot; you can stop anytime
          &middot; the methodology is fully public in the study repository.
        </p>
      </div>
      <div className="card">
        <h2>Which technology do you know?</h2>
        {smoke ? (
          <p className="pill">Internal smoke-test batch (mixed domains)</p>
        ) : (
          <p>
            {DOMAINS.map(([v, label]) => (
              <label key={v}>
                <input type="radio" name="domain" checked={domain === v} onChange={() => setDomain(v)} /> {label}
              </label>
            ))}
          </p>
        )}
        <h2>Your experience with it (optional)</h2>
        <p>
          {LEVELS.map((v) => (
            <label key={v}>
              <input type="radio" name="level" checked={level === v} onChange={() => setLevel(v)} /> {v}
            </label>
          ))}
        </p>
        <p>
          <button onClick={start} disabled={busy || (!smoke && !domain)}>
            Start annotating
          </button>
        </p>
        {error && <p style={{ color: "#c0392b" }}>{error}</p>}
        <p className="muted">
          By starting, you agree that your anonymous judgments become part of an openly published research
          dataset. Please read the annotator guide linked from where you received this study.
        </p>
      </div>
    </>
  );
}
