"use client";
import { useEffect, useState } from "react";

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
  const [resumeToken, setResumeToken] = useState("");
  const [search, setSearch] = useState("");

  useEffect(() => {
    setSearch(window.location.search);
    try {
      setResumeToken(localStorage.getItem("ragnarok_token") ?? "");
    } catch {}
  }, []);

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

  const smoke = new URLSearchParams(search).get("smoke") === "1";

  return (
    <>
      <div className="topbar">
        <h1>RAGnarok Human Evaluation</h1>
        <a href={`/about${search}`}>About this study &rarr;</a>
      </div>

      <div className="card">
        <h2 style={{ marginTop: 0 }}>Can we trust AI-based evaluation of RAG systems?</h2>
        <p>
          RAG systems (retrieval-augmented generation) answer questions using information retrieved from
          technical documentation. Increasingly, <b>automated evaluators</b> — including AI judges — are
          used to assess whether those answers are relevant, faithful to their sources, and complete. This
          open-source research study measures how well those automated evaluations agree with{" "}
          <b>human judgment</b>: yours.
        </p>
        <p>
          Your role is simple: review a small set of anonymized RAG answers and judge them against the
          documentation excerpts provided. No coding required.
        </p>
        <div className="blocks">
          <div className="block">
            <b>🔬 The study</b>
            A reproducible, openly published study on human vs. automated RAG evaluation.
          </div>
          <div className="block">
            <b>👤 Your role</b>
            Review 10&ndash;15 anonymized cases and answer four yes/no questions about each.
          </div>
          <div className="block">
            <b>⏱️ Your time</b>
            About 30&ndash;45 minutes total. Progress is saved after each case — you can stop and resume
            later.
          </div>
        </div>
        <p className="muted">
          No account or email is required. No personal information is collected. The full methodology is
          public — see <a href={`/about${search}`}>About this study</a>.
        </p>
      </div>

      {resumeToken && (
        <div className="card">
          <p style={{ margin: 0 }}>
            <b>You have an annotation session in progress on this device.</b>{" "}
            <a href={`/annotate?token=${resumeToken}`}>Resume where you left off &rarr;</a>
          </p>
        </div>
      )}

      <div className="card">
        <h2>Which technology are you most comfortable evaluating?</h2>
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
        <h2>How familiar are you with it? (optional)</h2>
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
          By continuing, you agree that your anonymous annotations may be included in the publicly released
          research dataset and study results. No email, name, account, or other directly identifying
          information is required. Please read the annotator guide linked from where you received this
          study.
        </p>
      </div>
    </>
  );
}
