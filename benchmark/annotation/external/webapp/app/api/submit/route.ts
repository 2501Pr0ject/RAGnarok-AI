import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { loadBatch } from "@/lib/batches";

const CRITERIA = ["retrieval_relevance", "faithfulness", "answer_relevance", "completeness"];

export async function POST(req: NextRequest) {
  const body = await req.json();
  const client = db();
  const { data: ann } = await client.from("annotators").select("*").eq("token", body.token ?? "").single();
  if (!ann) return NextResponse.json({ error: "unknown token" }, { status: 403 });

  const batch = loadBatch(ann.batch_id);
  if (!batch.items.some((i) => i.item_key === body.item_key)) {
    return NextResponse.json({ error: "item not in your batch" }, { status: 400 });
  }
  const labels = body.labels ?? {};
  if (!CRITERIA.every((c) => labels[c] === 0 || labels[c] === 1)) {
    return NextResponse.json({ error: "all four criteria must be 0 or 1" }, { status: 400 });
  }
  if (!["low", "medium", "high"].includes(body.confidence)) {
    return NextResponse.json({ error: "confidence required" }, { status: 400 });
  }
  const { error } = await client.from("annotations").upsert({
    annotator_id: ann.id,
    item_key: body.item_key,
    labels,
    confidence: body.confidence,
    ambiguity: body.ambiguity === true,
    note: String(body.note ?? ""),
    started_at: body.started_at ?? null,
  });
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json({ ok: true });
}
