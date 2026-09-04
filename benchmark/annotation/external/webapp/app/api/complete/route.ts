import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { loadBatch } from "@/lib/batches";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const client = db();
  const { data: ann } = await client.from("annotators").select("*").eq("token", body.token ?? "").single();
  if (!ann) return NextResponse.json({ error: "unknown token" }, { status: 403 });

  const batch = loadBatch(ann.batch_id);
  const { data: done } = await client.from("annotations").select("item_key").eq("annotator_id", ann.id);
  const remaining = batch.items.length - (done?.length ?? 0);
  if (remaining > 0) return NextResponse.json({ error: `${remaining} case(s) remaining` }, { status: 400 });

  await client.from("annotators").update({ completed_at: new Date().toISOString() }).eq("id", ann.id);
  return NextResponse.json({ ok: true, annotator_id: ann.id, cases: batch.items.length });
}
