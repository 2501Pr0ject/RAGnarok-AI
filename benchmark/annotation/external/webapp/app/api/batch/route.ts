import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { loadBatch, toClient } from "@/lib/batches";

export async function GET(req: NextRequest) {
  const token = req.nextUrl.searchParams.get("token") ?? "";
  const client = db();
  const { data: ann } = await client.from("annotators").select("*").eq("token", token).single();
  if (!ann) return NextResponse.json({ error: "unknown token" }, { status: 403 });

  const batch = loadBatch(ann.batch_id);
  const { data: done } = await client
    .from("annotations")
    .select("item_key")
    .eq("annotator_id", ann.id);
  return NextResponse.json({
    annotator_id: ann.id,
    batch_id: batch.batch_id,
    items: batch.items.map(toClient),
    done: (done ?? []).map((d) => d.item_key),
  });
}
