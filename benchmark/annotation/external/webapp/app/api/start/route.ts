import { NextRequest, NextResponse } from "next/server";
import { randomBytes } from "crypto";
import { db } from "@/lib/db";
import { batchIdsForDomain } from "@/lib/batches";

const DOMAINS = ["docker", "python", "fastapi", "kubernetes"];
const LEVELS = ["beginner", "intermediate", "advanced", "expert", "undisclosed"];
const SOURCES = ["direct", "reddit", "discord", "other", "internal"];

export async function POST(req: NextRequest) {
  const body = await req.json();
  const domain = String(body.domain || "");
  const level = LEVELS.includes(body.level) ? body.level : "undisclosed";
  const source = SOURCES.includes(body.source) ? body.source : "direct";
  const smoke = body.smoke === true;
  if (!smoke && !DOMAINS.includes(domain)) {
    return NextResponse.json({ error: "invalid domain" }, { status: 400 });
  }

  const client = db();
  // Quota: assign the batch with the fewest annotators so far.
  const ids = batchIdsForDomain(domain, smoke);
  const { data: counts } = await client.from("annotators").select("batch_id");
  const tally: Record<string, number> = Object.fromEntries(ids.map((b) => [b, 0]));
  for (const row of counts ?? []) if (row.batch_id in tally) tally[row.batch_id]++;
  const batchId = ids.sort((a, b) => tally[a] - tally[b])[0];

  const id = "ANN-" + randomBytes(3).toString("hex").toUpperCase();
  const token = randomBytes(16).toString("hex");
  const { error } = await client
    .from("annotators")
    .insert({ id, token, domain: smoke ? "mixed" : domain, level, batch_id: batchId, source });
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json({ annotator_id: id, token, batch_id: batchId });
}
