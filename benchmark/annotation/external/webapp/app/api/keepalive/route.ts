import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";

// Pinged daily by the Vercel cron (see vercel.json): free-tier Supabase
// pauses projects after ~7 days without activity, which silently takes
// the whole collection site down. One lightweight read per day keeps
// the project active; the response doubles as a health check.
export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const secret = process.env.CRON_SECRET;
  if (secret && req.headers.get("authorization") !== `Bearer ${secret}`) {
    return NextResponse.json({ error: "forbidden" }, { status: 403 });
  }
  const { error } = await db().from("annotators").select("id").limit(1);
  if (error) {
    return NextResponse.json({ ok: false, error: error.message }, { status: 500 });
  }
  return NextResponse.json({ ok: true });
}
