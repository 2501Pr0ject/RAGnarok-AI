import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";

// Operator-only raw export; joins with key_map.json happen OFFLINE in
// export_external.py so configurations never live on the server.
export async function GET(req: NextRequest) {
  if (req.nextUrl.searchParams.get("key") !== process.env.ADMIN_EXPORT_KEY) {
    return NextResponse.json({ error: "forbidden" }, { status: 403 });
  }
  const client = db();
  const { data: annotators, error: annotatorsError } = await client.from("annotators").select("*");
  if (annotatorsError) {
    return NextResponse.json({ error: annotatorsError.message }, { status: 500 });
  }
  const { data: annotations, error: annotationsError } = await client.from("annotations").select("*");
  if (annotationsError) {
    return NextResponse.json({ error: annotationsError.message }, { status: 500 });
  }
  return NextResponse.json({ annotators: annotators ?? [], annotations: annotations ?? [] });
}
