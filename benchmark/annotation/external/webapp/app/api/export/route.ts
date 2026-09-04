import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";

// Operator-only raw export; joins with key_map.json happen OFFLINE in
// export_external.py so configurations never live on the server.
export async function GET(req: NextRequest) {
  if (req.nextUrl.searchParams.get("key") !== process.env.ADMIN_EXPORT_KEY) {
    return NextResponse.json({ error: "forbidden" }, { status: 403 });
  }
  const client = db();
  const { data: annotators } = await client.from("annotators").select("*");
  const { data: annotations } = await client.from("annotations").select("*");
  return NextResponse.json({ annotators, annotations });
}
