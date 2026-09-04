import { createClient } from "@supabase/supabase-js";
import { localDb } from "./localdb";

// Server-side only: the service-role key never reaches the browser.
// LOCAL_DB=1 switches to a JSON-file stand-in (chain validation / dev).
export function db() {
  if (process.env.LOCAL_DB === "1") return localDb();
  return createClient(process.env.SUPABASE_URL!, process.env.SUPABASE_SERVICE_KEY!, {
    auth: { persistSession: false },
  });
}
