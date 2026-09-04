import fs from "fs";
import path from "path";

// Batches are synced from ../batches by `npm run sync-data` and shipped
// with the deployment. They contain only annotator-visible fields
// (enforced by check_blindness.py before publication).

const DATA = path.join(process.cwd(), "data");

export type Item = {
  item_key: string;
  case_id: string;
  domain: string;
  question: string;
  reference_information: string;
  reference_chunks: string[];
  retrieved_context: { id: string; text: string }[];
  answer: string;
};

export type Batch = { batch_id: string; domain: string; smoke: boolean; study_version: string; items: Item[] };

export function manifest(): { batches: Record<string, number> } {
  return JSON.parse(fs.readFileSync(path.join(DATA, "batches_manifest.json"), "utf8"));
}

export function loadBatch(batchId: string): Batch {
  const domain = batchId.startsWith("smoke") ? "smoke" : batchId.split("-")[0];
  return JSON.parse(fs.readFileSync(path.join(DATA, "batches", domain, `${batchId}.json`), "utf8"));
}

export function batchIdsForDomain(domain: string, smoke: boolean): string[] {
  return Object.keys(manifest().batches)
    .filter((b) => (smoke ? b.startsWith("smoke") : b.startsWith(`${domain}-`)))
    .sort();
}

/** Browser-safe projection: never send case_id (raises repo-lookup cost). */
export function toClient(item: Item) {
  const { case_id, ...rest } = item;
  return rest;
}
