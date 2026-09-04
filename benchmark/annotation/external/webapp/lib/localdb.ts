// Local JSON-file database used ONLY when LOCAL_DB=1 (chain validation
// and local development without a Supabase project). Implements the
// minimal subset of the supabase-js query surface this app uses.
// Never enabled in production.
import fs from "fs";
import path from "path";

const FILE = path.join(process.cwd(), "data", "localdb.json");

type Row = Record<string, any>;
type Store = { annotators: Row[]; annotations: Row[] };

function load(): Store {
  try {
    return JSON.parse(fs.readFileSync(FILE, "utf8"));
  } catch {
    return { annotators: [], annotations: [] };
  }
}
function save(s: Store) {
  fs.mkdirSync(path.dirname(FILE), { recursive: true });
  fs.writeFileSync(FILE, JSON.stringify(s, null, 2));
}

class Query {
  private filters: [string, any][] = [];
  constructor(private table: keyof Store, private columns: string) {}

  eq(col: string, val: any) {
    this.filters.push([col, val]);
    return this;
  }
  private rows(): Row[] {
    return load()[this.table].filter((r) => this.filters.every(([c, v]) => r[c] === v));
  }
  then(resolve: (v: { data: Row[]; error: null }) => void) {
    resolve({ data: this.rows(), error: null });
  }
  async single() {
    const rows = this.rows();
    return { data: rows[0] ?? null, error: rows.length ? null : { message: "not found" } };
  }
}

class Table {
  constructor(private table: keyof Store) {}

  select(columns = "*") {
    return new Query(this.table, columns);
  }
  async insert(row: Row) {
    const s = load();
    s[this.table].push(row);
    save(s);
    return { error: null };
  }
  async upsert(row: Row) {
    const s = load();
    const keyOf = (r: Row) => (this.table === "annotations" ? `${r.annotator_id}:${r.item_key}` : r.id);
    const idx = s[this.table].findIndex((r) => keyOf(r) === keyOf(row));
    const withTs = { submitted_at: new Date().toISOString(), ...row };
    if (idx >= 0) s[this.table][idx] = { ...s[this.table][idx], ...withTs };
    else s[this.table].push(withTs);
    save(s);
    return { error: null };
  }
  update(patch: Row) {
    const table = this.table;
    return {
      async eq(col: string, val: any) {
        const s = load();
        for (const r of s[table]) if (r[col] === val) Object.assign(r, patch);
        save(s);
        return { error: null };
      },
    };
  }
}

export function localDb() {
  return { from: (t: keyof Store) => new Table(t) } as any;
}
