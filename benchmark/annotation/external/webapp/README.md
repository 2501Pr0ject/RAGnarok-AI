# Annotation collection webapp

Next.js app collecting external human annotations for the RAGnarok
evaluation study. It is a **collection tool only**: the scientific
source of truth is the repository (batches, schemas, protocol). The
server never holds the item→configuration mapping (`key_map.json`
stays offline with the operator).

## Deploy (operator)

1. **Supabase**: create a project, run `supabase/schema.sql` in the SQL
   editor. Note the project URL and the service-role key.
2. **Vercel**: import the repository, set the *Root Directory* to
   `benchmark/annotation/external/webapp`, and add env vars:
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_KEY` (service role — server-side only)
   - `ADMIN_EXPORT_KEY` (any long random string)
3. Build runs `npm run sync-data`, which copies `../batches` into the
   deployment. Re-deploy after any batch regeneration (and re-run
   `check_blindness.py` first).

## Local dev

```bash
npm install
SUPABASE_URL=... SUPABASE_SERVICE_KEY=... ADMIN_EXPORT_KEY=dev npm run dev
```

## Flows

- `/` — qualification (domain + optional level) → creates `ANN-XXXX` +
  session token, assigns the least-assigned open batch for the domain.
  Query params: `?src=reddit|discord|direct` (recruitment channel),
  `?smoke=1` (internal smoke-test batches).
- `/annotate?token=…` — case-by-case annotation; per-case saves
  (resumable); completion marks the annotator done.
- `/api/export?key=$ADMIN_EXPORT_KEY` — operator-only raw dump, to feed
  `../export_external.py`.

## What the browser never receives

Configuration names/letters, question types, expected behaviors,
timings, model or run parameters, `case_id`s. Batches are checked by
`../check_blindness.py` before publication.
