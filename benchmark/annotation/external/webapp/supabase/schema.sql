-- RAGnarok annotation collection schema (run once in the Supabase SQL editor)

create table if not exists annotators (
  id text primary key,                -- pseudonymous, e.g. ANN-7F42
  token text unique not null,         -- session token in the participant link
  domain text not null,
  level text,                         -- self-reported, optional
  batch_id text not null,
  source text not null default 'direct',
  created_at timestamptz not null default now(),
  completed_at timestamptz
);

create table if not exists annotations (
  annotator_id text not null references annotators(id),
  item_key text not null,
  labels jsonb not null,              -- {retrieval_relevance,faithfulness,answer_relevance,completeness}: 0|1
  confidence text not null,           -- low | medium | high
  ambiguity boolean not null default false,
  note text not null default '',
  started_at timestamptz,
  submitted_at timestamptz not null default now(),
  primary key (annotator_id, item_key)
);

-- The site uses the service-role key server-side only; lock the tables down.
alter table annotators enable row level security;
alter table annotations enable row level security;
