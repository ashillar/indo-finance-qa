create table if not exists public.reviews (
  filename   text   not null,
  model_key  text   not null,
  qa_index   int    not null,
  data       jsonb  not null default '{}'::jsonb,
  primary key (filename, model_key, qa_index)
);

-- The app uses the anon key, so allow it to read/write this table.
alter table public.reviews enable row level security;
create policy "app read"   on public.reviews for select using (true);
create policy "app insert" on public.reviews for insert with check (true);
create policy "app update" on public.reviews for update using (true) with check (true);
