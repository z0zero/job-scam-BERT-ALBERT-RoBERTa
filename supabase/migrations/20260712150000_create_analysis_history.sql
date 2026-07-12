create table public.analysis_history (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    input_text text not null check (length(btrim(input_text)) > 0),
    input_source text not null check (input_source in ('text', 'image')),
    prediction_label text not null check (
        prediction_label in ('Legitimate Job', 'Potential Scam')
    ),
    confidence double precision not null check (
        confidence >= 0.0 and confidence <= 1.0
    ),
    red_flags jsonb not null default '[]'::jsonb check (
        jsonb_typeof(red_flags) = 'array'
    ),
    created_at timestamptz not null default now()
);

create index analysis_history_user_created_idx
    on public.analysis_history (user_id, created_at desc);

alter table public.analysis_history enable row level security;

revoke all on table public.analysis_history from anon;
revoke all on table public.analysis_history from authenticated;
grant select, insert on table public.analysis_history to authenticated;

create policy "Users can read their own analysis history"
on public.analysis_history
for select
to authenticated
using ((select auth.uid()) = user_id);

create policy "Users can insert their own analysis history"
on public.analysis_history
for insert
to authenticated
with check ((select auth.uid()) = user_id);
