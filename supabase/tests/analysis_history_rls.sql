begin;

do $$
declare
    user_a uuid;
    user_b uuid;
begin
    select id into user_a from auth.users order by created_at limit 1;
    select id into user_b from auth.users order by created_at offset 1 limit 1;
    if user_a is null or user_b is null then
        raise exception 'RLS test requires two verified auth users';
    end if;
    perform set_config('test.user_a', user_a::text, true);
    perform set_config('test.user_b', user_b::text, true);
    insert into public.analysis_history (
        user_id, input_text, input_source, prediction_label, confidence, red_flags
    ) values
        (user_a, 'owned by A', 'text', 'Legitimate Job', 0.91, '[]'),
        (user_b, 'owned by B', 'text', 'Potential Scam', 0.88, '["flag"]');
end $$;

set local role authenticated;
select set_config(
    'request.jwt.claims',
    json_build_object(
        'sub', current_setting('test.user_a'),
        'role', 'authenticated'
    )::text,
    true
);

do $$
declare
    visible_count integer;
    rejected boolean := false;
begin
    select count(*) into visible_count from public.analysis_history;
    if visible_count <> 1 then
        raise exception 'User A should see exactly one owned row, saw %', visible_count;
    end if;

    begin
        insert into public.analysis_history (
            user_id, input_text, input_source,
            prediction_label, confidence, red_flags
        ) values (
            current_setting('test.user_b')::uuid,
            'forged owner', 'text', 'Potential Scam', 0.5, '[]'
        );
    exception when insufficient_privilege then
        rejected := true;
    end;
    if not rejected then
        raise exception 'Cross-user insert was not rejected';
    end if;

    rejected := false;
    begin
        update public.analysis_history set confidence = 0.1;
    exception when insufficient_privilege then
        rejected := true;
    end;
    if not rejected then
        raise exception 'UPDATE was not rejected';
    end if;

    rejected := false;
    begin
        delete from public.analysis_history;
    exception when insufficient_privilege then
        rejected := true;
    end;
    if not rejected then
        raise exception 'DELETE was not rejected';
    end if;
end $$;

reset role;
set local role anon;
select set_config(
    'request.jwt.claims',
    json_build_object('role', 'anon')::text,
    true
);

do $$
declare
    rejected boolean := false;
begin
    begin
        perform count(*) from public.analysis_history;
    exception when insufficient_privilege then
        rejected := true;
    end;
    if not rejected then
        raise exception 'Anonymous SELECT was not rejected';
    end if;
end $$;

reset role;
rollback;
