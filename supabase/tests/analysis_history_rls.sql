begin;

do $$
declare
    user_a uuid;
    user_b uuid;
    marker text := gen_random_uuid()::text;
    rejected boolean;
begin
    select id into user_a
    from auth.users
    where email_confirmed_at is not null
    order by created_at
    limit 1;

    select id into user_b
    from auth.users
    where email_confirmed_at is not null
    order by created_at
    offset 1 limit 1;

    if user_a is null or user_b is null then
        raise exception 'RLS test requires two verified auth users';
    end if;

    perform set_config('test.user_a', user_a::text, true);
    perform set_config('test.user_b', user_b::text, true);
    perform set_config('test.marker', marker, true);

    if to_regclass('public.analysis_history_user_created_idx') is null
        or not exists (
            select 1
            from pg_index i
            where i.indexrelid = 'public.analysis_history_user_created_idx'::regclass
              and i.indrelid = 'public.analysis_history'::regclass
              and i.indisvalid
              and i.indnkeyatts = 2
              and pg_get_indexdef(i.indexrelid, 1, true) = 'user_id'
              and pg_get_indexdef(i.indexrelid, 2, true) = 'created_at DESC'
        ) then
        raise exception 'Expected valid (user_id, created_at DESC) history index';
    end if;

    rejected := false;
    begin
        insert into public.analysis_history (
            user_id, input_text, input_source, prediction_label, confidence
        ) values (gen_random_uuid(), marker || ':bad-fk', 'text', 'Legitimate Job', 0.5);
    exception when foreign_key_violation then
        rejected := true;
    end;
    if not rejected then
        raise exception 'Unknown user_id was not rejected by the foreign key';
    end if;

    rejected := false;
    begin
        insert into public.analysis_history (
            user_id, input_text, input_source, prediction_label, confidence
        ) values (user_a, '   ', 'text', 'Legitimate Job', 0.5);
    exception when check_violation then
        rejected := true;
    end;
    if not rejected then
        raise exception 'Blank input_text was not rejected';
    end if;

    rejected := false;
    begin
        insert into public.analysis_history (
            user_id, input_text, input_source, prediction_label, confidence
        ) values (user_a, marker || ':bad-source', 'audio', 'Legitimate Job', 0.5);
    exception when check_violation then
        rejected := true;
    end;
    if not rejected then
        raise exception 'Invalid input_source was not rejected';
    end if;

    rejected := false;
    begin
        insert into public.analysis_history (
            user_id, input_text, input_source, prediction_label, confidence
        ) values (user_a, marker || ':bad-label', 'text', 'Unknown', 0.5);
    exception when check_violation then
        rejected := true;
    end;
    if not rejected then
        raise exception 'Invalid prediction_label was not rejected';
    end if;

    rejected := false;
    begin
        insert into public.analysis_history (
            user_id, input_text, input_source, prediction_label, confidence
        ) values (user_a, marker || ':low-confidence', 'text', 'Legitimate Job', -0.01);
    exception when check_violation then
        rejected := true;
    end;
    if not rejected then
        raise exception 'Confidence below zero was not rejected';
    end if;

    rejected := false;
    begin
        insert into public.analysis_history (
            user_id, input_text, input_source, prediction_label, confidence
        ) values (user_a, marker || ':high-confidence', 'text', 'Legitimate Job', 1.01);
    exception when check_violation then
        rejected := true;
    end;
    if not rejected then
        raise exception 'Confidence above one was not rejected';
    end if;

    rejected := false;
    begin
        insert into public.analysis_history (
            user_id, input_text, input_source, prediction_label, confidence, red_flags
        ) values (
            user_a, marker || ':bad-flags', 'text', 'Legitimate Job', 0.5, '{}'::jsonb
        );
    exception when check_violation then
        rejected := true;
    end;
    if not rejected then
        raise exception 'Non-array red_flags was not rejected';
    end if;

    insert into public.analysis_history (
        user_id, input_text, input_source, prediction_label, confidence, red_flags
    ) values
        (user_a, marker || ':owned-by-a', 'text', 'Legitimate Job', 0.91, '[]'),
        (user_b, marker || ':owned-by-b', 'image', 'Potential Scam', 0.88, '["flag"]');
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
    select count(*) into visible_count
    from public.analysis_history
    where input_text like current_setting('test.marker') || ':%';
    if visible_count <> 1 then
        raise exception 'User A should see exactly one marked owned row, saw %', visible_count;
    end if;

    begin
        insert into public.analysis_history (
            user_id, input_text, input_source,
            prediction_label, confidence, red_flags
        ) values (
            current_setting('test.user_b')::uuid,
            current_setting('test.marker') || ':forged-owner',
            'text', 'Potential Scam', 0.5, '[]'
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

    rejected := false;
    begin
        insert into public.analysis_history (
            user_id, input_text, input_source, prediction_label, confidence
        ) values (
            current_setting('test.user_a')::uuid,
            current_setting('test.marker') || ':anonymous-insert',
            'text', 'Legitimate Job', 0.5
        );
    exception when insufficient_privilege then
        rejected := true;
    end;
    if not rejected then
        raise exception 'Anonymous INSERT was not rejected';
    end if;
end $$;

reset role;
rollback;
