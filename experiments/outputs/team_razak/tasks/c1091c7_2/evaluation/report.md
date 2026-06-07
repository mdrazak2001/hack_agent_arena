─────────────────────────────────────── Overall Stats ───────────────────────────────────────
Num Passed Tests : 4
Num Failed Tests : 0
Num Total  Tests : 4
────────────────────────────────────────── Passes ───────────────────────────────────────────
>> Passed Requirement
assert answers match.
>> Passed Requirement
assert model changes match phone.Alarm.
>> Passed Requirement
obtain updated, removed phone.Alarm records using models.changed_records,
and assert 0 alarms were added or removed.
>> Passed Requirement
assert set of updated alarm ids match exactly to private_data.to_disable_alarm_ids (ignore
order).
─────────────────────────────────────────── Fails ───────────────────────────────────────────
None