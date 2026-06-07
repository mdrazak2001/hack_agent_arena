─────────────────────────────────────── Overall Stats ───────────────────────────────────────
Num Passed Tests : 10
Num Failed Tests : 0
Num Total  Tests : 10
────────────────────────────────────────── Passes ───────────────────────────────────────────
>> Passed Requirement
assert answers match.
>> Passed Requirement
assert model changes match splitwise.Expense, splitwise.ExpenseShare, splitwise.Notification,
ignoring file_system.Directory, file_system.File.
>> Passed Requirement
obtain added, updated, removed splitwise.Expense records using models.changed_records,
and assert 0 were updated or removed.
>> Passed Requirement
assert all added expenses have group_id matching private_data.group_id.
>> Passed Requirement
assert all added expenses have payer_id of private_data.splitwise_user_id.
>> Passed Requirement
assert expense_description_to_amount obtained from added expenses
match the private_data.expense_description_to_amount.
>> Passed Requirement
obtain added, updated, removed splitwise.ExpenseShare records using models.changed_records,
and assert 0 were updated or removed.
>> Passed Requirement
assert all added expense shares have expense_id matching the added expenses.
>> Passed Requirement
assert member_ids from added expense shares from all unique expenses match
private_data.member_ids.
>> Passed Requirement
assert user_id_to_debt_amounts from added expense shares match
private_data.user_id_to_debt_amounts.
─────────────────────────────────────────── Fails ───────────────────────────────────────────
None