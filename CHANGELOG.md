# Changelog

## Pre-loop (staged v10 prompt fixes)

- **CASING / SPLITWISE / MEETING / CHECKLIST playbooks** in `agent.py` + Hydra seed v10
  - Lowercase `account_name` lookups (`simple_note` not `SimpleNote`)
  - Splitwise cable: `show_groups_balance` → `Roommates` group; month from subject not enumerate
  - Meeting drafts: match note title; parse `Attendees:` block; `phone.search_contacts` + `first_name`
  - Husband checklist: regex parse attachment; never hardcode quantities after download
- **Why**: trace post-mortems on eval failures (general patterns, no task_id branches)

## baseline-prep

- **Windows UTF-8 fix**: replaced Unicode checkmarks in prints; `sys.stdout.reconfigure(utf-8)`
- **EXEC_HELPERS preamble**: `account_password`, `parse_simulated_today`, `valid_payment_cards`,
  `parse_checklist_lines`, `splitwise_roommates_group`, `contact_email_by_first_name` injected
  before every `world.execute()` (general, not task-specific)
- **Tooling**: `scripts/analyze_traces.py`, `scripts/check_no_task_ids.py`, SCOREBOARD/CHANGELOG/FAILURES

## baseline

- Pending full eval run (10 tasks, submission model).
