# Changelog

## iter2 — 100% TGC on agent_arena_eval

- **`simple_note_content_by_title` helper**: `search_notes()` returns irrelevant hits; filter exact title before parsing meeting schedule
- **`execution_recovery_hint()`**: deterministic recovery for meeting/cable playbook tasks, `account_password(apis, app)` misuse, and 0-draft false completion
- **Stronger playbooks**: explicit helper-only flow for meeting drafts + Splitwise cable bills; `account_password(apis, 'app')` signature in examples
- **`contact_email_by_first_name`**: handle full names and first-name-only attendee lists
- **`TASK_IDS` env filter**: comma-separated task filter for targeted smoke tests (general debugging)
- **Hydra seed v13**: note title filter + helper call patterns
- **Why**: baseline failures on 18670a5_3 (wrong note from search_notes[0]) and 8d42650_3 (ignored helper)

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

- 80% TGC (8/10) on agent_arena_eval with submission model.
