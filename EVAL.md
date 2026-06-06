# 🏁 Final Evaluation Set — `agent_arena_eval` (20 tasks)

This is the **official set you'll be ranked on.** It's a fixed, stratified random
subset of AppWorld `test_challenge` — **6 easy / 7 medium / 7 hard**, 20 distinct
scenarios. Same set for every team.

**Integrity:** this set was committed at the fireside chat. Verify it's unchanged —
the SHA-256 of these 20 IDs must equal the hash announced there:
```
c8d2443301ceb9ad1d6fcc3d3004523793db449eb5d62086acee55b0953cb314
```

## 1. Install the set
After `git pull`, copy it into your AppWorld data dir:
```bash
mkdir -p data/datasets
cp eval/agent_arena_eval.txt data/datasets/agent_arena_eval.txt
```
(Not pulling? Create `data/datasets/agent_arena_eval.txt` with the 20 IDs at the bottom of this file.)

## 2. Run your agent on it
```bash
source .venv/bin/activate
export APPWORLD_EXPERIMENT=team_<yourname>     # your unique team id
export APPWORLD_DATASET=agent_arena_eval MAX_TASKS=0
python agent.py
```
⏱️ Budget ~30–45 min — these are the hard challenge tasks.

## 3. Check your own score
```bash
appworld evaluate $APPWORLD_EXPERIMENT agent_arena_eval
```
Primary metric = **TGC** (Task Goal Completion). SGC breaks ties.

## 4. Submit before the deadline
Zip and send your output folder:
```
experiments/outputs/team_<yourname>/
```
It must contain `evaluations/agent_arena_eval.json` and the `tasks/<id>/dbs/` folders.

> Rules: build a **general** agent. No hardcoding answers to specific `task_id`s —
> submitted code is reviewed and such entries are disqualified.

---
### The 20 task IDs
```
eb5ad85_1
ba46d91_3
5238afc_1
6a5e690_3
dbc0276_3
98d2608_3
258796c_2
0de03ad_2
b0934aa_3
1c4bd27_3
4242c97_1
96bf160_3
ce73d68_1
cdaaea5_3
fa327a6_3
f6be291_1
e70b117_2
4441ee9_2
69ba40f_1
adb1060_2
```
