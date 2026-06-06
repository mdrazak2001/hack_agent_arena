"""
://agent_arena — AppWorld starter agent (ReAct code agent).

This is a WORKING template you can hack on. The loop and every AppWorld API
call below were verified against appworld==0.1.3. Your job is to make the agent
smarter: better prompting, planning, error recovery, retrieval, etc.

How AppWorld works (the rules your agent plays by):
  - Each task gives you a natural-language instruction from your "supervisor".
  - You act by writing PYTHON code. The env runs it and returns whatever you
    print(). A preloaded object `apis` is your only interface to the 9 apps.
  - Discover APIs at runtime:
        apis.api_docs.show_app_descriptions()
        apis.api_docs.show_api_descriptions(app_name='spotify')
        apis.api_docs.show_api_doc(app_name='spotify', api_name='login')
  - Get credentials to log into apps:
        apis.supervisor.show_account_passwords()
    (most app APIs need an access_token returned by that app's `login`).
  - Finish with:
        apis.supervisor.complete_task(answer=<answer or None>)
    Pass `answer` only when the task asks a question; otherwise leave it None.

Run:
  export GROQ_API_KEY=gsk_...                 # or put it in .env
  export APPWORLD_EXPERIMENT=team_<yourname>   # your unique team id
  export APPWORLD_DATASET=dev                  # dev while building; switch to the
                                               # official split at submission time
  python agent.py
"""

import os
import re
import json
import traceback
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:  # optional: load GROQ_API_KEY etc. from a local .env
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from appworld import AppWorld, load_task_ids
import litellm
from hydradb_context import HydraContext, error_query

# ---- config ---------------------------------------------------------------
# MODEL is litellm's "provider/model" string, so you can point the agent at any
# backend by setting MODEL + the matching key in .env (see README):
#   anthropic/claude-haiku-4-5   gemini/gemini-2.0-flash   groq/llama-3.3-70b-versatile
#   openrouter/...               ollama/llama3.1 (fully local)
MODEL = os.environ.get("MODEL", "groq/llama-3.3-70b-versatile")
DATASET = os.environ.get("APPWORLD_DATASET", "dev")          # dev | test_normal | test_challenge
EXPERIMENT = os.environ.get("APPWORLD_EXPERIMENT", "team_demo")
MAX_INTERACTIONS = int(os.environ.get("MAX_INTERACTIONS", "50"))
MAX_TASKS = int(os.environ.get("MAX_TASKS", "0"))            # 0 = all tasks in split
LLM_NUM_RETRIES = int(os.environ.get("LLM_NUM_RETRIES", "8"))
TRACE_DIR = Path(os.environ.get("TRACE_DIR", "traces")) / EXPERIMENT
HYDRA = HydraContext(cache_dir=TRACE_DIR.parent if TRACE_DIR.name else Path("traces"))

APPWORLD_KWARGS = {}
if not hasattr(signal, "SIGALRM"):
    APPWORLD_KWARGS["timeout_seconds"] = None

SYSTEM_PROMPT = """You are an autonomous coding agent operating inside AppWorld.
You complete the supervisor's task by writing Python code that the environment executes.

RULES:

=== LOGIN (READ FIRST — #1 CAUSE OF FAILURE) ===
- EVERY login returns a DICT. You MUST extract the token:
    token = apis.<app>.login(username=<...>, password=<...>)["access_token"]
  NEVER write `token = apis.<app>.login(...)` and pass it directly — that passes
  the whole dict and every later call returns 401.
- Log into each app ONCE, store its token in a variable (e.g. gmail_token,
  amazon_token, phone_token), and REUSE it. Do NOT log in again every step.
- A 401 "not authorized" means your access_token is the dict or the wrong app's
  token — fix by using ["access_token"] and the RIGHT app's token. Do NOT just
  retry the same code.
- phone login username = the PHONE NUMBER (digits), all others = the EMAIL.

=== EXACT API NAMES (never invent these) ===
- When you get "No API named X", STOP guessing variants. Immediately call
  apis.api_docs.show_api_descriptions(app_name='<app>') and pick the EXACT closest
  name from the list — do NOT retry similar spellings.
- Call APIs as attributes ONLY: apis.amazon.show_wish_list(access_token=token).
  NEVER use dynamic indexing like apis.amazon[name](...) — it is invalid.

GMAIL — there is NO show_threads / show_emails / show_inbox / search_emails /
list_drafts. To list/search received mail use show_inbox_threads(query=...,
access_token=...); sent mail: show_outbox_threads. Details: show_thread(
email_thread_id=...). Drafts: show_drafts / delete_draft(draft_id=...).
Send: send_email(email_addresses=[...], subject=..., body=..., access_token=...).
Schedule: create_draft(recipient_email_addresses=[...], subject=..., body=...,
scheduled_send_at="YYYY-MM-DDTHH:MM:SS", access_token=...).

PHONE — there is NO disable_alarm. Use show_alarms, then update_alarm(
alarm_id=..., enabled=False, access_token=...) to disable, or delete_alarm.
Text: send_text_message(phone_number=..., message=..., access_token=...).
Current time: get_current_date_and_time().

SIMPLE_NOTE — there is NO show_notes. Use search_notes (list) then
show_note(note_id=...) for content.

AMAZON — removing a cart item is delete_product_from_cart(product_id=...,
access_token=...) (NOT remove_product_from_cart); clear_cart() empties it.
Also: show_wish_list show_cart add_product_to_cart move_product_from_wish_list_to_cart
place_order show_payment_cards show_addresses show_prime_subscriptions show_orders.
Wrong: show_wishlist add_to_cart remove_product_from_cart show_payment_methods.

=== PAYMENT CARDS (critical) ===
- NEVER call add_payment_card with made-up numbers. Use the user's EXISTING
  cards from show_payment_cards(access_token=...).
- Fields: payment_card_id, expiry_year, expiry_month. No is_active field.
- A card is VALID if it is not expired relative to the SIMULATED current date
  (get it from apis.phone.get_current_date_and_time — the world is in 2023, do
  NOT hardcode 2024 or use datetime.now()). Valid if expiry_year > cur_year OR
  (expiry_year == cur_year AND expiry_month >= cur_month). Pick any valid card.
- NEVER use payment_cards[0] blindly — loop ALL cards and pick the first valid one.
- If place_order says a card has expired, try the NEXT valid card.
- If place_order says insufficient balance, try the NEXT valid card.
- NEVER add_payment_card to fix expiry or balance issues.

=== NEVER STUB / NEVER GUESS RECIPIENTS ===
- NEVER leave an API call commented out or replace it with a print placeholder.
  If the task says create an expense, actually CALL apis.splitwise.create_expense
  (look up its exact params via show_api_doc first).
- NEVER fabricate email addresses like name@example.com. Get real emails from
  apis.phone.search_contacts (by name/relationship) or gmail search_users.

- Reply with EXACTLY ONE Python code block per turn, nothing else:
  ```python
  # your code
  ```

=== ANSWER RULES (MOST IMPORTANT — read carefully) ===
- complete_task(answer=...) has TWO modes:
  1. ACTION tasks (do/make/send/order/star/schedule/reply/forward/buy/initiate/
     reset/pay): the task asks you to CHANGE state, not to report a value.
     => ALWAYS call complete_task(answer=None). NEVER pass a confirmation
        sentence like "Order placed" or "Emails sent". The grader expects null;
        any string FAILS the task.
  2. QUESTION tasks (how many / how much / what is / list ...): return ONLY the
     bare value — a number like 15 or 38634.99, or the exact short string/list
     requested. NO sentences, NO units, NO "The answer is...".
     Example: if asked "how many hours", answer=15  (NOT "15 hours", NOT a sentence).
- Before calling complete_task, ask: "Is this a question, or an action?"
  If action -> answer=None. If question -> answer=<bare value only>.

=== KNOWN ENVIRONMENT FACTS (use these exact shapes; do NOT re-derive them) ===
- apis.supervisor.show_account_passwords() returns a LIST of
  {"account_name","password"}. Get one: next(p["password"] for p in pwds if
  p["account_name"]=="<app>").
- apis.supervisor.show_addresses() returns a LIST of
  {"name","street_address","city","state","country","zip_code"} where name is
  "Home" or "Work". There is NO address_id, NO type, NO label. Do NOT use this
  for amazon place_order.
- For amazon shipping: apis.amazon.show_addresses(access_token=...) returns
  {"address_id","name","street_address",...}. Pick home with name=="Home", use
  its address_id in place_order.
  NEVER use address_type, type, label, or 'home' lowercase — the field is
  name and the value is exactly "Home" (capital H):
    home = next(a for a in apis.amazon.show_addresses(access_token=t) if a["name"]=="Home")
    address_id = home["address_id"]
  Do NOT use addresses[0] blindly — verify name=="Home".
- ALL list endpoints return a PLAIN LIST. Never use .get("success")/.get("threads").
- LOGIN usernames: gmail/amazon/spotify/file_system/venmo/simple_note use the
  user's EMAIL. phone uses the PHONE NUMBER (a digit string), NOT the email.
- gmail threads: fields are "email_thread_id"(int), "email_ids"(list),
  "starred"(bool), "archived"(bool). There is NO "id"/"is_starred"/"participants".
- gmail show_thread(email_thread_id=...) returns {"emails":[{"email_id",
  "subject","sender":{"email"},"recipients":[{"email"}], ...}]}.
- gmail reply_to_email and forward_email_from_thread need BOTH email_thread_id
  AND email_id (ints). forward and download_attachment ALSO need a
  file_system_access_token.
- file_system: there is NO read_file. To read a file use show_file(file_path=...,
  access_token=...) which returns {"content": "..."}. To list use show_directory.
  Paths use the real home dir (resolve ~ by listing the directory first).
- amazon orders: fields are "created_at" and "paid_amount" (NOT order_date/
  total_amount). show_orders is sorted newest-first.
- amazon products (search_products): items have "product_id"(not id), "rating",
  "num_product_reviews"(NOT review_count), "price", "product_type", "name".
  Filter/sort with these exact keys.
- amazon payment cards: see PAYMENT CARDS (critical) above. show_payment_cards — NOT
  show_payment_methods.
- amazon place_order(payment_card_id, address_id, access_token) orders the ENTIRE
  CART. So BEFORE placing an order for "X", make sure the cart contains ONLY X:
  call show_cart, and delete_product_from_cart for anything not requested (or only
  add exactly what's asked to an otherwise-empty-relevant cart). Verify with
  show_cart before place_order.
  place_order ALWAYS needs all three: access_token, payment_card_id, address_id.
  If you get "field required", you forgot one — do NOT call place_order with only
  access_token.
- amazon show_wish_list items have product_id, product_name, quantity, price.
  To order wishlist: move_product_from_wish_list_to_cart(product_id=..., access_token=...)
  then place_order with valid card + Home address_id.
- amazon initiate_return needs order_id, product_id, quantity, deliverer_id.
  show_return_deliverers gives deliverers; FedEx is one of them (match by name).
- venmo create_transaction(receiver_email, amount, description, access_token)
  SENDS money; there's a separate request API. transaction fields: "amount",
  "description","sender","receiver".
- phone APIs need a PHONE access_token from apis.phone.login(username=<phone
  number>, password=...). Never pass an amazon/gmail token to phone APIs.
- RELATIONSHIP RECIPIENTS (husband/wife/partner/roommate/coworker/friend/manager):
  get them from apis.phone.search_contacts(relationship="<rel>", access_token=...)
  — each contact has "email" and "phone_number". Do NOT guess via gmail search_users.
  For "the REST of my roommates", exclude the original sender.

=== BEHAVIOR RULES ===
- ONE STEP PER TURN: do NOT cram login+search+order+complete_task into one block.
  Run one small action, print/inspect the result, then proceed on the next turn.
- If you get "No API named X", call show_api_descriptions immediately — do NOT
  guess another variant of X (see EXACT API NAMES above).
- If you get KeyError on a field name, STOP guessing alternates (id/type/label/
  is_active/address_type/review_count/emails). Print list(item.keys()) from the
  last API response, then use those exact keys.
- If place_order says "payment card has expired" or "insufficient balance", try
  the next valid card from show_payment_cards — never add_payment_card.
- NEVER stub: if an action is required, call the real API — no commented-out
  calls and no print("would send...") placeholders.
- IDEMPOTENT ACTIONS: If an API returns 422 "already starred/unstarred/liked/
  downloaded/friends/returned", treat it as SUCCESS and CONTINUE. Never loop
  retrying the same call. Wrap such calls in try/except and ignore that message.
- COWORKER/RELATION FILTERING for emails: identify the people first (phone
  contacts by relationship), collect their emails, THEN match threads by
  sender/recipient email against that set.

=== RESPONSE SHAPE RULES ===
- Most AppWorld list endpoints return a PLAIN LIST, not a dict. Do NOT call
  response.get("success") or response.get("threads"). Iterate the list directly:
  results = apis.<app>.<list_api>(...); for item in results: ...
- Pagination: page_limit MAXIMUM is 20. Start with page_limit=20, page_index=0,
  and increment page_index until a page returns fewer than 20 items. Never pass
  page_limit > 20.

=== VERIFY BEFORE COMPLETING ===
- Re-read the task and act on EXACTLY what it specifies:
  * "all weightlifting benches in my cart" = ONLY those items, not the whole cart.
  * "two same-colored T-shirts" = quantity 2 of ONE product, check inventory.
  * recipients/contacts: look up REAL email/phone from apis.phone.search_contacts
    (by relationship or name) — never invent placeholders like "husband@example.com".
- After doing the action, do a quick read-back to confirm the intended state,
  THEN call complete_task(answer=None) for action tasks.

- A preloaded object `apis` is the ONLY way to interact with the apps. Whatever
  you print() is returned to you as the next observation.
- You do NOT know the APIs in advance. Discover them at runtime:
    print(apis.api_docs.show_app_descriptions())
    print(apis.api_docs.show_api_descriptions(app_name='<app>'))
    print(apis.api_docs.show_api_doc(app_name='<app>', api_name='<api>'))
- To act on the supervisor's accounts, get credentials once with
  show_account_passwords(), log in once per app, extract ["access_token"], reuse.
- Work in small steps: inspect results before the next action. Never invent API
  names or fields — look them up first.
- First read the relevant api_docs. Use show_api_descriptions(app_name=...) for
  apps you need, then show_api_doc only for the specific APIs you intend to call.
- Variables persist across turns within a task, so save tokens/passwords/docs to
  plain variables and reuse them — but always define a helper in the same turn
  you call it.
- Do not call show_api_doc for every API in an app. Pull the exact API doc only
  after choosing the API from show_api_descriptions.
- Fetch supervisor passwords once, then log into only the apps needed for the task.
- Execute exactly one step at a time, and verify the result before proceeding.

DATA-COMPLETENESS RULES (critical for correctness):
- PAGINATION: Any API with page_index/page_limit returns ONE page only. Use
  page_limit=20 (the maximum), page_index=0,1,2,... and keep going until a page
  returns fewer than 20 items (or empty). NEVER assume page 0 has everything.
- RANKINGS ("most played", "top", "highest", "popular", "best"): Do NOT guess
  from order or frequency. Find the actual numeric field (e.g. play_count,
  rating, like_count) by reading the item's full details (e.g. show_song for each
  song). Sort by that exact field, then take the requested count.
  For "most played / top / popular <genre> songs in my library":
  (a) Page through show_song_library FULLY (page_index 0,1,2,... until a short
      page). List endpoints do NOT include genre or play_count.
  (b) For EACH song_id, call show_song(song_id) to get genre AND play_count.
  (c) Filter to the requested genre, sort by play_count DESCENDING.
  (d) Return EXACTLY the requested number of titles, in that order.
  NEVER rank by frequency-across-libraries, added_at, or list order.
- DERIVED FIELDS (genre, etc.): If a field isn't in a list response, fetch it
  from the item's detail API (show_song gives a song's genre and play_count) or
  via its album. Don't infer it.
- COUNTS: Re-read the task for the EXACT number requested (top 4 vs top 6) and
  the EXACT ordering. Return precisely that many, in that order.
- SANITY CHECK BEFORE complete_task: If your computed answer is empty, or has
  fewer items than the task asked for, STOP and investigate (you probably missed
  pagination, a detail lookup, or a filter). Never submit an empty or obviously
  incomplete answer.
- For challenge tasks, prefer fetching per-item details over inferring from list
  summaries. If an answer looks empty/too short, you missed pagination or a
  detail lookup — investigate.

PLANNING:
- For any "find/rank/filter" task, first write the full plan as comments:
  (1) what complete data do I need, (2) which detail lookups give the ranking
  field, (3) how do I sort and cut to the requested count. Then execute it.

- Never touch unrelated user data or app records.
- Before every action, ask: did the task ask me to do this? If no, skip it.
- Never delete, overwrite, send, or post anything unless the instruction explicitly requires it.
- If you are unsure whether an action is required, do NOT take it.
- When and ONLY when the task is fully done, call complete_task per ANSWER RULES:
    apis.supervisor.complete_task(answer=None)          # action tasks
    apis.supervisor.complete_task(answer=<bare value>)  # question tasks only
"""


def call_llm(messages: list[dict]) -> str:
    model = MODEL
    kwargs = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        "max_tokens": 2500,
        "temperature": 0.0,
        "num_retries": LLM_NUM_RETRIES,   # ride out free-tier rate limits (429) with backoff
    }
    if model.startswith("groq/"):
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required for MODEL=groq/...")
        if groq_api_key.startswith("sk-or-"):
            raise RuntimeError(
                "GROQ_API_KEY looks like an OpenRouter key (sk-or-...). "
                "Use OPENROUTER_API_KEY + MODEL=openrouter/<model> instead, "
                "or put a Groq key (gsk_...) from console.groq.com in GROQ_API_KEY."
            )
        kwargs.update({
            "model": model.split("/", 1)[1],
            "custom_llm_provider": "groq",
            "api_key": groq_api_key,
        })
    elif model.startswith("openrouter/") or (
        os.environ.get("OPENROUTER_API_KEY")
        and not model.startswith(("groq/", "openai/", "anthropic/", "gemini/", "ollama/"))
    ):
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required. Set MODEL=openrouter/<slug> "
                "(e.g. openrouter/meta-llama/llama-3.3-70b-instruct:free)."
            )
        if not model.startswith("openrouter/"):
            model = f"openrouter/{model}"
        kwargs["model"] = model
        kwargs["api_key"] = openrouter_api_key
    resp = litellm.completion(**kwargs)
    return resp.choices[0].message.content or ""


def resolve_task_ids(dataset_name: str) -> list[str]:
    """Load task IDs for a dataset split, with a direct file fallback."""
    try:
        return load_task_ids(dataset_name)
    except Exception:
        pass
    dataset_file = Path("data") / "datasets" / f"{dataset_name}.txt"
    if not dataset_file.is_file():
        try:
            from appworld.common.path_store import path_store
            dataset_file = Path(path_store.data) / "datasets" / f"{dataset_name}.txt"
        except Exception:
            pass
    if not dataset_file.is_file():
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found via load_task_ids or {dataset_file}"
        )
    task_ids = [
        line.strip().split(":")[0]
        for line in dataset_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not task_ids:
        raise ValueError(f"No task IDs in {dataset_file}")
    return task_ids


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()


def format_supervisor(supervisor: dict) -> str:
    email = supervisor.get("email", "")
    phone = supervisor.get("phone_number", "")
    name = f"{supervisor.get('first_name', '')} {supervisor.get('last_name', '')}".strip()
    return (
        f"Name: {name}\n"
        f"Email (use as username for app logins): {email}\n"
        f"Phone (use as username for phone app login, digits only): {phone}"
    )


def restore_safety_guard(world: Optional[AppWorld] = None) -> None:
    """AppWorld patches builtins.open globally; ensure it is restored between tasks."""
    try:
        if world is not None:
            world.safety_guard.disable()
    except Exception:
        pass
    try:
        from appworld.common.safety_guard import SafetyGuard
        SafetyGuard().disable()
    except Exception:
        pass


def preview_output(output: str, limit: int = 120) -> str:
    if "Execution failed" in output or "Execution raised an exception" in output:
        lines = [ln for ln in output.splitlines() if ln.strip()]
        if lines:
            return lines[-1][:400]
    return output[:limit]


def safe_task_id(world: AppWorld) -> str:
    task_id = getattr(world, "task_id", None) or getattr(world.task, "id", None)
    if not task_id:
        task_id = re.sub(r"\W+", "_", world.task.instruction[:60]).strip("_") or "task"
    return str(task_id)


def new_trace(world: AppWorld) -> dict:
    return {
        "task_id": safe_task_id(world),
        "dataset": DATASET,
        "experiment": EXPERIMENT,
        "model": MODEL,
        "max_interactions": MAX_INTERACTIONS,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "supervisor": world.task.supervisor,
        "instruction": world.task.instruction,
        "steps": [],
        "final_status": "running",
    }


def write_trace(trace: dict) -> None:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", trace["task_id"]).strip("_") or "task"
    path = TRACE_DIR / f"{safe_name}.json"
    trace["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(trace, indent=2), encoding="utf-8")


def execute_action(world: AppWorld, code: str) -> tuple[str, Optional[str]]:
    try:
        return str(world.execute(code)), None
    except Exception:
        error = traceback.format_exc()
        try:
            world.safety_guard.disable()
        except Exception:
            pass
        observation = f"Execution raised an exception. Full traceback:\n{error}"
        return observation, error


def check_task_completed(world: AppWorld) -> tuple[bool, Optional[str]]:
    try:
        return world.task_completed(), None
    except Exception:
        return False, traceback.format_exc()


def solve(world: AppWorld) -> None:
    trace = new_trace(world)
    write_trace(trace)
    task_id = trace["task_id"]
    instruction = world.task.instruction
    initial_hydra = HYDRA.retrieve(
        f"AppWorld agent task: {instruction}\n"
        "login access_token API names amazon gmail phone place_order"
    )
    user_content = (
        f"{format_supervisor(world.task.supervisor)}\n\n"
        f"Task: {instruction}\n\n"
    )
    if initial_hydra:
        user_content += HYDRA.format_block(initial_hydra, "HYDRADB CONTEXT (retrieved hints)") + "\n"
    user_content += (
        "Begin. Remember: one python code block per turn. "
        "Extract login tokens with [\"access_token\"]; phone login uses the phone number."
    )
    messages = [{"role": "user", "content": user_content}]
    for step in range(MAX_INTERACTIONS):
        if step > 0 and HYDRA.enabled and trace["steps"]:
            last = trace["steps"][-1]
            obs = str(last.get("observation") or "")
            failed = (
                last.get("execution_error")
                or "Execution failed" in obs
                or "Exception:" in obs
                or "401" in obs
                or "422" in obs
                or "KeyError" in obs
            )
            if failed:
                hint = HYDRA.retrieve(error_query(instruction, obs))
                if hint:
                    messages.append({
                        "role": "user",
                        "content": HYDRA.format_block(
                            hint,
                            "HYDRADB CONTEXT (relevant to last error — apply these patterns)",
                        ),
                    })
        try:
            reply = call_llm(messages)
        except Exception:
            error = traceback.format_exc()
            print(f"  ! LLM call failed on step {step+1}; moving to next task")
            trace["steps"].append({
                "step": step + 1,
                "reply": None,
                "code": None,
                "observation": None,
                "llm_error": error,
                "completed": False,
            })
            trace["final_status"] = "llm_error"
            write_trace(trace)
            restore_safety_guard(world)
            return
        code = extract_code(reply)
        output, execution_error = execute_action(world, code)
        print(f"  step {step+1}: ran {len(code)} chars -> {preview_output(str(output))!r}")
        if execution_error:
            print(f"  step {step+1}: execution raised; fed traceback back")
        completed, completion_error = check_task_completed(world)
        if completion_error:
            output = f"{output}\n\nTask completion check raised an exception:\n{completion_error}"
        trace["steps"].append({
            "step": step + 1,
            "reply": reply,
            "code": code,
            "observation": str(output),
            "execution_error": execution_error,
            "completion_error": completion_error,
            "completed": completed,
        })
        HYDRA.remember_step(
            task_id=task_id,
            step=step + 1,
            instruction=instruction,
            code=code,
            observation=str(output),
            succeeded=completed and not execution_error,
        )
        write_trace(trace)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Execution output:\n{output}"})
        if completed:
            trace["final_status"] = "completed"
            write_trace(trace)
            print("  ✓ task_completed")
            restore_safety_guard(world)
            return
    print("  ✗ hit MAX_INTERACTIONS without completion")
    trace["final_status"] = "max_interactions"
    write_trace(trace)
    restore_safety_guard(world)


def main() -> None:
    HYDRA.bootstrap()
    task_ids = resolve_task_ids(DATASET)
    if MAX_TASKS:
        task_ids = task_ids[:MAX_TASKS]
    print(f"Running '{EXPERIMENT}' on {len(task_ids)} '{DATASET}' tasks with {MODEL}")
    for i, task_id in enumerate(task_ids, 1):
        print(f"[{i}/{len(task_ids)}] {task_id}")
        restore_safety_guard()
        world: Optional[AppWorld] = None
        try:
            world = AppWorld(task_id=task_id, experiment_name=EXPERIMENT, **APPWORLD_KWARGS)
            with world:
                solve(world)
        except Exception as e:  # never let one task kill the whole run
            print(f"  ! error: {e}")
            print(traceback.format_exc())
        finally:
            restore_safety_guard(world)
    print(f"\nDone. Outputs in ./experiments/outputs/{EXPERIMENT}/")
    print("Hand that folder to the organizers (or zip and submit per instructions).")


if __name__ == "__main__":
    main()
