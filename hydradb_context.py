"""HydraDB v2 integration — retrieve AppWorld hints and store episodic memories."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

try:
    from hydra_db import HydraDB
    from hydra_db.helpers import build_string
except ImportError:
    HydraDB = None  # type: ignore[misc, assignment]
    build_string = None  # type: ignore[misc, assignment]

# Condensed playbook seeded once as knowledge (mirrors critical SYSTEM_PROMPT facts).
APPWORLD_KNOWLEDGE_SEED = """AppWorld agent playbook — use these exact patterns.

LOGIN: token = apis.<app>.login(username=..., password=...)["access_token"]
Never pass the login dict as access_token (causes 401). Phone login username = phone digits; others = email.
Log in once per app and reuse tokens.
Passwords: account_name always lowercase (simple_note not SimpleNote).

API NAMES (exact): gmail.show_inbox_threads show_outbox_threads show_thread show_drafts
delete_draft send_email create_draft. NO show_threads show_emails show_inbox list_drafts.
amazon.show_wish_list show_cart add_product_to_cart delete_product_from_cart (NOT remove_product_from_cart)
clear_cart place_order show_payment_cards show_addresses.
phone.show_alarms update_alarm(enabled=False) send_text_message get_current_date_and_time.
NO disable_alarm. simple_note.search_notes show_note (NOT show_notes).
Wrong: list_drafts show_wishlist add_to_cart remove_product_from_cart show_payment_methods.

ADDRESSES: amazon.show_addresses has name "Home"/"Work" and address_id. Use name=="Home" for shipping.
NEVER address_type type label or lowercase home. supervisor.show_addresses has NO address_id.

PAYMENT CARDS: show_payment_cards fields payment_card_id expiry_year expiry_month. No is_active field.
get_current_date_and_time returns {"date":"Thursday, May 18, 2023","time":"..."} — NO year/month keys.
Parse: today=datetime.datetime.strptime(now["date"], "%A, %B %d, %Y"); cur_year,cur_month=today.year,today.month
Valid if expiry_year > cur_year OR (expiry_year==cur_year AND expiry_month>=cur_month). On expired/balance try next card.
Never add_payment_card with fake numbers. Never stub actions with print placeholders.

PRIME SUBSCRIPTIONS: show_prime_subscriptions fields prime_subscription_id start_date end_date
payment_card_digits paid_amount. NO subscription_type NO expiration_date.
end_date format "2024-03-18T23:59:59". Months left: parse end_date with "%Y-%m-%dT%H:%M:%S", subtract today, round(days/30).

AMAZON CART: show_cart returns DICT with "cart_items" list — NOT a plain list. Use cart["cart_items"].
clear_cart() to empty. add_product_to_cart(product_id=..., quantity=N, ...). delete_product_from_cart NOT remove_product_from_cart.
search_products includes inventory_quantity. "For each roommate" with stock=1: loop clear_cart add qty=1 place_order per person.

GMAIL DRAFTS: delete where subject=="" OR body=="". Re-fetch show_drafts() before each delete pass.
409/draft does not exist = success (idempotent). Paginate page_limit=20.

GMAIL ATTACHMENTS: email attachments use {"id", "file_name"} — read id as attachment_id param.
download_attachment(attachment_id=att["id"], access_token=gmail_token, file_system_access_token=fs_token).
Then show_file on returned file_path. Find emails via show_inbox_threads + show_thread.

PHONE ALARMS: fields alarm_id time label enabled repeat_days — NO description. Disable via update_alarm.
show_alarms DEFAULT page_limit=5 — paginate page_limit=20 until short page.
Cancel alarms: subject EXACTLY Skip this time? / Cancel Meeting? / Cannot make it; label EXACT One-on-One with {first_name}.
Gift per roommate: among max(rating) pick product with inventory_quantity >= n roommates; loop exactly n orders.
Splitwise cable: show_groups_balance → find group_name Roommates → show_group(group_id).
debtor_emails=[m["email"] for m in group["members"]]; month from subject not enumerate; desc "cable bill [MM-YY]".
Husband checklist: regex `- N X Name` from attachment; never hardcode qty after download.
Meeting note: simple_note_content_by_title(apis,sn_token,'Weekly Meetings Times') — NEVER search_notes()[0].
create_meeting_reminder_drafts(apis,gmail_token,phone_token,content,supervisor_email) — call directly, never hand-parse.
Cable: record_roommate_cable_bills(apis,gmail_token,splitwise_token,fs_token,payer_email) — call directly.
cable_bill_month_year(subject); parse_cable_bill_amount(content); month_num < cur_month filter.

AMAZON SELLERS: orders have NO seller field. show_order -> order_items -> show_product(product_id) -> seller_id.
Trusted sellers = seller_ids from past orders. search_products results also have seller_id not seller.

SPLITWISE: apis.splitwise.record_expense(description, paid_amount, payer_email, debtor_emails, access_token, group_id=...).
WRONG: create_expense add_expense create_transaction. Roommates group via show_groups_balance not show_groups.
API docs: apis.api_docs.show_api_descriptions(app_name='splitwise') — never apis.splitwise.show_api_descriptions.

FILE PATHS: use exact path from download_attachment file_path. Case-sensitive /home/carl/downloads.
Never concatenate paths (no double slashes). show_directory returns full paths.

PLACE_ORDER: requires access_token payment_card_id address_id. Orders ENTIRE cart — show_cart first,
remove unwanted items. Wishlist order: move_product_from_wish_list_to_cart then place_order.

ANSWERS: action tasks -> complete_task(answer=None). Questions -> bare number/string only.

PHONE: phone APIs need phone_token from phone.login(username=phone_number). search_contacts for partner/roommate emails.

GMAIL: email_thread_id starred archived. show_thread for details. Plain lists not .get("success").

AMAZON PRODUCTS: product_id num_product_reviews (not review_count) rating price name product_type.
"""

# Shared hackathon key from organizers — used when HYDRA_DB_API_KEY is not in .env.
DEFAULT_HYDRA_DB_API_KEY = (
    "sk_live_i9l10ZRyeYB4.AnhcOUWlUpwYkeot6K5z0v6BEocVmpVpyg9nKmPJ01U"
)
# Matches APPWORLD_EXPERIMENT / experiments/outputs/team_prod for this submission.
DEFAULT_HYDRA_TENANT_ID = "team_prod"


class HydraContext:
    """Optional HydraDB client for retrieval-augmented agent steps."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        max_results: int = 5,
        graph_context: bool = True,
        seed_knowledge: bool = True,
        cache_dir: Path = Path("traces"),
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("HYDRA_DB_API_KEY")
            or os.environ.get("HYDRA_API_KEY")
            or DEFAULT_HYDRA_DB_API_KEY
        )
        self.tenant_id = (
            tenant_id
            or os.environ.get("HYDRA_TENANT_ID")
            or os.environ.get("APPWORLD_EXPERIMENT")
            or DEFAULT_HYDRA_TENANT_ID
        )
        self.max_results = int(os.environ.get("HYDRA_MAX_RESULTS", str(max_results)))
        self.graph_context = os.environ.get("HYDRA_GRAPH_CONTEXT", "1") not in ("0", "false", "False")
        self.seed_knowledge = os.environ.get("HYDRA_SEED_KNOWLEDGE", "1" if seed_knowledge else "0") not in (
            "0", "false", "False"
        )
        self.cache_dir = cache_dir
        self.enabled = bool(self.api_key) and HydraDB is not None and os.environ.get(
            "HYDRA_ENABLED", "1"
        ) not in ("0", "false", "False")
        self.client: Any = None
        self._bootstrapped = False

    def bootstrap(self) -> None:
        if not self.enabled:
            return
        try:
            self.client = HydraDB(token=self.api_key)
            self._ensure_tenant_ready()
            if self.seed_knowledge:
                self._seed_playbook_if_needed()
            self._bootstrapped = True
            print(f"  HydraDB ready (tenant={self.tenant_id})")
        except Exception as exc:
            self.enabled = False
            print(f"  ! HydraDB disabled: {exc}")

    def retrieve(self, query: str, *, search_type: str = "all") -> str:
        if not self.enabled or not self.client or not query.strip():
            return ""
        try:
            response = self.client.query(
                tenant_id=self.tenant_id,
                query=query.strip(),
                type=search_type,
                max_results=self.max_results,
                graph_context=self.graph_context,
            )
            text = build_string(response).strip()
            if text == "No relevant context found.":
                return ""
            return text
        except Exception:
            return ""

    def remember(self, text: str) -> None:
        if not self.enabled or not self.client or not text.strip():
            return
        try:
            self.client.context.ingest(
                type="memory",
                tenant_id=self.tenant_id,
                memories=json.dumps([{"text": text.strip()[:4000]}]),
            )
        except Exception:
            pass

    def remember_step(
        self,
        *,
        task_id: str,
        step: int,
        instruction: str,
        code: str,
        observation: str,
        succeeded: bool,
    ) -> None:
        status = "success" if succeeded else "failure"
        obs_preview = observation.strip().replace("\n", " ")[:600]
        code_preview = code.strip().replace("\n", " ")[:400]
        text = (
            f"[{status}] task={task_id} step={step}\n"
            f"Instruction: {instruction[:300]}\n"
            f"Code: {code_preview}\n"
            f"Observation: {obs_preview}"
        )
        self.remember(text)

    def _ensure_tenant_ready(self, timeout_s: int = 120) -> None:
        assert self.client is not None
        try:
            self.client.tenants.create(tenant_id=self.tenant_id)
        except Exception:
            pass
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            status = self.client.tenants.status(tenant_id=self.tenant_id)
            infra = status.data.infra if status.data else None
            if infra and getattr(infra, "ready_for_ingestion", False):
                return
            time.sleep(2)
        raise TimeoutError(f"HydraDB tenant {self.tenant_id!r} not ready for ingestion")

    def _seed_marker(self) -> Path:
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in self.tenant_id)
        return self.cache_dir / f".hydra_seeded_v13_{safe}"

    def _seed_playbook_if_needed(self) -> None:
        assert self.client is not None
        marker = self._seed_marker()
        if marker.is_file():
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        doc_name = "appworld_agent_playbook.txt"
        ingest = self.client.context.ingest(
            type="knowledge",
            tenant_id=self.tenant_id,
            documents=[(doc_name, APPWORLD_KNOWLEDGE_SEED.encode("utf-8"), "text/plain")],
        )
        source_id = ingest.data.results[0].id
        deadline = time.time() + 180
        while time.time() < deadline:
            st = self.client.context.status(tenant_id=self.tenant_id, ids=[source_id])
            indexing_status = st.data.statuses[0].indexing_status
            if indexing_status == "completed":
                marker.write_text(source_id, encoding="utf-8")
                return
            if indexing_status == "errored":
                err = getattr(st.data.statuses[0], "error_message", indexing_status)
                raise RuntimeError(f"HydraDB seed failed: {err}")
            time.sleep(2)
        raise TimeoutError("HydraDB knowledge seed indexing timed out")

    def format_block(self, text: str, title: str = "HYDRADB CONTEXT") -> str:
        if not text:
            return ""
        return f"=== {title} ===\n{text}\n"


def error_query(instruction: str, observation: str, limit: int = 500) -> str:
    obs = observation.strip()
    if len(obs) > limit:
        obs = obs[-limit:]
    return f"AppWorld task: {instruction}\nError or failure:\n{obs}"
