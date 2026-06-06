# Failure taxonomy (live)

_Updated: 2026-06-06T18:43:10.026346+00:00_

## Ranked categories (frequency × tasks blocked)

| rank | category | hits | tasks |
|------|----------|------|-------|
| 1 | VALIDATION_422 | 33 | 9 |
| 2 | CARD_VALIDITY | 18 | 6 |
| 3 | FAKE_RECIPIENT | 10 | 6 |
| 4 | LLM_ERROR | 6 | 6 |
| 5 | FIELD_KEYERROR | 6 | 4 |
| 6 | CASING | 7 | 3 |
| 7 | TOKEN_MISUSE | 5 | 3 |
| 8 | WRONG_API_NAME | 3 | 2 |

## Representative snippets

### VALIDATION_422
- 0de03ad_2: Execution failed. Traceback:
  File "<python-input>", line 14, in <module>
    process_threads(inbox_threads, coworker_emails, access_token)
- 0de03ad_2: Execution failed. Traceback:
  File "<python-input>", line 15, in <module>
    process_threads(inbox_threads, coworker_emails, access_token)
- 0de03ad_2: Execution failed. Traceback:
  File "<python-input>", line 15, in <module>
    process_threads(inbox_threads, coworker_emails, access_token)

### CARD_VALIDITY
- 1c4bd27_3: [
 {
  "name": "show_account",
  "description": "Show your account information. Unlike show_profile, this includes private information."
 },
- 4242c97_1: [
 {
  "name": "show_account",
  "description": "Show your account information. Unlike show_profile, this includes private information."
 },
- 4242c97_1: Execution failed. Traceback:
  File "<python-input>", line 7, in <module>
    order_response = apis.amazon.place_order(
                    

### FAKE_RECIPIENT
- 0de03ad_2: {
 "app_name": "gmail",
 "api_name": "show_inbox_threads",
 "path": "/email_threads/category/inbox",
 "method": "GET",
 "description": "Show
- 0de03ad_2: {
 "app_name": "gmail",
 "api_name": "search_users",
 "path": "/users",
 "method": "GET",
 "description": "Search Gmail users by name or ema
- 69ba40f_1: {
 "app_name": "gmail",
 "api_name": "show_outbox_threads",
 "path": "/email_threads/category/outbox",
 "method": "GET",
 "description": "Sh

### LLM_ERROR
- 258796c_2: Traceback (most recent call last):
  File "C:\Users\moham\Projects\hack_agent_arena\.venv\Lib\site-packages\litellm\llms\openai.py", line 41
- 5238afc_1: Traceback (most recent call last):
  File "C:\Users\moham\Projects\hack_agent_arena\.venv\Lib\site-packages\litellm\llms\openai.py", line 41
- 6a5e690_3: Traceback (most recent call last):
  File "C:\Users\moham\Projects\hack_agent_arena\.venv\Lib\site-packages\litellm\llms\openai.py", line 41

### FIELD_KEYERROR
- 4242c97_1: KeyError id
- 4441ee9_2: KeyError is_starred
- 96bf160_3: KeyError id

### CASING
- 1c4bd27_3: StopIteration on password lookup
- cdaaea5_3: Title-case account_name lookup
- cdaaea5_3: Title-case account_name lookup

### TOKEN_MISUSE
- 0de03ad_2: Starring thread 26753 with participants ['deniseburch@gmail.com', 'notifications@amazon.com']
Failed to star thread 26753: Response status c
- 0de03ad_2: Failed to star thread 26753: Response status code is 422:
{"message":"This email thread is already marked as starred."}
Failed to star threa
- adb1060_2: Execution failed. Traceback:
  File "<python-input>", line 2, in <module>
    print(apis.phone.show_contact_relationships())
          ^^^^^

### WRONG_API_NAME
- 1c4bd27_3: No API named 'list_orders'
- fa327a6_3: No API named 'list_orders'
- fa327a6_3: No API named 'list_transactions'

## Eval pass/fail

- `18670a5_3`: PASS
- `20c1328_3`: PASS
- `23d431c_3`: PASS
- `5e27cd7_2`: PASS
- `8d42650_3`: PASS
- `9871968_2`: PASS
- `ba46d91_2`: PASS
- `c1091c7_2`: PASS
- `dbc0276_3`: PASS
- `f6be291_1`: PASS
