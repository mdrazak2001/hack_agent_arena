# Scoreboard

Submission model: `openrouter/meta-llama/llama-3.3-70b-instruct` (no GROQ_API_KEY in `.env`).

| iter | model | TGC | SGC | diff1 | diff2 | diff3 | dev TGC | notes |
|------|-------|-----|-----|-------|-------|-------|---------|-------|
| baseline | openrouter/meta-llama/llama-3.3-70b-instruct | 80.0 | 80.0 | 100% | 100% | 50% | — | 8/10 pass; fail 18670a5_3, 8d42650_3 |
| iter2 | openrouter/meta-llama/llama-3.3-70b-instruct | **100.0** | **100.0** | 100% | 100% | **100%** | 10/10 completed | helpers + recovery hints; simple_note title filter |
