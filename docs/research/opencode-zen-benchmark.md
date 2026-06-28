# OpenCode Zen Free Model Benchmark

Tested 2026-06-28. Free-tier API key, endpoint `https://opencode.ai/zen/v1/chat/completions`,
auth via `X-API-Key` header. Median of 5 requests per batch unless noted.

## Free models available

| Model ID | Status |
|---|---|
| deepseek-v4-flash-free | Free |
| mimo-v2.5-free | Free |
| north-mini-code-free | Free |
| big-pickle | Free (stealth) |
| nemotron-3-ultra-free | Free (limited time) |
| qwen3.6-plus | Paid only (401 on free tier) |
| minimax-m2.7 | Paid only (401 on free tier) |
| minimax-m2.5 | Paid only (401 on free tier) |

## Latency at 5 concurrent

| Model | Median latency | Wall time | Success |
|---|---|---|---|
| big-pickle | 1.7s | 1.9s | 5/5 |
| deepseek-v4-flash-free | 1.8s | 2.0s | 5/5 |
| mimo-v2.5-free | 1.7s | 3.4s | 5/5 |
| north-mini-code-free | 0.9s | 4.6s | 5/5 |
| nemotron-3-ultra-free | 1.6s | 5.0s | 5/5 |

## Concurrency scaling (no rate limits observed)

| Model | 5 concurrent | 10 concurrent | 15 concurrent | 20 concurrent |
|---|---|---|---|---|
| deepseek-v4-flash-free | 5/5 ok | 10/10 ok | 15/15 ok | 20/20 ok |
| big-pickle | 5/5 ok | 10/10 ok | 15/15 ok | 20/20 ok |
| mimo-v2.5-free | 5/5 ok | 10/10 ok | 15/15 ok | 20/20 ok |

- **No HTTP 429 at any level** up to 20 concurrent.
- Latency stays flat as concurrency increases — gateway handles parallel well.
- Wall time grows linearly with batch size (server-side serialization).

## Recommendation

**deepseek-v4-flash-free at 10-15 concurrent** is the sweet spot: fastest consistent
latency, zero failures, 100% success at all tested concurrency levels.

nemotron-3-ultra-free works but is slower (550B MoE model) — use when reasoning
quality matters more than throughput.
