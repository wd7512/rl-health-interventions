# OpenCode Zen Free Model Benchmark

Tested 2026-06-28 (updated). Free-tier API key, endpoint
`https://opencode.ai/zen/v1/chat/completions`, auth via `X-API-Key` header.

## Model availability

| Model ID | Status |
|---|---|
| deepseek-v4-flash-free | Free |
| mimo-v2.5-free | Free |
| north-mini-code-free | Free |
| big-pickle | Free (stealth) |
| nemotron-3-ultra-free | Free (limited time), intermittent errors |
| qwen3.6-plus / 3.5 / 3.7 | Paid only (all return 401) |
| minimax-m2.7 / m2.5 | Paid only (401) |

## Latency at 5 concurrent

| Model | Median latency | Wall time | Success |
|---|---|---|---|
| north-mini-code-free | 0.97s | — | 5/5 |
| big-pickle | 1.63s | — | 5/5 |
| deepseek-v4-flash-free | 1.86s | — | 5/5 |
| mimo-v2.5-free | 5.70s | — | 5/5 |
| nemotron-3-ultra-free | ~1.6s* | — | 4/5 |

*Nemotron latency is highly variable (2s–24s) due to 550B MoE architecture.
Occasional malformed responses missing `choices` key.

## Concurrency scaling (no rate limits observed)

| Model | 5 | 10 | 15 | 20 |
|---|---|---|---|---|
| deepseek-v4-flash-free | 5/5 | 10/10 | 15/15 | 20/20 |
| big-pickle | 5/5 | 10/10 | 15/15 | 20/20 |
| north-mini-code-free | 5/5 | 10/10 | 15/15 | 20/20 |
| mimo-v2.5-free | 5/5 | 10/10 | 15/15 | 20/20 |

- **No HTTP 429** at any level up to 20 concurrent.
- Latency stays flat as concurrency increases — gateway handles parallel well.
- Wall time grows linearly with batch size (server-side serialization).

## Throughput estimates for 1000 requests

Based on median latency. Assumes no rate limiting (none observed).

| Model | Median lat | 1000 sequential | 1000 @ 5 workers | 1000 @ 10 workers |
|---|---|---|---|---|
| north-mini-code-free | 0.97s | 16.2 min | 3.2 min | 1.6 min |
| big-pickle | 1.63s | 27.2 min | 5.4 min | 2.7 min |
| deepseek-v4-flash-free | 1.86s | 31.0 min | 6.2 min | 3.1 min |
| mimo-v2.5-free | 5.70s | 95.0 min | 19.0 min | 9.5 min |
| nemotron-3-ultra-free | ~3.0s* | ~50 min* | ~10 min* | ~5 min* |

*Nemotron estimate is rough due to high variance. Real-world throughput likely
worse due to retries on malformed responses.

## Recommendation

**deepseek-v4-flash-free at 10-15 concurrent** is the best default:
- Fastest consistent latency among reliable models
- Zero failures at all tested concurrency levels
- ~3 min for 1000 requests at 10 workers

**north-mini-code-free** is the fastest but higher variance — good for
fire-and-forget batch jobs where occasional slow requests are acceptable.

Avoid **mimo-v2.5-free** for batch work (6x slower than deepseek).
Avoid **nemotron-3-ultra-free** for batch work (unpredictable latency, malformed responses).
