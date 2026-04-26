# bench/

Performance benchmarks for mnemon. Numbers here are **indicative**, not guarantees — they're recorded against a single development machine to catch regressions and inform marketing claims, not to set SLAs.

## search_stress.py

Times hybrid BM25 + vector search across a synthetic corpus.

```bash
# Default — 1000 memories, 50 queries
python bench/search_stress.py

# Stress at higher scale
python bench/search_stress.py --memories 5000 --queries 100

# Save baseline JSON for CI / regression detection
python bench/search_stress.py --json bench/baseline-1k.json
```

Uses an isolated `MNEMON_VAULT_DIR` under `tempfile.mkdtemp()`, so it never touches the user's real `~/.mnemon`. Corpus is generated from a fixed seed (`--seed 42` by default), so re-runs on the same machine differ only by machine variance, not corpus randomness.

## Latest baseline numbers

Captured on a 2024 MacBook Pro (M3 Pro, 36GB RAM) running `mnemon-memory==0.6.0rc5`. `mnemon` is single-process Python; numbers should generalize to similar consumer hardware.

| corpus | save (per memory) | embed (per memory) | search p50 | search p95 | search p99 |
|--------|-------------------|--------------------|-----------:|-----------:|-----------:|
| 1,000  | 0.1 ms            | 7.7 ms             |    3.0 ms  |    3.7 ms  |    3.8 ms  |
| 5,000  | 0.1 ms            | 8.7 ms             |    6.1 ms  |    7.5 ms  |    7.8 ms  |

Embedding cost is the FastEmbed bge-small-en-v1.5 forward pass; it grows per-memory but is one-time-on-save and runs in a background path in the MCP server. Search latency grows roughly linearly in corpus size — consistent with the brute-force cosine over numpy arrays the vector store uses today. README's "comfortable up to ~50k memories" claim implies extrapolated p99 around 70-80 ms; that's the next number worth measuring.

## When to re-run

- Before publishing a new release — confirm no regression from the prior baseline.
- After any change to `search.py`, `store.py:search_bm25` / `search_vector`, `vecstore.py`, or the embedder.
- When adding a new query class that should be benchmarked alongside the existing pool.

If a measurement comes in materially worse than the table above, treat it as a regression unless you can explain it (laptop thermal throttling, background CPU contention, etc.).

## Adding benchmarks

Same conventions as `examples/`:

- Throwaway vault under `tempfile.mkdtemp` — never touch `~/.mnemon`
- Deterministic seed for reproducibility
- Print enough that someone reading stdout without running it understands the result
- `--json` output for CI integration

Open a PR. Benchmarks are throwaway scripts, not production code — no test coverage required, but a comment block at the top explaining what's being measured and why is non-negotiable.
