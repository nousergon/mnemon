# Examples

Runnable scripts that demonstrate mnemon's public API directly, no MCP transport required.

## quickstart.py

```bash
pip install mnemon-memory
python examples/quickstart.py
```

Saves three memories with distinct content profiles, embeds them for vector search, then runs three queries against the hybrid BM25 + vector search:

1. **Lexical query** with terms that appear verbatim in one memory
2. **Semantic query** where none of the search terms appear in any memory but one is conceptually related
3. **Diagnostic query** showing how the score function tied between two reasonable matches

Uses an isolated `MNEMON_VAULT_DIR` under `tempfile.mkdtemp()`, so it never touches your real `~/.mnemon` vault. Run repeatedly without side effects — `Store.save` deduplicates by content hash.

## Adding examples

If you write a new example, follow the quickstart pattern:

- Set `MNEMON_VAULT_DIR` to a temp dir before `import mnemon` (the import is cached at first use of any Store-backed function)
- Import from the public API (`mnemon.store.Store`, `mnemon.search.search`) — not `mnemon._private`
- Keep it under ~80 lines and runnable from a fresh `pip install mnemon-memory`
- Print enough output that someone reading the script's stdout *without running it* can see the value

Open a PR. Examples are doc, not benchmarks — see [`bench/`](../bench) (when it lands) for performance numbers.
