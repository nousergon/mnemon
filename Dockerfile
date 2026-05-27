FROM python:3.13-slim

WORKDIR /app

# Install mnemon with server deps (uvicorn, starlette, pyjwt[crypto])
COPY . .
RUN pip install --no-cache-dir ".[server]"

# Bake the FastEmbed bge-small-en-v1.5 ONNX model into the image so
# cold starts don't trigger a 5-15 second download from HuggingFace Hub.
# Without this, every Fly machine restart wipes /tmp where FastEmbed
# would otherwise cache the model, forcing a re-download on the first
# memory_search call. The cache lives in /app/.cache/fastembed which
# is a layer in the image and survives all restarts.
ENV FASTEMBED_CACHE_DIR=/app/.cache/fastembed
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/app/.cache/fastembed')"

# Bake the NLI cross-encoder (~87 MB INT8 ONNX) for
# memory_check_contradictions — same rationale as the FastEmbed bake.
# Without this, the first contradiction check pays a 5-15 second
# download cost AND risks Anthropic's MCP-proxy timeout on the call.
# Model lives in /app/.cache/huggingface (default HF cache root).
#
# NLI cache resolution (audit 2026-05-27): nli.py:_model_dir() reads
# MNEMON_NLI_MODEL_DIR with a default of ~/.cache/huggingface/hub.
# HF_HOME=/app/.cache/huggingface (set here) is the huggingface_hub
# library's own cache-root convention; with HOME=/root the two paths
# coincide at /app/.cache/huggingface/hub via hf_hub's internal
# resolution. If a future operator needs to override the location,
# set BOTH env vars (HF_HOME for the hf_hub download path AND
# MNEMON_NLI_MODEL_DIR for the runtime load path) to avoid a
# silent-divergence trap.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='cross-encoder/nli-deberta-v3-xsmall', filename='onnx/model_qint8_avx512.onnx'); \
    hf_hub_download(repo_id='cross-encoder/nli-deberta-v3-xsmall', filename='tokenizer.json'); \
    hf_hub_download(repo_id='cross-encoder/nli-deberta-v3-xsmall', filename='config.json')"

# Vault data persists in /data (mount a Fly volume here)
ENV MNEMON_VAULT_DIR=/data
RUN mkdir -p /data

# Default port (Fly.io uses 8080 internally)
ENV PORT=8080

EXPOSE 8080

# Health check has a generous start period because the server pre-loads
# BOTH the embedding model and the NLI classifier on startup (see
# server_remote.py) — uvicorn does not bind the port until both loads
# complete (~5-8 seconds on warm disk, longer on first-ever boot if
# models aren't yet cached). Start period bumped to 45s for the dual
# pre-warm.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health', timeout=3).status == 200 else 1)"

CMD ["mnemon", "serve-remote"]
