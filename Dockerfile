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

# Vault data persists in /data (mount a Fly volume here)
ENV MNEMON_VAULT_DIR=/data
RUN mkdir -p /data

# Default port (Fly.io uses 8080 internally)
ENV PORT=8080

EXPOSE 8080

# Health check has a generous start period because the server pre-loads
# the embedding model on startup (see server_remote.py) — uvicorn does
# not bind the port until that load completes (~3-5 seconds on warm
# disk, longer on first-ever boot if the model isn't yet cached).
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health', timeout=3).status == 200 else 1)"

CMD ["mnemon", "serve-remote"]
