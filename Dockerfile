FROM python:3.13-slim

WORKDIR /app

# Install mnemon with server deps (uvicorn, starlette, pyjwt[crypto])
COPY . .
RUN pip install --no-cache-dir ".[server]"

# Vault data persists in /data (mount a Fly volume here)
ENV MNEMON_VAULT_DIR=/data
RUN mkdir -p /data

# Default port (Fly.io uses 8080 internally)
ENV PORT=8080

EXPOSE 8080

# Lightweight health check — hits /health which bypasses auth
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health', timeout=3).status == 200 else 1)"

CMD ["mnemon", "serve-remote"]
