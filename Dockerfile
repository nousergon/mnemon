FROM python:3.13-slim

WORKDIR /app

# Install mnemon with all dependencies
COPY . .
RUN pip install --no-cache-dir ".[ui]" uvicorn starlette

# Vault data persists in /data (mount a Fly volume here)
ENV MNEMON_VAULT_DIR=/data
RUN mkdir -p /data

# Default port (Fly.io uses 8080 internally)
ENV PORT=8080

EXPOSE 8080

CMD ["mnemon", "serve-remote"]
