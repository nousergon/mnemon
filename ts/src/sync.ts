/**
 * S3 vault sync — push/pull SQLite vault + vector store to/from S3.
 *
 * Uses AWS CLI (no SDK dependency). Content-addressable storage
 * prevents content duplication. Last-write-wins for metadata.
 *
 * Usage:
 *   MNEMON_S3_BUCKET=my-bucket bun run src/sync.ts push
 *   MNEMON_S3_BUCKET=my-bucket bun run src/sync.ts pull
 */

import { existsSync, statSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

const S3_BUCKET = process.env.MNEMON_S3_BUCKET ?? "";
const S3_PREFIX = process.env.MNEMON_S3_PREFIX ?? "mnemon/vaults";
const VAULT_NAME = process.env.MNEMON_VAULT_NAME ?? "default";

function vaultDir(): string {
  return join(homedir(), ".mnemon");
}

function vaultFiles(): { sqlite: string; vec: string } {
  const dir = vaultDir();
  return {
    sqlite: join(dir, `${VAULT_NAME}.sqlite`),
    vec: join(dir, `${VAULT_NAME}.vec`),
  };
}

function s3Path(filename: string): string {
  return `s3://${S3_BUCKET}/${S3_PREFIX}/${filename}`;
}

async function runCmd(cmd: string): Promise<{ ok: boolean; output: string }> {
  const proc = Bun.spawn(["bash", "-c", cmd], {
    stdout: "pipe",
    stderr: "pipe",
  });

  const stdout = await new Response(proc.stdout).text();
  const stderr = await new Response(proc.stderr).text();
  const exitCode = await proc.exited;

  return {
    ok: exitCode === 0,
    output: exitCode === 0 ? stdout.trim() : stderr.trim(),
  };
}

/**
 * Push local vault to S3.
 */
export async function push(): Promise<{ pushed: string[]; errors: string[] }> {
  if (!S3_BUCKET) {
    return { pushed: [], errors: ["MNEMON_S3_BUCKET not set"] };
  }

  const files = vaultFiles();
  const pushed: string[] = [];
  const errors: string[] = [];

  for (const [label, localPath] of Object.entries(files)) {
    if (!existsSync(localPath)) continue;

    const s3Target = s3Path(`${VAULT_NAME}.${label === "sqlite" ? "sqlite" : "vec"}`);
    const result = await runCmd(`aws s3 cp "${localPath}" "${s3Target}" --only-show-errors`);

    if (result.ok) {
      const size = statSync(localPath).size;
      pushed.push(`${label}: ${(size / 1024).toFixed(1)}KB → ${s3Target}`);
    } else {
      errors.push(`${label}: ${result.output}`);
    }
  }

  return { pushed, errors };
}

/**
 * Pull vault from S3 to local.
 */
export async function pull(): Promise<{ pulled: string[]; errors: string[] }> {
  if (!S3_BUCKET) {
    return { pulled: [], errors: ["MNEMON_S3_BUCKET not set"] };
  }

  const files = vaultFiles();
  const pulled: string[] = [];
  const errors: string[] = [];

  for (const [label, localPath] of Object.entries(files)) {
    const s3Source = s3Path(`${VAULT_NAME}.${label === "sqlite" ? "sqlite" : "vec"}`);

    // Check if file exists on S3
    const exists = await runCmd(`aws s3 ls "${s3Source}" 2>/dev/null`);
    if (!exists.ok || !exists.output) continue;

    const result = await runCmd(`aws s3 cp "${s3Source}" "${localPath}" --only-show-errors`);

    if (result.ok) {
      const size = existsSync(localPath) ? statSync(localPath).size : 0;
      pulled.push(`${label}: ${s3Source} → ${(size / 1024).toFixed(1)}KB`);
    } else {
      errors.push(`${label}: ${result.output}`);
    }
  }

  return { pulled, errors };
}

// ── CLI ─────────────────────────────────────────────────────────────────────

if (import.meta.main) {
  const action = process.argv[2];

  if (action === "push") {
    const result = await push();
    if (result.pushed.length > 0) {
      console.log("Pushed:");
      result.pushed.forEach((p) => console.log(`  ${p}`));
    }
    if (result.errors.length > 0) {
      console.error("Errors:");
      result.errors.forEach((e) => console.error(`  ${e}`));
      process.exit(1);
    }
  } else if (action === "pull") {
    const result = await pull();
    if (result.pulled.length > 0) {
      console.log("Pulled:");
      result.pulled.forEach((p) => console.log(`  ${p}`));
    }
    if (result.errors.length > 0) {
      console.error("Errors:");
      result.errors.forEach((e) => console.error(`  ${e}`));
      process.exit(1);
    }
  } else {
    console.log("Usage: bun run src/sync.ts <push|pull>");
    console.log("\nEnv vars:");
    console.log("  MNEMON_S3_BUCKET  — S3 bucket name (required)");
    console.log("  MNEMON_S3_PREFIX  — S3 key prefix (default: mnemon/vaults)");
    console.log("  MNEMON_VAULT_NAME — vault name (default: default)");
  }
}
