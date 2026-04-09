/**
 * Hook framework — reads Claude Code hook JSON from stdin,
 * dispatches to the appropriate handler, writes JSON to stdout.
 *
 * Handles deduplication (SHA-256, 600s window) and noise filtering.
 */

import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

// ── Types ────────────────────────────────────────────────────────────────────

export interface HookInput {
  session_id: string;
  transcript_path?: string;
  cwd?: string;
  hook_event_name: string;
  prompt?: string;           // UserPromptSubmit
  source?: string;           // SessionStart (startup|resume|clear|compact)
  trigger?: string;          // PreCompact (manual|auto)
  context_used?: number;     // PreCompact
}

export interface HookOutput {
  continue?: boolean;
  hookSpecificOutput?: {
    hookEventName: string;
    additionalContext?: string;
    decision?: "block" | "allow";
  };
}

export type HookHandler = (input: HookInput) => Promise<HookOutput | null>;

// ── Dedup ────────────────────────────────────────────────────────────────────

const DEDUP_WINDOW_SEC = 600; // 10 minutes

interface DedupEntry {
  hash: string;
  timestamp: number;
}

function dedupPath(): string {
  return join(homedir(), ".mnemon", "dedup.json");
}

function loadDedupCache(): DedupEntry[] {
  const path = dedupPath();
  if (!existsSync(path)) return [];
  try {
    return JSON.parse(readFileSync(path, "utf-8"));
  } catch {
    return [];
  }
}

function saveDedupCache(entries: DedupEntry[]): void {
  const dir = join(homedir(), ".mnemon");
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  writeFileSync(dedupPath(), JSON.stringify(entries));
}

export function isDuplicate(text: string): boolean {
  const hash = createHash("sha256").update(text).digest("hex");
  const now = Date.now() / 1000;

  let entries = loadDedupCache();

  // Prune expired entries
  entries = entries.filter((e) => now - e.timestamp < DEDUP_WINDOW_SEC);

  // Check for duplicate
  if (entries.some((e) => e.hash === hash)) {
    return true;
  }

  // Add new entry
  entries.push({ hash, timestamp: now });
  saveDedupCache(entries);
  return false;
}

// ── Noise Filtering ──────────────────────────────────────────────────────────

const NOISE_PATTERNS = [
  /^\s*$/,                          // empty
  /^\/\w/,                          // slash commands (/help, /clear, etc.)
  /^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|sure|yep|nope)\s*[.!?]?\s*$/i,
  /^(good morning|good night|bye|goodbye)\s*[.!?]?\s*$/i,
  /^y$/i,                           // single letter confirmations
  /^n$/i,
];

export function isNoise(prompt: string): boolean {
  const trimmed = prompt.trim();
  if (trimmed.length < 3) return true;
  return NOISE_PATTERNS.some((p) => p.test(trimmed));
}

// ── Stdin Reader ─────────────────────────────────────────────────────────────

export async function readStdin(): Promise<HookInput> {
  const chunks: Buffer[] = [];
  for await (const chunk of Bun.stdin.stream()) {
    chunks.push(Buffer.from(chunk));
  }
  const raw = Buffer.concat(chunks).toString("utf-8");
  return JSON.parse(raw);
}

// ── Stdout Writer ────────────────────────────────────────────────────────────

export function writeOutput(output: HookOutput): void {
  process.stdout.write(JSON.stringify(output));
}

// ── Transcript Reader ────────────────────────────────────────────────────────

export interface TranscriptMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

/**
 * Read the last N characters from the transcript JSONL file.
 * Returns concatenated user + assistant messages.
 */
export function readTranscript(transcriptPath: string, maxChars = 8000): string {
  if (!transcriptPath || !existsSync(transcriptPath)) return "";

  try {
    const raw = readFileSync(transcriptPath, "utf-8");
    const lines = raw.trim().split("\n");
    const messages: string[] = [];
    let totalChars = 0;

    // Read from the end
    for (let i = lines.length - 1; i >= 0 && totalChars < maxChars; i--) {
      try {
        const msg = JSON.parse(lines[i]!);
        const role = msg.role ?? "unknown";
        let content = "";

        if (typeof msg.content === "string") {
          content = msg.content;
        } else if (Array.isArray(msg.content)) {
          content = msg.content
            .filter((c: any) => c.type === "text")
            .map((c: any) => c.text)
            .join("\n");
        }

        if (content && (role === "user" || role === "assistant")) {
          messages.unshift(`[${role}]: ${content}`);
          totalChars += content.length;
        }
      } catch {
        // Skip malformed lines
      }
    }

    return messages.join("\n\n");
  } catch {
    return "";
  }
}
