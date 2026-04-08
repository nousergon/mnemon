/**
 * mnemon — Universal long-term memory layer for AI agents.
 *
 * CLI entry point. Dispatches to subcommands:
 *   mnemon serve     — start MCP server (stdio)
 *   mnemon status    — show vault health
 *   mnemon search    — search memories from CLI
 *   mnemon save      — save a memory from CLI
 *   mnemon setup     — configure Claude Code / Cursor / Gemini CLI
 */

import { Store } from "./store.ts";
import { search } from "./search.ts";
import { embedDocument, VECTOR_DIM } from "./embedder.ts";
import { join } from "node:path";
import { homedir } from "node:os";
import { existsSync, writeFileSync, readFileSync } from "node:fs";

const args = process.argv.slice(2);
const command = args[0];

async function main() {
  switch (command) {
    case "serve":
      // Import and run MCP server
      await import("./mcp.ts");
      break;

    case "status": {
      const store = new Store(undefined, VECTOR_DIM);
      const stats = store.status();
      console.log(`Vault: ${stats.vault_path}`);
      console.log(`Total memories: ${stats.total_documents}`);
      console.log(`Vectors: ${stats.total_vectors}`);
      console.log(`Pinned: ${stats.pinned}`);
      console.log(`Invalidated: ${stats.invalidated}`);
      console.log("\nBy type:");
      for (const t of stats.by_type as Array<{ content_type: string; count: number }>) {
        console.log(`  ${t.content_type}: ${t.count}`);
      }
      store.close();
      break;
    }

    case "search": {
      const query = args.slice(1).join(" ");
      if (!query) {
        console.error("Usage: mnemon search <query>");
        process.exit(1);
      }
      const store = new Store(undefined, VECTOR_DIM);
      const results = await search(store, { query, limit: 10, useVector: true });
      if (results.length === 0) {
        console.log("No memories found.");
      } else {
        for (const r of results) {
          const snippet = r.content.slice(0, 200);
          console.log(`[${r.content_type}] ${r.title} (score: ${r.composite_score.toFixed(3)})`);
          console.log(`  ${snippet}${r.content.length > 200 ? "..." : ""}`);
          console.log();
        }
      }
      store.close();
      break;
    }

    case "save": {
      const title = args[1];
      const content = args.slice(2).join(" ");
      if (!title || !content) {
        console.error("Usage: mnemon save <title> <content>");
        process.exit(1);
      }
      const store = new Store(undefined, VECTOR_DIM);
      const docId = store.save({ title, content, source_client: "cli" });
      console.log(`Saved memory #${docId}: "${title}"`);

      // Embed
      const doc = store.get(docId);
      if (doc) {
        const count = await embedDocument(store, doc.hash, title, content);
        console.log(`Embedded ${count} fragments.`);
      }
      store.close();
      break;
    }

    case "setup": {
      const target = args[1];
      if (!target || !["claude-code", "cursor", "gemini", "hooks"].includes(target)) {
        console.error("Usage: mnemon setup <claude-code|cursor|gemini|hooks>");
        process.exit(1);
      }
      setupIntegration(target);
      break;
    }

    default:
      console.log(`mnemon v0.1.0 — Universal long-term memory for AI agents

Usage:
  mnemon serve          Start MCP server (stdio transport)
  mnemon status         Show vault health stats
  mnemon search <query> Search memories
  mnemon save <title> <content>  Save a memory
  mnemon setup <target> Configure integration (claude-code, cursor, gemini)
`);
      break;
  }
}

function setupIntegration(target: string) {
  const mnemonPath = join(process.cwd(), "src", "mcp.ts");
  const mcpConfig = {
    command: "bun",
    args: ["run", mnemonPath],
  };

  switch (target) {
    case "claude-code": {
      const settingsPath = join(homedir(), ".claude", "settings.json");
      let settings: any = {};
      if (existsSync(settingsPath)) {
        settings = JSON.parse(readFileSync(settingsPath, "utf-8"));
      }
      if (!settings.mcpServers) settings.mcpServers = {};
      settings.mcpServers.mnemon = mcpConfig;
      writeFileSync(settingsPath, JSON.stringify(settings, null, 2));
      console.log(`Claude Code MCP configured at ${settingsPath}`);
      console.log("Restart Claude Code to pick up the new MCP server.");
      break;
    }

    case "cursor": {
      const cursorPath = join(homedir(), ".cursor", "mcp.json");
      let config: any = {};
      if (existsSync(cursorPath)) {
        config = JSON.parse(readFileSync(cursorPath, "utf-8"));
      }
      if (!config.mcpServers) config.mcpServers = {};
      config.mcpServers.mnemon = mcpConfig;
      writeFileSync(cursorPath, JSON.stringify(config, null, 2));
      console.log(`Cursor MCP configured at ${cursorPath}`);
      console.log("Restart Cursor to pick up the new MCP server.");
      break;
    }

    case "gemini": {
      console.log("Gemini CLI MCP configuration:");
      console.log(`Add to your Gemini CLI config:\n`);
      console.log(JSON.stringify({ mnemon: mcpConfig }, null, 2));
      break;
    }

    case "hooks": {
      setupHooks();
      break;
    }
  }
}

function setupHooks() {
  const settingsPath = join(homedir(), ".claude", "settings.json");
  let settings: any = {};
  if (existsSync(settingsPath)) {
    settings = JSON.parse(readFileSync(settingsPath, "utf-8"));
  }

  if (!settings.hooks) settings.hooks = {};

  const hooksDir = join(process.cwd(), "src", "hooks");

  // UserPromptSubmit — context surfacing
  settings.hooks.UserPromptSubmit = [
    {
      matcher: "",
      hooks: [
        {
          type: "command",
          command: `bun run ${join(hooksDir, "context-surfacing.ts")}`,
          timeout: 8,
        },
      ],
    },
  ];

  // Stop — session extractor + handoff generator
  settings.hooks.Stop = [
    {
      matcher: "",
      hooks: [
        {
          type: "command",
          command: `bun run ${join(hooksDir, "session-extractor.ts")}`,
          timeout: 30,
        },
        {
          type: "command",
          command: `bun run ${join(hooksDir, "handoff-generator.ts")}`,
          timeout: 30,
        },
      ],
    },
  ];

  writeFileSync(settingsPath, JSON.stringify(settings, null, 2));
  console.log(`Claude Code hooks configured at ${settingsPath}`);
  console.log("\nHooks installed:");
  console.log("  UserPromptSubmit → context-surfacing (8s timeout)");
  console.log("  Stop → session-extractor (30s timeout)");
  console.log("  Stop → handoff-generator (30s timeout)");
  console.log("\nRestart Claude Code to activate hooks.");
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
