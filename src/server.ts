/**
 * Remote HTTP server — Streamable HTTP transport for MCP.
 *
 * Exposes the same MCP tools as stdio mode, accessible from Claude.ai
 * web and iOS via Streamable HTTP. Bearer token auth.
 *
 * Usage:
 *   bun run src/server.ts                    # port 8502, no auth
 *   MNEMON_TOKEN=secret bun run src/server.ts  # with bearer token auth
 *   PORT=9000 bun run src/server.ts          # custom port
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { z } from "zod";
import { Store } from "./store.ts";
import { search } from "./search.ts";
import { VECTOR_DIM } from "./embedder.ts";

const PORT = parseInt(process.env.PORT ?? "8502", 10);
const AUTH_TOKEN = process.env.MNEMON_TOKEN ?? "";
const VAULT_PATH = process.env.MNEMON_VAULT ?? undefined;

// ── Store (remote mode — BM25 only, no GGUF models) ────────────────────────

const store = new Store(VAULT_PATH, VECTOR_DIM);

// ── MCP Server (same tools as stdio, search defaults to BM25-only) ──────────

function createMcpServer(): McpServer {
  const server = new McpServer({
    name: "mnemon",
    version: "0.1.0",
  });

  // ── Retrieval ────────────────────────────────────────────────────────────

  server.tool(
    "memory_search",
    "Search memories using BM25 keyword search with composite scoring.",
    {
      query: z.string().describe("Natural language search query"),
      limit: z.number().optional().default(10),
      content_type: z.enum(["decision", "preference", "antipattern", "observation", "research", "project", "handoff", "note"]).optional(),
    },
    async ({ query, limit, content_type }) => {
      const results = await search(store, {
        query,
        limit,
        contentType: content_type as any,
        useVector: false, // remote = no GPU, BM25 only
        useExpansion: false,
      });

      if (results.length === 0) {
        return { content: [{ type: "text" as const, text: "No memories found." }] };
      }

      const text = results
        .map((r, i) => `${i + 1}. [${r.content_type}] **${r.title}** (score: ${r.composite_score.toFixed(3)})\n   ${r.content.slice(0, 300)}${r.content.length > 300 ? "..." : ""}`)
        .join("\n\n");

      return { content: [{ type: "text" as const, text }] };
    },
  );

  server.tool(
    "memory_get",
    "Get a specific memory by ID.",
    { id: z.number() },
    async ({ id }) => {
      const doc = store.get(id);
      if (!doc) return { content: [{ type: "text" as const, text: `Memory #${id} not found.` }] };
      return {
        content: [{ type: "text" as const, text: `# ${doc.title}\n\n**Type:** ${doc.content_type} | **Confidence:** ${doc.confidence.toFixed(2)} | **Created:** ${doc.created_at}\n\n${(doc as any).doc}` }],
      };
    },
  );

  server.tool(
    "memory_timeline",
    "Get recent memories in chronological order.",
    {
      limit: z.number().optional().default(20),
      content_type: z.enum(["decision", "preference", "antipattern", "observation", "research", "project", "handoff", "note"]).optional(),
    },
    async ({ limit, content_type }) => {
      const docs = store.timeline(limit, content_type as any);
      if (docs.length === 0) return { content: [{ type: "text" as const, text: "No memories found." }] };
      const text = docs.map((d: any) => `- **${d.title}** [${d.content_type}] (id: ${d.id}, ${d.created_at})`).join("\n");
      return { content: [{ type: "text" as const, text }] };
    },
  );

  // ── Mutations ────────────────────────────────────────────────────────────

  server.tool(
    "memory_save",
    "Save a new memory.",
    {
      title: z.string(),
      content: z.string(),
      content_type: z.enum(["decision", "preference", "antipattern", "observation", "research", "project", "handoff", "note"]).optional().default("note"),
      source_client: z.string().optional(),
    },
    async ({ title, content, content_type, source_client }) => {
      const docId = store.save({
        title,
        content,
        content_type: content_type as any,
        source_client: source_client ?? "claude-web",
      });
      return { content: [{ type: "text" as const, text: `Saved memory #${docId}: "${title}" [${content_type}]` }] };
    },
  );

  server.tool(
    "memory_pin",
    "Pin a memory.",
    { id: z.number() },
    async ({ id }) => {
      const ok = store.pin(id);
      return { content: [{ type: "text" as const, text: ok ? `Pinned #${id}.` : `Not found.` }] };
    },
  );

  server.tool(
    "memory_forget",
    "Soft-delete a memory.",
    { id: z.number() },
    async ({ id }) => {
      const ok = store.forget(id);
      return { content: [{ type: "text" as const, text: ok ? `Forgot #${id}.` : `Not found.` }] };
    },
  );

  // ── Lifecycle ────────────────────────────────────────────────────────────

  server.tool(
    "memory_status",
    "Get vault health stats.",
    {},
    async () => {
      const stats = store.status();
      const byType = (stats.by_type as Array<{ content_type: string; count: number }>)
        .map((t) => `  ${t.content_type}: ${t.count}`).join("\n");
      return {
        content: [{ type: "text" as const, text: `Vault: ${stats.vault_path}\nTotal: ${stats.total_documents}\nVectors: ${stats.total_vectors}\nPinned: ${stats.pinned}\n\nBy type:\n${byType}` }],
      };
    },
  );

  // ── Profile ──────────────────────────────────────────────────────────────

  server.tool(
    "profile_get",
    "Get user profile from preferences and decisions.",
    {},
    async () => {
      const preferences = store.timeline(50, "preference" as any);
      const decisions = store.timeline(50, "decision" as any);
      if (preferences.length === 0 && decisions.length === 0) {
        return { content: [{ type: "text" as const, text: "No profile data yet." }] };
      }
      const sections: string[] = [];
      if (preferences.length > 0) {
        sections.push("## Preferences\n" + preferences.map((p: any) => `- **${p.title}**: ${p.doc.slice(0, 200)}`).join("\n"));
      }
      if (decisions.length > 0) {
        sections.push("## Decisions\n" + decisions.map((d: any) => `- **${d.title}**: ${d.doc.slice(0, 200)}`).join("\n"));
      }
      return { content: [{ type: "text" as const, text: sections.join("\n\n") }] };
    },
  );

  return server;
}

// ── HTTP Server ─────────────────────────────────────────────────────────────

// Track transports per session for cleanup
const transports = new Map<string, WebStandardStreamableHTTPServerTransport>();

async function handleMcpRequest(req: Request): Promise<Response> {
  // Auth check
  if (AUTH_TOKEN) {
    const authHeader = req.headers.get("authorization");
    if (!authHeader || authHeader !== `Bearer ${AUTH_TOKEN}`) {
      return new Response("Unauthorized", { status: 401 });
    }
  }

  // Get or create session transport
  const sessionId = req.headers.get("mcp-session-id") ?? undefined;

  if (req.method === "POST") {
    let transport: WebStandardStreamableHTTPServerTransport;

    if (sessionId && transports.has(sessionId)) {
      transport = transports.get(sessionId)!;
    } else {
      // New session
      transport = new WebStandardStreamableHTTPServerTransport({
        sessionIdGenerator: () => crypto.randomUUID(),
      });

      const server = createMcpServer();
      await server.connect(transport);

      // Store transport by session ID after connection
      if (transport.sessionId) {
        transports.set(transport.sessionId, transport);
      }

      transport.onclose = () => {
        if (transport.sessionId) {
          transports.delete(transport.sessionId);
        }
      };
    }

    return transport.handleRequest(req);
  }

  if (req.method === "GET") {
    // SSE stream for server-initiated messages
    if (sessionId && transports.has(sessionId)) {
      return transports.get(sessionId)!.handleRequest(req);
    }
    return new Response("Session not found", { status: 404 });
  }

  if (req.method === "DELETE") {
    // Close session
    if (sessionId && transports.has(sessionId)) {
      const transport = transports.get(sessionId)!;
      await transport.close();
      transports.delete(sessionId);
      return new Response(null, { status: 204 });
    }
    return new Response("Session not found", { status: 404 });
  }

  return new Response("Method not allowed", { status: 405 });
}

// ── Start ───────────────────────────────────────────────────────────────────

Bun.serve({
  port: PORT,
  fetch(req) {
    const url = new URL(req.url);

    // Health check
    if (url.pathname === "/health") {
      return new Response(JSON.stringify({ status: "ok", version: "0.1.0" }), {
        headers: { "content-type": "application/json" },
      });
    }

    // MCP endpoint
    if (url.pathname === "/mcp") {
      return handleMcpRequest(req);
    }

    return new Response("Not found", { status: 404 });
  },
});

console.log(`mnemon remote server running on http://localhost:${PORT}/mcp`);
console.log(`Auth: ${AUTH_TOKEN ? "enabled (Bearer token)" : "disabled"}`);
console.log(`Health: http://localhost:${PORT}/health`);
