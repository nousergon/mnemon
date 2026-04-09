/**
 * MCP server — exposes mnemon memory tools via stdio transport.
 *
 * Tools: memory_search, memory_get, memory_related, memory_timeline,
 *        memory_save, memory_pin, memory_forget,
 *        memory_status, memory_sweep, memory_rebuild
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { Store, type ContentType } from "./store.ts";
import { search } from "./search.ts";
import { embedDocument, VECTOR_DIM } from "./embedder.ts";

// ── Initialize ──────────────────────────────────────────────────────────────

const store = new Store(undefined, VECTOR_DIM);

const server = new McpServer({
  name: "mnemon",
  version: "0.1.0",
});

// ── Retrieval Tools ─────────────────────────────────────────────────────────

server.tool(
  "memory_search",
  "Search memories using hybrid BM25 + vector search with composite scoring. This is the primary entry point for finding relevant memories.",
  {
    query: z.string().describe("Natural language search query"),
    limit: z.number().optional().default(10).describe("Max results to return"),
    content_type: z.enum(["decision", "preference", "antipattern", "observation", "research", "project", "handoff", "note"]).optional().describe("Filter by content type"),
  },
  async ({ query, limit, content_type }) => {
    const results = await search(store, {
      query,
      limit,
      contentType: content_type as ContentType | undefined,
      useVector: true,
    });

    if (results.length === 0) {
      return {
        content: [{ type: "text" as const, text: "No memories found matching your query." }],
      };
    }

    const text = results
      .map((r, i) => {
        const snippet = r.content.slice(0, 300);
        return `${i + 1}. [${r.content_type}] **${r.title}** (score: ${r.composite_score.toFixed(3)}, confidence: ${r.confidence.toFixed(2)})\n   ${snippet}${r.content.length > 300 ? "..." : ""}\n   _id: ${r.doc_id} | created: ${r.created_at}_`;
      })
      .join("\n\n");

    return {
      content: [{ type: "text" as const, text }],
    };
  },
);

server.tool(
  "memory_get",
  "Get a specific memory by its ID. Returns the full content.",
  {
    id: z.number().describe("Document ID"),
  },
  async ({ id }) => {
    const doc = store.get(id);
    if (!doc) {
      return {
        content: [{ type: "text" as const, text: `Memory #${id} not found.` }],
      };
    }

    return {
      content: [{
        type: "text" as const,
        text: `# ${doc.title}\n\n**Type:** ${doc.content_type} | **Confidence:** ${doc.confidence.toFixed(2)} | **Created:** ${doc.created_at}\n\n${doc.doc}`,
      }],
    };
  },
);

server.tool(
  "memory_related",
  "Find memories related to a given memory via the relationship graph.",
  {
    id: z.number().describe("Document ID to find relations for"),
    limit: z.number().optional().default(10).describe("Max results"),
  },
  async ({ id, limit }) => {
    const related = store.getRelated(id, limit);
    if (related.length === 0) {
      return {
        content: [{ type: "text" as const, text: `No related memories found for #${id}.` }],
      };
    }

    const text = related
      .map((r: any) => `- [${r.relation_type}] **${r.title}** (id: ${r.id}, weight: ${r.weight.toFixed(2)})`)
      .join("\n");

    return {
      content: [{ type: "text" as const, text }],
    };
  },
);

server.tool(
  "memory_timeline",
  "Get recent memories in reverse chronological order.",
  {
    limit: z.number().optional().default(20).describe("Max results"),
    content_type: z.enum(["decision", "preference", "antipattern", "observation", "research", "project", "handoff", "note"]).optional().describe("Filter by type"),
  },
  async ({ limit, content_type }) => {
    const docs = store.timeline(limit, content_type as ContentType | undefined);
    if (docs.length === 0) {
      return {
        content: [{ type: "text" as const, text: "No memories found." }],
      };
    }

    const text = docs
      .map((d: any) => `- **${d.title}** [${d.content_type}] (id: ${d.id}, ${d.created_at})`)
      .join("\n");

    return {
      content: [{ type: "text" as const, text }],
    };
  },
);

// ── Mutation Tools ──────────────────────────────────────────────────────────

server.tool(
  "memory_save",
  "Save a new memory. Use this to explicitly store important information — decisions, preferences, observations, project context, or session handoffs.",
  {
    title: z.string().describe("Short descriptive title for the memory"),
    content: z.string().describe("Full content of the memory"),
    content_type: z.enum(["decision", "preference", "antipattern", "observation", "research", "project", "handoff", "note"]).optional().default("note").describe("Type of memory"),
    collection: z.string().optional().default("default").describe("Vault collection namespace"),
    source_client: z.string().optional().describe("Which client is saving this (claude-code, cursor, gemini, etc.)"),
  },
  async ({ title, content, content_type, collection, source_client }) => {
    const docId = store.save({
      title,
      content,
      content_type: content_type as ContentType,
      collection,
      source_client,
    });

    // Embed asynchronously (don't block the response)
    const doc = store.get(docId);
    if (doc) {
      embedDocument(store, doc.hash, title, content).catch((err) =>
        console.error("Embedding failed (non-fatal):", err),
      );
    }

    return {
      content: [{
        type: "text" as const,
        text: `Saved memory #${docId}: "${title}" [${content_type}]`,
      }],
    };
  },
);

server.tool(
  "memory_pin",
  "Pin an important memory to boost its confidence and prevent it from being archived.",
  {
    id: z.number().describe("Document ID to pin"),
  },
  async ({ id }) => {
    const success = store.pin(id);
    return {
      content: [{
        type: "text" as const,
        text: success ? `Pinned memory #${id}.` : `Memory #${id} not found.`,
      }],
    };
  },
);

server.tool(
  "memory_forget",
  "Soft-delete a memory. The memory is marked as invalidated but not physically removed.",
  {
    id: z.number().describe("Document ID to forget"),
  },
  async ({ id }) => {
    const success = store.forget(id);
    return {
      content: [{
        type: "text" as const,
        text: success ? `Forgot memory #${id}.` : `Memory #${id} not found or already forgotten.`,
      }],
    };
  },
);

// ── Lifecycle Tools ─────────────────────────────────────────────────────────

server.tool(
  "memory_status",
  "Get vault health stats — document counts by type, vector coverage, pinned/invalidated counts.",
  {},
  async () => {
    const stats = store.status();
    const byType = (stats.by_type as Array<{ content_type: string; count: number }>)
      .map((t) => `  ${t.content_type}: ${t.count}`)
      .join("\n");

    return {
      content: [{
        type: "text" as const,
        text: `Vault: ${stats.vault_path}\nTotal memories: ${stats.total_documents}\nVectors: ${stats.total_vectors}\nPinned: ${stats.pinned}\nInvalidated: ${stats.invalidated}\n\nBy type:\n${byType}`,
      }],
    };
  },
);

server.tool(
  "memory_sweep",
  "Archive stale memories that have exceeded their half-life. Runs in dry-run mode by default.",
  {
    dry_run: z.boolean().optional().default(true).describe("If true, only show candidates without archiving"),
  },
  async ({ dry_run }) => {
    const result = store.sweep(dry_run);

    if (result.candidates.length === 0) {
      return {
        content: [{ type: "text" as const, text: "No stale memories to archive." }],
      };
    }

    const list = result.candidates
      .map((c) => `- #${c.id} "${c.title}" [${c.content_type}] — ${c.age_days} days old`)
      .join("\n");

    const action = dry_run ? "Would archive" : "Archived";
    return {
      content: [{
        type: "text" as const,
        text: `${action} ${result.candidates.length} memories:\n${list}`,
      }],
    };
  },
);

server.tool(
  "memory_rebuild",
  "Re-embed all documents. Use after upgrading the embedding model.",
  {},
  async () => {
    const docs = store.timeline(1000); // Get all active docs
    let embedded = 0;
    let failed = 0;

    for (const doc of docs) {
      try {
        await embedDocument(store, (doc as any).hash, doc.title, (doc as any).doc);
        embedded++;
      } catch {
        failed++;
      }
    }

    return {
      content: [{
        type: "text" as const,
        text: `Rebuild complete: ${embedded} documents embedded, ${failed} failed.`,
      }],
    };
  },
);

// ── Profile Tools ───────────────────────────────────────────────────────────

server.tool(
  "profile_get",
  "Get a synthesized user profile from stored preferences and decisions. Shows what mnemon knows about the user's habits, preferences, and key decisions.",
  {},
  async () => {
    const preferences = store.timeline(50, "preference" as any);
    const decisions = store.timeline(50, "decision" as any);

    if (preferences.length === 0 && decisions.length === 0) {
      return {
        content: [{ type: "text" as const, text: "No profile data yet. Preferences and decisions will be collected automatically over time." }],
      };
    }

    const sections: string[] = [];

    if (preferences.length > 0) {
      sections.push("## Preferences\n" +
        preferences.map((p: any) => `- **${p.title}**: ${p.doc.slice(0, 200)}`).join("\n"));
    }

    if (decisions.length > 0) {
      sections.push("## Key Decisions\n" +
        decisions.map((d: any) => `- **${d.title}**: ${d.doc.slice(0, 200)}`).join("\n"));
    }

    return {
      content: [{ type: "text" as const, text: sections.join("\n\n") }],
    };
  },
);

server.tool(
  "profile_update",
  "Manually add a fact to the user profile. Saved as a preference memory.",
  {
    title: z.string().describe("Short title for the preference"),
    content: z.string().describe("Description of the preference or habit"),
  },
  async ({ title, content }) => {
    const docId = store.save({
      title,
      content,
      content_type: "preference",
      source_client: "mcp-profile",
    });

    const doc = store.get(docId);
    if (doc) {
      embedDocument(store, doc.hash, title, content).catch(() => {});
    }

    return {
      content: [{ type: "text" as const, text: `Profile updated — saved preference #${docId}: "${title}"` }],
    };
  },
);

// ── Contradiction Check Tool ────────────────────────────────────────────────

server.tool(
  "memory_check_contradictions",
  "Check a memory for contradictions against existing memories. Uses vector similarity + LLM classification to find conflicts.",
  {
    id: z.number().describe("Document ID to check"),
  },
  async ({ id }) => {
    const doc = store.get(id);
    if (!doc) {
      return {
        content: [{ type: "text" as const, text: `Memory #${id} not found.` }],
      };
    }

    const { checkContradictions } = await import("./contradiction.ts");
    const result = await checkContradictions(store, doc.title, (doc as any).doc, id);

    if (result.relationships.length === 0) {
      return {
        content: [{ type: "text" as const, text: `No contradictions found for memory #${id}.` }],
      };
    }

    const lines = result.relationships
      .map((r) => `- #${r.docId} "${r.title}" → **${r.relationship}**`)
      .join("\n");

    return {
      content: [{
        type: "text" as const,
        text: `Contradiction check for #${id} "${doc.title}":\n${lines}\n\n${result.decayed} memories had their confidence decayed.`,
      }],
    };
  },
);

// ── Start Server ────────────────────────────────────────────────────────────

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("mnemon MCP server running on stdio");
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
