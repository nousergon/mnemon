/**
 * Embedding pipeline — EmbeddingGemma-300M via node-llama-cpp.
 *
 * Auto-downloads the GGUF model from HuggingFace on first use.
 * Runs on Apple Silicon Metal for GPU acceleration.
 */

import {
  getLlama,
  type Llama,
  type LlamaModel,
  type LlamaEmbeddingContext,
} from "node-llama-cpp";

const MODEL_URI =
  "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";

const VECTOR_DIM = 768;

let _llama: Llama | null = null;
let _model: LlamaModel | null = null;
let _ctx: LlamaEmbeddingContext | null = null;
let _initPromise: Promise<void> | null = null;

/**
 * Initialize the embedding model (lazy, singleton).
 * Downloads ~314MB on first run.
 */
async function ensureModel(): Promise<LlamaEmbeddingContext> {
  if (_ctx) return _ctx;

  if (!_initPromise) {
    _initPromise = (async () => {
      _llama = await getLlama();
      _model = await _llama.loadModel({ modelPath: MODEL_URI });
      _ctx = await _model.createEmbeddingContext({
        contextSize: 2048,
      });
    })();
  }

  await _initPromise;
  return _ctx!;
}

/**
 * Embed a single text string. Returns a Float32Array of dimension 768.
 */
export async function embed(text: string): Promise<Float32Array> {
  const ctx = await ensureModel();
  const result = await ctx.getEmbeddingFor(text);
  return new Float32Array(result.vector);
}

/**
 * Embed multiple texts in batch.
 */
export async function embedBatch(texts: string[]): Promise<Float32Array[]> {
  const results: Float32Array[] = [];
  for (const text of texts) {
    results.push(await embed(text));
  }
  return results;
}

/**
 * Split a document into fragments for embedding.
 * Embeds the full document plus individual sections.
 */
export function fragmentize(title: string, content: string): Array<{ seq: number; text: string }> {
  const fragments: Array<{ seq: number; text: string }> = [];

  // seq=0: full document (title + content, truncated)
  const fullText = `title: ${title} | text: ${content}`.slice(0, 2000);
  fragments.push({ seq: 0, text: fullText });

  // Split by markdown headers or double newlines
  const sections = content.split(/(?=^#{1,3}\s)/m).filter((s) => s.trim().length > 50);

  for (let i = 0; i < Math.min(sections.length, 5); i++) {
    fragments.push({
      seq: i + 1,
      text: `title: ${title} | section: ${sections[i]!.trim().slice(0, 1000)}`,
    });
  }

  return fragments;
}

/**
 * Embed and store all fragments for a document.
 */
export async function embedDocument(
  store: { saveEmbedding: (hash: string, seq: number, emb: Float32Array) => void },
  contentHash: string,
  title: string,
  content: string,
): Promise<number> {
  const fragments = fragmentize(title, content);
  let count = 0;

  for (const frag of fragments) {
    const emb = await embed(frag.text);
    store.saveEmbedding(contentHash, frag.seq, emb);
    count++;
  }

  return count;
}

export { VECTOR_DIM };
