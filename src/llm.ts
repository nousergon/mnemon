/**
 * Local LLM abstraction — QMD-query-expansion-1.7B via node-llama-cpp.
 *
 * Used for observation extraction, query expansion, and contradiction detection.
 * Runs on Apple Silicon Metal. Auto-downloads ~1.1GB on first use.
 */

import {
  getLlama,
  type Llama,
  type LlamaModel,
  LlamaChatSession,
} from "node-llama-cpp";

const MODEL_URI =
  "hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf";

let _llama: Llama | null = null;
let _model: LlamaModel | null = null;
let _initPromise: Promise<void> | null = null;

async function ensureModel(): Promise<{ llama: Llama; model: LlamaModel }> {
  if (_llama && _model) return { llama: _llama, model: _model };

  if (!_initPromise) {
    _initPromise = (async () => {
      _llama = await getLlama();
      _model = await _llama.loadModel({ modelPath: MODEL_URI });
    })();
  }

  await _initPromise;
  return { llama: _llama!, model: _model! };
}

/**
 * Generate text from a system prompt + user message using the local 1.7B model.
 */
export async function generate(
  systemPrompt: string,
  userMessage: string,
  maxTokens = 2000,
): Promise<string> {
  const { model } = await ensureModel();
  const context = await model.createContext({ contextSize: 4096 });

  const session = new LlamaChatSession({
    contextSequence: context.getSequence(),
    systemPrompt,
  });

  try {
    const response = await session.prompt(userMessage, {
      maxTokens,
      temperature: 0.3,
    });
    return response;
  } finally {
    await context.dispose();
  }
}
