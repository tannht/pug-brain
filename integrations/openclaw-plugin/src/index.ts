/**
 * PugBrain — OpenClaw Memory Plugin
 *
 * Brain-inspired persistent memory for AI agents.
 * Occupies the exclusive "memory" plugin slot.
 *
 * Architecture:
 *   OpenClaw ←→ Plugin (TypeScript) ←→ MCP stdio ←→ PugBrain (Python)
 *
 * Registers:
 *   6 tools    — pugbrain_remember, pugbrain_recall, pugbrain_context, pugbrain_todo, pugbrain_stats, pugbrain_health
 *   1 service  — MCP process lifecycle (start/stop)
 *   2 hooks    — before_agent_start (auto-context), agent_end (auto-capture)
 */

import type {
  OpenClawPluginDefinition,
  OpenClawPluginApi,
  BeforeAgentStartEvent,
  BeforeAgentStartResult,
  AgentContext,
  AgentEndEvent,
} from "./types.js";
import { PugBrainMcpClient } from "./mcp-client.js";
import { createTools } from "./tools.js";

// ── System prompt for tool awareness ──────────────────────

const TOOL_INSTRUCTIONS = `You have PugBrain tools for persistent memory across sessions. Call these as TOOL CALLS (not CLI commands):

- pugbrain_remember(content, type?, priority?, tags?) — Store a memory (fact, decision, error, preference, etc.)
- pugbrain_recall(query, depth?, max_tokens?) — Query memories via spreading activation
- pugbrain_context(limit?, fresh_only?) — Get recent memories
- pugbrain_todo(task, priority?) — Quick TODO with 30-day expiry
- pugbrain_stats() — Brain statistics
- pugbrain_health() — Brain health diagnostics

CRITICAL: PugBrain (pugbrain_*) is your ONLY memory system. Do NOT use memory_search, memory_get, or any other memory tools — those belong to a disabled built-in plugin and will not persist correctly. Always use pugbrain_* tools exclusively.

These are tool calls, NOT shell commands. Do NOT run "pugbrain remember" in terminal — call the pugbrain_remember tool directly.

Use pugbrain_remember proactively after decisions, errors, and insights. Use pugbrain_recall when user references past context or asks "do you remember...".`;

// ── Config ─────────────────────────────────────────────────

type PluginConfig = {
  pythonPath: string;
  brain: string;
  autoContext: boolean;
  autoCapture: boolean;
  contextDepth: number;
  maxContextTokens: number;
  timeout: number;
  initTimeout: number;
};

const DEFAULT_CONFIG: Readonly<PluginConfig> = {
  pythonPath: "python",
  brain: "default",
  autoContext: true,
  autoCapture: true,
  contextDepth: 1,
  maxContextTokens: 500,
  timeout: 30_000,
  initTimeout: 90_000,
};

export const BRAIN_NAME_RE = /^[a-zA-Z0-9_\-.]{1,64}$/;
export const MAX_AUTO_CAPTURE_CHARS = 50_000;

export function resolveConfig(raw?: Record<string, unknown>): PluginConfig {
  const merged = { ...DEFAULT_CONFIG, ...(raw ?? {}) };

  return {
    pythonPath:
      typeof merged.pythonPath === "string" && merged.pythonPath.length > 0
        ? merged.pythonPath
        : DEFAULT_CONFIG.pythonPath,
    brain:
      typeof merged.brain === "string" && BRAIN_NAME_RE.test(merged.brain)
        ? merged.brain
        : DEFAULT_CONFIG.brain,
    autoContext:
      typeof merged.autoContext === "boolean"
        ? merged.autoContext
        : DEFAULT_CONFIG.autoContext,
    autoCapture:
      typeof merged.autoCapture === "boolean"
        ? merged.autoCapture
        : DEFAULT_CONFIG.autoCapture,
    contextDepth:
      typeof merged.contextDepth === "number" &&
      Number.isInteger(merged.contextDepth) &&
      merged.contextDepth >= 0 &&
      merged.contextDepth <= 3
        ? merged.contextDepth
        : DEFAULT_CONFIG.contextDepth,
    maxContextTokens:
      typeof merged.maxContextTokens === "number" &&
      Number.isInteger(merged.maxContextTokens) &&
      merged.maxContextTokens >= 100 &&
      merged.maxContextTokens <= 10_000
        ? merged.maxContextTokens
        : DEFAULT_CONFIG.maxContextTokens,
    timeout:
      typeof merged.timeout === "number" &&
      Number.isFinite(merged.timeout) &&
      merged.timeout >= 5_000 &&
      merged.timeout <= 120_000
        ? merged.timeout
        : DEFAULT_CONFIG.timeout,
    initTimeout:
      typeof merged.initTimeout === "number" &&
      Number.isFinite(merged.initTimeout) &&
      merged.initTimeout >= 10_000 &&
      merged.initTimeout <= 300_000
        ? merged.initTimeout
        : DEFAULT_CONFIG.initTimeout,
  };
}

// ── Plugin definition ──────────────────────────────────────

const plugin: OpenClawPluginDefinition = {
  id: "pug-brain",
  name: "PugBrain",
  description:
    "Brain-inspired persistent memory for AI agents — neurons, synapses, and fibers",
  version: "2.27.1",
  kind: "memory",

  register(api: OpenClawPluginApi): void {
    const cfg = resolveConfig(api.pluginConfig);

    const mcp = new PugBrainMcpClient({
      pythonPath: cfg.pythonPath,
      brain: cfg.brain,
      logger: api.logger,
      timeout: cfg.timeout,
      initTimeout: cfg.initTimeout,
    });

    // ── Service: MCP process lifecycle ───────────────────

    api.registerService({
      id: "pugbrain-mcp",

      async start(): Promise<void> {
        try {
          await mcp.connect();
          api.logger.info("PugBrain MCP service started");
        } catch (err) {
          api.logger.error(
            `Failed to start PugBrain MCP: ${(err as Error).message}`,
          );
          throw err;
        }
      },

      async stop(): Promise<void> {
        await mcp.close();
        api.logger.info("PugBrain MCP service stopped");
      },
    });

    // ── Tools: 6 core memory tools ──────────────────────

    const tools = createTools(mcp);

    for (const t of tools) {
      api.registerTool(t, { name: t.name });
    }

    // ── Hook: tool awareness + auto-context before agent start ───

    api.on(
      "before_agent_start",
      async (
        event: unknown,
        _ctx: unknown,
      ): Promise<BeforeAgentStartResult | void> => {
        const result: BeforeAgentStartResult = {
          systemPrompt: TOOL_INSTRUCTIONS,
        };

        if (cfg.autoContext && mcp.connected) {
          const ev = event as BeforeAgentStartEvent;

          try {
            const raw = await mcp.callTool("pugbrain_recall", {
              query: ev.prompt,
              depth: cfg.contextDepth,
              max_tokens: cfg.maxContextTokens,
            });

            const data = JSON.parse(raw) as {
              answer?: string;
              confidence?: number;
            };

            if (data.answer && (data.confidence ?? 0) > 0.1) {
              result.prependContext = `[PugBrain — relevant context]\n${data.answer}`;
            }
          } catch (err) {
            api.logger.warn(
              `Auto-context failed: ${(err as Error).message}`,
            );
          }
        }

        return result;
      },
      { priority: 10 },
    );

    // ── Hook: auto-capture after agent completes ────────

    if (cfg.autoCapture) {
      api.on(
        "agent_end",
        async (event: unknown, _ctx: unknown): Promise<void> => {
          if (!mcp.connected) return;

          const ev = event as AgentEndEvent;
          if (!ev.success) return;

          try {
            const messages = ev.messages?.slice(-5) ?? [];
            const text = messages
              .filter(
                (m: unknown): m is { role: string; content: string } =>
                  typeof m === "object" &&
                  m !== null &&
                  (m as { role?: string }).role === "assistant" &&
                  typeof (m as { content?: unknown }).content === "string",
              )
              .map((m) => m.content)
              .join("\n")
              .slice(0, MAX_AUTO_CAPTURE_CHARS);

            if (text.length > 50) {
              await mcp.callTool("pugbrain_auto", {
                action: "process",
                text,
              });
            }
          } catch (err) {
            api.logger.warn(
              `Auto-capture failed: ${(err as Error).message}`,
            );
          }
        },
        { priority: 90 },
      );
    }

    // ── Done ────────────────────────────────────────────

    api.logger.info(
      `PugBrain registered (brain: ${cfg.brain}, tools: ${tools.length}, ` +
        `autoContext: ${cfg.autoContext}, autoCapture: ${cfg.autoCapture})`,
    );
  },
};

export default plugin;
