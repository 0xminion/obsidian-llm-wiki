import { spawn as nodeSpawn } from "node:child_process";

import type {
  CliOperation,
  CliRequest,
  CliRunResult,
  CliStreamEvent,
  OlwJsonEvent,
} from "./types.ts";

type DataChunk = string | Uint8Array;

interface DataStream {
  on(event: "data", listener: (chunk: DataChunk) => void): unknown;
}

export interface SpawnedProcess {
  stdout?: DataStream | null;
  stderr?: DataStream | null;
  on(event: "close", listener: (code: number | null, signal: string | null) => void): unknown;
  on(event: "error", listener: (error: Error) => void): unknown;
  kill(signal?: string): boolean;
}

export type SpawnProcess = (
  executable: string,
  args: string[],
  options: { windowsHide: boolean },
) => SpawnedProcess;

export interface CliClientOptions {
  executable?: string;
  spawn?: SpawnProcess;
}

export interface CliRun {
  cancel(): void;
  completed: Promise<CliRunResult>;
}

/** Build argv only; no compiler behavior belongs in this adapter. */
export function cliArguments(request: CliRequest): string[] {
  const base = [request.vaultPath];

  switch (request.operation) {
    case "ingest":
      return ["ingest", ...base, "--json", ...urlArguments(request.urls)];
    case "preview":
      return ["ingest", ...base, "--preview", "--json", ...urlArguments(request.urls)];
    case "health":
      return ["health", ...base, "--json"];
    case "fix":
      return [
        "fix",
        ...base,
        "--json",
        ...(request.applyFixes ? ["--apply"] : ["--dry-run"]),
      ];
    case "query":
      if (!request.query?.trim()) {
        throw new Error("A query request requires a question.");
      }
      return ["query", ...base, "--json", "--ask", request.query];
  }
}

function urlArguments(urls: string[] | undefined): string[] {
  return (urls ?? []).flatMap((url) => ["--url", url]);
}

function toText(chunk: DataChunk): string {
  return typeof chunk === "string" ? chunk : new TextDecoder().decode(chunk);
}

/**
 * Local process adapter for the structured CLI. It forwards events and owns
 * only process lifecycle; parsing, extraction, rendering, and retrieval stay
 * in `olw`.
 */
export class CliClient {
  private readonly executable: string;
  private readonly spawnProcess: SpawnProcess;

  constructor(options: CliClientOptions = {}) {
    this.executable = options.executable?.trim() || "olw";
    this.spawnProcess = options.spawn ?? (nodeSpawn as unknown as SpawnProcess);
  }

  start(request: CliRequest, onEvent: (event: CliStreamEvent) => void): CliRun {
    const operation = request.operation;
    const child = this.spawnProcess(this.executable, cliArguments(request), { windowsHide: true });
    let cancelled = false;
    let settled = false;
    let stdoutBuffer = "";
    let stderrBuffer = "";
    let resolveCompleted: (result: CliRunResult) => void = () => undefined;
    const completed = new Promise<CliRunResult>((resolve) => {
      resolveCompleted = resolve;
    });

    const finish = (result: CliRunResult) => {
      if (settled) return;
      settled = true;
      resolveCompleted(result);
    };

    const emitJsonLine = (line: string) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      try {
        const value: unknown = JSON.parse(trimmed);
        if (!isEvent(value)) throw new Error("JSON event must be an object");
        onEvent({ kind: "event", operation, event: value });
      } catch {
        onEvent({ kind: "malformed", operation, line: trimmed });
      }
    };

    const drainStdout = (final = false) => {
      const lines = stdoutBuffer.split(/\r?\n/);
      stdoutBuffer = final ? "" : (lines.pop() ?? "");
      for (const line of final ? lines : lines) emitJsonLine(line);
    };

    const drainStderr = (final = false) => {
      const lines = stderrBuffer.split(/\r?\n/);
      stderrBuffer = final ? "" : (lines.pop() ?? "");
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed) onEvent({ kind: "stderr", operation, line: trimmed });
      }
    };

    child.stdout?.on("data", (chunk) => {
      stdoutBuffer += toText(chunk);
      drainStdout();
    });
    child.stderr?.on("data", (chunk) => {
      stderrBuffer += toText(chunk);
      drainStderr();
    });
    child.on("error", (error) => {
      onEvent({ kind: "error", operation, message: error.message });
      finish({ exitCode: null, signal: null, cancelled });
    });
    child.on("close", (exitCode, signal) => {
      drainStdout(true);
      drainStderr(true);
      finish({ exitCode, signal, cancelled });
    });

    return {
      completed,
      cancel: () => {
        if (cancelled || settled) return;
        cancelled = true;
        child.kill("SIGTERM");
      },
    };
  }
}

function isEvent(value: unknown): value is OlwJsonEvent {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
