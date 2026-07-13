export type CliOperation = "ingest" | "preview" | "health" | "fix" | "query";

/** A single structured event emitted by `olw --json`. */
export type OlwJsonEvent = Record<string, unknown>;

export interface CliRequest {
  operation: CliOperation;
  vaultPath: string;
  urls?: string[];
  query?: string;
  /** Runs `olw fix --apply`; omitted means the non-destructive dry run. */
  applyFixes?: boolean;
}

export type CliStreamEvent =
  | { kind: "event"; operation: CliOperation; event: OlwJsonEvent }
  | { kind: "malformed"; operation: CliOperation; line: string }
  | { kind: "stderr"; operation: CliOperation; line: string }
  | { kind: "error"; operation: CliOperation; message: string };

export interface CliRunResult {
  exitCode: number | null;
  signal: string | null;
  cancelled: boolean;
}
