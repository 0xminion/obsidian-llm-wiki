import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import test from "node:test";

import { CliClient, cliArguments } from "../src/cli-client.ts";

class FakeChildProcess extends EventEmitter {
  readonly stdout = new EventEmitter();
  readonly stderr = new EventEmitter();
  killCalls: string[] = [];

  kill(signal = "SIGTERM"): boolean {
    this.killCalls.push(signal);
    return true;
  }
}

test("builds JSON-mode arguments for every supported operation", () => {
  assert.deepEqual(
    cliArguments({ operation: "ingest", vaultPath: "/vault", urls: ["https://a.test", "https://b.test"] }),
    ["ingest", "/vault", "--json", "--url", "https://a.test", "--url", "https://b.test"],
  );
  assert.deepEqual(
    cliArguments({ operation: "preview", vaultPath: "/vault", urls: ["https://a.test"] }),
    ["ingest", "/vault", "--preview", "--json", "--url", "https://a.test"],
  );
  assert.deepEqual(cliArguments({ operation: "health", vaultPath: "/vault" }), ["health", "/vault", "--json"]);
  assert.deepEqual(cliArguments({ operation: "fix", vaultPath: "/vault" }), ["fix", "/vault", "--json", "--dry-run"]);
  assert.deepEqual(cliArguments({ operation: "query", vaultPath: "/vault", query: "What changed?" }), [
    "query", "/vault", "--json", "--ask", "What changed?",
  ]);
});

test("streams parsed newline-delimited JSON events across arbitrary chunks", async () => {
  const child = new FakeChildProcess();
  const received: unknown[] = [];
  const client = new CliClient({ spawn: () => child });

  const run = client.start(
    { operation: "health", vaultPath: "/vault" },
    (event) => received.push(event),
  );

  child.stdout.emit("data", Buffer.from('{"type":"start"}\n{"type":"result","count":'));
  child.stdout.emit("data", Buffer.from('2}\n'));
  child.emit("close", 0, null);

  const result = await run.completed;
  assert.equal(result.exitCode, 0);
  assert.deepEqual(received, [
    { kind: "event", operation: "health", event: { type: "start" } },
    { kind: "event", operation: "health", event: { type: "result", count: 2 } },
  ]);
});

test("reports malformed output without dropping subsequent structured events", async () => {
  const child = new FakeChildProcess();
  const received: unknown[] = [];
  const client = new CliClient({ spawn: () => child });
  const run = client.start({ operation: "health", vaultPath: "/vault" }, (event) => received.push(event));

  child.stdout.emit("data", Buffer.from("not json\n{\"type\":\"result\"}\n"));
  child.emit("close", 1, null);

  const result = await run.completed;
  assert.equal(result.exitCode, 1);
  assert.deepEqual(received, [
    { kind: "malformed", operation: "health", line: "not json" },
    { kind: "event", operation: "health", event: { type: "result" } },
  ]);
});

test("cancels a running CLI process exactly once", () => {
  const child = new FakeChildProcess();
  const client = new CliClient({ spawn: () => child });
  const run = client.start({ operation: "health", vaultPath: "/vault" }, () => {});

  run.cancel();
  run.cancel();

  assert.deepEqual(child.killCalls, ["SIGTERM"]);
  child.emit("close", null, "SIGTERM");
});
