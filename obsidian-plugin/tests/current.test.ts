import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import test from "node:test";
import vm from "node:vm";
import { buildSync } from "esbuild";
import { cliArguments } from "../src/cli-client.ts";

test("current delegates resolution to the validating CLI", () => {
  assert.deepEqual(cliArguments({ operation: "current", vaultPath: "/fixture" }),
    ["current", "/fixture", "--json"]);
});

for (const scenario of ["success", "failure", "unsafe"] as const) {
  test(`current opens only a validated successful entrypoint: ${scenario}`, async () => {
    class Child extends EventEmitter {
      stdout = new EventEmitter(); stderr = new EventEmitter(); kill() { return true; }
    }
    const children: Child[] = [];
    const opened: string[] = [];
    const obsidian = { Plugin: class {}, ItemView: class {}, PluginSettingTab: class {},
      Modal: class {}, Notice: class {}, Setting: class {} };
    const context: any = { module: { exports: {} }, TextDecoder, require: (name: string) => {
      if (name === "obsidian") return obsidian;
      if (name === "node:child_process") return { spawn: () => {
        const child = new Child(); children.push(child); return child;
      } };
      throw new Error(`Unexpected import: ${name}`);
    } };
    const compiled = buildSync({ entryPoints: ["src/main.ts"], bundle: true,
      external: ["obsidian", "node:child_process"], format: "cjs", write: false });
    vm.runInNewContext(compiled.outputFiles[0].text, context);
    const plugin = new context.module.exports.default();
    plugin.showHistory = async () => {};
    plugin.refreshHistoryViews = () => {};
    plugin.app = { workspace: { openLinkText: async (path: string) => opened.push(path) } };
    const running = plugin.run({ operation: "current", vaultPath: "/fixture" });
    await new Promise<void>(resolve => setImmediate(resolve));
    const generation = "a".repeat(32);
    const entrypoint = `04-Wiki/OLW-Publications/generations/${generation}/index.md`;
    children[0].stdout.emit("data", JSON.stringify({ generation,
      entrypoint: scenario === "unsafe" ? "../../outside.md" : entrypoint }) + "\n");
    assert.deepEqual(opened, []);
    children[0].emit("close", scenario === "failure" ? 1 : 0, null);
    await running;
    assert.deepEqual(opened, scenario === "success" ? [entrypoint] : []);
  });
}
