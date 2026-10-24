import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import test from "node:test";
import vm from "node:vm";
import { buildSync } from "esbuild";

class Child extends EventEmitter {
  stdout = new EventEmitter();
  stderr = new EventEmitter();
  kill() { return true; }
}

test("reserves the run before awaiting history view creation", async () => {
  const children: Child[] = [];
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
  const Plugin = context.module.exports.default;
  const plugin = new Plugin();
  plugin.refreshHistoryViews = () => {};
  const release: (() => void)[] = [];
  plugin.showHistory = () => new Promise<void>(resolve => release.push(resolve));
  const first = plugin.run({ operation: "health", vaultPath: "/fixture" });
  const second = plugin.run({ operation: "health", vaultPath: "/fixture" });
  release.forEach(resolve => resolve());
  await new Promise<void>(resolve => setImmediate(resolve));
  const spawned = children.length;
  children.forEach(child => child.emit("close", 0, null));
  await Promise.all([first, second]);
  assert.equal(spawned, 1);
});
