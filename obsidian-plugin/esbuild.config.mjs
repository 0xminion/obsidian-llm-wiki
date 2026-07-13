import esbuild from "esbuild";

const production = process.argv.includes("--production");
const context = await esbuild.context({
  entryPoints: ["src/main.ts"],
  bundle: true,
  external: ["obsidian", "electron", "node:child_process"],
  format: "cjs",
  target: "es2022",
  logLevel: "info",
  minify: production,
  outfile: "main.js",
  sourcemap: production ? false : "inline",
});

if (production) {
  await context.rebuild();
  await context.dispose();
} else {
  await context.watch();
}
