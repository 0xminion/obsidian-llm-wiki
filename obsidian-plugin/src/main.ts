import {
  App,
  ItemView,
  Modal,
  Notice,
  Plugin,
  PluginSettingTab,
  Setting,
  WorkspaceLeaf,
} from "obsidian";

import { CliClient, type CliRun } from "./cli-client.ts";
import type { CliRequest, CliStreamEvent } from "./types.ts";

const VIEW_TYPE = "olw-history";
const MAX_HISTORY_LINES = 250;

interface BridgeSettings {
  executable: string;
}

const DEFAULT_SETTINGS: BridgeSettings = {
  executable: "olw",
};

class TextPromptModal extends Modal {
  constructor(
    app: App,
    private readonly title: string,
    private readonly placeholder: string,
    private readonly onSubmit: (value: string) => void,
  ) {
    super(app);
  }

  onOpen(): void {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: this.title });
    const input = contentEl.createEl("input", {
      attr: { type: "text", placeholder: this.placeholder },
    });
    input.focus();

    const submit = () => {
      const value = input.value.trim();
      if (!value) return;
      this.close();
      this.onSubmit(value);
    };
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") submit();
    });
    new Setting(contentEl).addButton((button) =>
      button.setButtonText("Run").setCta().onClick(submit),
    );
  }

  onClose(): void {
    this.contentEl.empty();
  }
}

class OlwHistoryView extends ItemView {
  constructor(leaf: WorkspaceLeaf, private readonly bridge: ObsidianLlmWikiBridge) {
    super(leaf);
  }

  getViewType(): string {
    return VIEW_TYPE;
  }

  getDisplayText(): string {
    return "OLW Results";
  }

  async onOpen(): Promise<void> {
    this.render();
  }

  render(): void {
    this.contentEl.empty();
    this.contentEl.createEl("h4", { text: "Obsidian LLM Wiki CLI" });
    const cancel = this.contentEl.createEl("button", { text: "Cancel active run" });
    cancel.disabled = !this.bridge.hasActiveRun();
    cancel.addEventListener("click", () => this.bridge.cancelActiveRun());

    const history = this.contentEl.createEl("pre", { cls: "olw-bridge-history" });
    history.setText(this.bridge.historyText());
  }
}

class BridgeSettingsTab extends PluginSettingTab {
  constructor(app: App, private readonly bridge: ObsidianLlmWikiBridge) {
    super(app, bridge);
  }

  display(): void {
    this.containerEl.empty();
    new Setting(this.containerEl)
      .setName("OLW executable")
      .setDesc("Local command to execute. Defaults to `olw` on your PATH.")
      .addText((text) =>
        text
          .setPlaceholder("olw")
          .setValue(this.bridge.settings.executable)
          .onChange(async (value) => {
            this.bridge.settings.executable = value.trim() || DEFAULT_SETTINGS.executable;
            await this.bridge.saveSettings();
          }),
      );
  }
}

/** A local UI shell around `olw --json`; compiler logic remains in the CLI. */
export default class ObsidianLlmWikiBridge extends Plugin {
  settings: BridgeSettings = { ...DEFAULT_SETTINGS };
  private activeRun: CliRun | null = null;
  private readonly history: string[] = [];

  async onload(): Promise<void> {
    await this.loadSettings();
    this.registerView(VIEW_TYPE, (leaf) => new OlwHistoryView(leaf, this));
    this.addSettingTab(new BridgeSettingsTab(this.app, this));

    this.addCommand({
      id: "olw-ingest-url",
      name: "OLW: Ingest URL",
      callback: () => this.promptForUrl("Ingest URL", "https://…", "ingest"),
    });
    this.addCommand({
      id: "olw-preview-url",
      name: "OLW: Preview URL ingestion",
      callback: () => this.promptForUrl("Preview URL ingestion", "https://…", "preview"),
    });
    this.addCommand({
      id: "olw-health",
      name: "OLW: Run health check",
      callback: () => void this.run({ operation: "health", vaultPath: this.vaultPath() }),
    });
    this.addCommand({
      id: "olw-fix-dry-run",
      name: "OLW: Preview maintenance fixes",
      callback: () => void this.run({ operation: "fix", vaultPath: this.vaultPath() }),
    });
    this.addCommand({
      id: "olw-query",
      name: "OLW: Query wiki",
      callback: () => {
        new TextPromptModal(this.app, "Query wiki", "Ask a question", (query) => {
          void this.run({ operation: "query", vaultPath: this.vaultPath(), query });
        }).open();
      },
    });
    this.addCommand({
      id: "olw-show-results",
      name: "OLW: Show result history",
      callback: () => void this.showHistory(),
    });
    this.addCommand({
      id: "olw-cancel",
      name: "OLW: Cancel active CLI run",
      checkCallback: (checking) => {
        if (!this.activeRun) return false;
        if (!checking) this.cancelActiveRun();
        return true;
      },
    });
  }

  onunload(): void {
    this.activeRun?.cancel();
    this.app.workspace.getLeavesOfType(VIEW_TYPE).forEach((leaf) => leaf.detach());
  }

  async loadSettings(): Promise<void> {
    const stored = (await this.loadData()) as Partial<BridgeSettings> | null;
    this.settings = { ...DEFAULT_SETTINGS, ...stored };
  }

  async saveSettings(): Promise<void> {
    await this.saveData(this.settings);
  }

  hasActiveRun(): boolean {
    return this.activeRun !== null;
  }

  historyText(): string {
    return this.history.length ? this.history.join("\n") : "No CLI events yet.";
  }

  cancelActiveRun(): void {
    if (!this.activeRun) {
      new Notice("No OLW CLI run is active.");
      return;
    }
    this.record("Cancellation requested.");
    this.activeRun.cancel();
    new Notice("Cancelling OLW CLI run…");
  }

  private promptForUrl(title: string, placeholder: string, operation: "ingest" | "preview"): void {
    new TextPromptModal(this.app, title, placeholder, (url) => {
      void this.run({ operation, vaultPath: this.vaultPath(), urls: [url] });
    }).open();
  }

  private vaultPath(): string {
    const adapter = this.app.vault.adapter as unknown as { getBasePath?: () => string };
    const basePath = adapter.getBasePath?.();
    if (!basePath) {
      throw new Error("This bridge requires a local desktop vault path.");
    }
    return basePath;
  }

  private async run(request: CliRequest): Promise<void> {
    if (this.activeRun) {
      new Notice("An OLW CLI run is already active. Cancel it before starting another.");
      return;
    }

    await this.showHistory();
    this.record(`$ ${this.settings.executable} ${request.operation} — started`);
    try {
      const client = new CliClient({ executable: this.settings.executable });
      const run = client.start(request, (event) => this.recordEvent(event));
      this.activeRun = run;
      const result = await run.completed;
      const outcome = result.cancelled
        ? "cancelled"
        : result.exitCode === 0
          ? "completed"
          : `failed (exit ${result.exitCode ?? "unknown"})`;
      this.record(`${request.operation} ${outcome}.`);
      new Notice(`OLW ${request.operation} ${outcome}.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.record(`${request.operation} failed to start: ${message}`);
      new Notice(`OLW failed to start: ${message}`);
    } finally {
      this.activeRun = null;
      this.refreshHistoryViews();
    }
  }

  private recordEvent(event: CliStreamEvent): void {
    switch (event.kind) {
      case "event":
        this.record(JSON.stringify(event.event));
        break;
      case "stderr":
        this.record(`[stderr] ${event.line}`);
        break;
      case "malformed":
        this.record(`[non-JSON stdout] ${event.line}`);
        break;
      case "error":
        this.record(`[process error] ${event.message}`);
        break;
    }
  }

  private record(line: string): void {
    this.history.push(line);
    if (this.history.length > MAX_HISTORY_LINES) this.history.splice(0, this.history.length - MAX_HISTORY_LINES);
    this.refreshHistoryViews();
  }

  private refreshHistoryViews(): void {
    for (const leaf of this.app.workspace.getLeavesOfType(VIEW_TYPE)) {
      const view = leaf.view;
      if (view instanceof OlwHistoryView) view.render();
    }
  }

  private async showHistory(): Promise<void> {
    const existing = this.app.workspace.getLeavesOfType(VIEW_TYPE)[0];
    if (existing) {
      await this.app.workspace.revealLeaf(existing);
      return;
    }
    const leaf = this.app.workspace.getRightLeaf(false);
    if (!leaf) return;
    await leaf.setViewState({ type: VIEW_TYPE, active: true });
    await this.app.workspace.revealLeaf(leaf);
  }
}
