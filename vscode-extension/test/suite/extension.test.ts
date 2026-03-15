/**
 * Integration tests for the NeuralMemory VS Code extension.
 *
 * Runs inside the VS Code Extension Development Host via @vscode/test-electron.
 */

import * as assert from "assert";
import * as vscode from "vscode";

suite("Extension", () => {
  test("extension should be present", () => {
    const ext = vscode.extensions.getExtension("pugbrain.pugbrain");
    assert.ok(ext, "Extension not found in registry");
  });

  test("all commands should be registered", async () => {
    const allCommands = await vscode.commands.getCommands(true);

    const expected = [
      "pugbrain.encode",
      "pugbrain.encodeInput",
      "pugbrain.recall",
      "pugbrain.openGraph",
      "pugbrain.switchBrain",
      "pugbrain.createBrain",
      "pugbrain.refreshMemories",
      "pugbrain.startServer",
      "pugbrain.connectServer",
      "pugbrain.recallFromTree",
    ];

    for (const cmd of expected) {
      assert.ok(
        allCommands.includes(cmd),
        `Command "${cmd}" not registered`,
      );
    }
  });

  test("configuration defaults should be set", () => {
    const config = vscode.workspace.getConfiguration("pugbrain");

    assert.strictEqual(config.get("pythonPath"), "python");
    assert.strictEqual(config.get("autoStart"), false);
    assert.strictEqual(config.get("serverUrl"), "http://127.0.0.1:8000");
    assert.strictEqual(config.get("graphNodeLimit"), 1000);
    assert.strictEqual(config.get("codeLensEnabled"), true);

    const triggers = config.get<string[]>("commentTriggers");
    assert.ok(Array.isArray(triggers));
    assert.ok(triggers!.includes("remember:"));
    assert.ok(triggers!.includes("note:"));
    assert.ok(triggers!.includes("decision:"));
    assert.ok(triggers!.includes("todo:"));
  });
});
