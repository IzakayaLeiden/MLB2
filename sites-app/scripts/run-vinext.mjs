import { spawn } from "node:child_process";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const cli = join(root, "node_modules", "vinext", "dist", "cli.js");
const child = spawn(process.execPath, [cli, ...process.argv.slice(2)], {
  cwd: root,
  env: { ...process.env, WRANGLER_LOG_PATH: ".wrangler/wrangler.log" },
  stdio: "inherit",
});
child.on("exit", (code) => process.exit(code ?? 1));
