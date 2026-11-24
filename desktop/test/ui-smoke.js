// Teste da interface (frontend) com o Electron do projeto, sem GPU e sem modelo de verdade:
// sobe o servidor de testes (tests/run_server.py, com o llama-server falso), abre a página numa janela escondida
// e usa a tela como uma pessoa usaria: escolhe o modelo, manda uma mensagem e lê a resposta, abre as Configurações.
// Falha se a página registrar erro de JavaScript.
//   cd desktop && npx electron test/ui-smoke.js          (no Linux sem tela: xvfb-run -a npx electron ...)
"use strict";
const { app, BrowserWindow } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");
const net = require("net");

const ROOT = path.join(__dirname, "..", "..");
const PY = process.env.PYTHON || (process.platform === "win32" ? "python" : "python3");
const TIMEOUT = 90000;
let server, home, win, failed = false;
const step = msg => console.log("  ✓ " + msg);
const fail = msg => { failed = true; throw new Error(msg); };

const freePort = () => new Promise((res, rej) => {
  // o motor falso usa a porta seguinte: procura duas livres em sequência
  const s = net.createServer().listen(0, "127.0.0.1", () => {
    const p = s.address().port; s.close(() => {
      const t = net.createServer().once("error", () => freePort().then(res, rej)).listen(p + 1, "127.0.0.1", () => t.close(() => res(p)));
    });
  });
});

async function waitFor(fn, what, ms = 20000) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    try { const v = await fn(); if (v) return v; } catch {}
    await new Promise(r => setTimeout(r, 150));
  }
  fail("demorou demais: " + what);
}

async function main() {
  home = fs.mkdtempSync(path.join(os.tmpdir(), "mikoshi-ui-"));
  const port = await freePort();
  const base = `http://127.0.0.1:${port}`;
  const env = { ...process.env, LLM_HOME: home, LLM_PORT: String(port), PYTHONIOENCODING: "utf-8",
    MIKOSHI_NO_CONSTRUCT: "1", ARASAKA_CONSTRUCT_EDITOR: path.join(home, "sem-construct") };
  delete env.LLM_PARENT_PID; delete env.ELECTRON_RUN_AS_NODE;
  server = spawn(PY, [path.join(ROOT, "tests", "run_server.py")], { env, stdio: ["ignore", "pipe", "pipe"] });
  let log = ""; server.stdout.on("data", d => log += d); server.stderr.on("data", d => log += d);
  server.on("exit", c => { if (c && !failed) console.error(log); });
  await waitFor(async () => (await fetch(base + "/api/status")).ok, "o servidor subir: " + log);
  step("servidor de testes no ar");

  const errors = [];
  win = new BrowserWindow({ show: false, width: 1240, height: 820, webPreferences: { contextIsolation: true, sandbox: true } });
  win.webContents.on("console-message", (e, ...old) => {
    // Electron novo: tudo no evento (level "error"); antigo: (evento, nível 3, mensagem, linha, arquivo)
    const level = e.level ?? old[0], message = e.message ?? old[1], where = `${e.sourceId ?? old[3] ?? ""}:${e.lineNumber ?? old[2] ?? ""}`;
    if (level === "error" || level === 3) errors.push(`${message} (${where})`);
  });
  win.webContents.on("render-process-gone", (e, d) => errors.push("renderer: " + d.reason));
  await win.loadURL(base + "/");
  const js = code => win.webContents.executeJavaScript(code);
  await waitFor(() => js(`document.querySelectorAll("#modelSelect option").length > 1`), "a lista de modelos aparecer");
  step("página carregou com a lista de modelos");

  const models = await js(`[...document.querySelectorAll("#modelSelect option")].map(o => o.value)`);
  if (!models.some(m => /Qwen3-8B/.test(m))) fail("modelo falso não está na lista: " + models.join(", "));
  // escolhe o modelo no seletor do topo e clica em Carregar; espera a tela (não só o servidor) saber que está pronto
  await js(`(() => { const s = document.querySelector("#modelSelect"); s.value = ${JSON.stringify(models.find(m => /Qwen3-8B/.test(m)))};
    s.dispatchEvent(new Event("change")); document.querySelector("#loadBtn").click(); })()`);
  await waitFor(async () => (await (await fetch(base + "/api/status")).json()).engine.state === "ready", "o modelo carregar");
  await waitFor(() => js(`/Qwen3-8B/.test(document.querySelector("#statusPill").innerText) && document.querySelector("#modelSelect").value.includes("Qwen3-8B")`),
    "a tela mostrar o modelo carregado");
  step("modelo escolhido e carregado pelo botão Carregar");

  // a tela atualiza o status sozinha a cada poucos segundos; antes disso o Enviar avisa "Aguarde o modelo terminar de carregar"
  await waitFor(() => js(`S.status?.engine?.state === "ready"`), "a tela saber que o modelo está pronto");
  await js(`(() => { const i = document.querySelector("#input"); i.value = "olá, teste da interface"; i.dispatchEvent(new Event("input"));
    document.querySelector("#sendBtn").click(); })()`);
  const reply = await waitFor(() => js(`(() => { const m = [...document.querySelectorAll(".msg.assistant")].pop();
    const t = m && m.innerText || ""; return /Qwen3-8B/.test(t) && !document.querySelector(".msg.assistant .live") ? t : ""; })()`),
    "a resposta aparecer na tela", 30000);
  if (!(await js(`[...document.querySelectorAll(".msg.user")].some(m => m.innerText.includes("olá, teste da interface"))`)))
    fail("a mensagem enviada não aparece na conversa");
  step("mensagem enviada e resposta exibida (" + reply.split("\n").find(l => /Qwen3/.test(l)).trim().slice(0, 60) + ")");
  await waitFor(() => js(`document.querySelectorAll("#chatList > *").length > 0`), "a conversa aparecer na lista");
  step("conversa salva na barra lateral");

  await js(`document.querySelector('[data-open="settingsModal"]').click()`);
  if (!(await js(`document.querySelector("#settingsModal").classList.contains("open")`))) fail("Configurações não abriu");
  await js(`document.querySelector("#settingsModal [data-close]").dispatchEvent(new MouseEvent("mousedown", { bubbles: true }))`); // o ✕ fecha no mousedown
  if (await js(`document.querySelector("#settingsModal").classList.contains("open")`)) fail("Configurações não fechou");
  step("Configurações abre e fecha");

  await js(`document.querySelector("#newChat").click()`);
  if (await js(`document.querySelectorAll(".msg").length`)) fail("Nova conversa não limpou a tela");
  step("nova conversa");

  if (errors.length) fail("erros de JavaScript na página:\n" + errors.join("\n"));
  step("sem erros de JavaScript");
}

function cleanup() {
  try {
    if (server && server.exitCode === null) {
      if (process.platform === "win32") require("child_process").spawnSync("taskkill", ["/pid", String(server.pid), "/T", "/F"]);
      else server.kill("SIGKILL");
    }
  } catch {}
  try { fs.rmSync(home, { recursive: true, force: true, maxRetries: 5, retryDelay: 300 }); } catch {}
}

app.commandLine.appendSwitch("no-sandbox"); // CI no Linux roda como root/sem namespaces
app.whenReady().then(() => Promise.race([main(), new Promise((_, rej) => setTimeout(() => rej(new Error("tempo esgotado")), TIMEOUT))]))
  .then(() => { console.log("✓ interface ok"); cleanup(); app.exit(0); })
  .catch(async e => {
    console.error("✗ " + e.message);
    try { // o que estava na tela, para entender a falha
      console.error(await win.webContents.executeJavaScript(`JSON.stringify({ status: document.querySelector("#statusPill")?.innerText,
        model: document.querySelector("#modelSelect")?.value, state: typeof S !== "undefined" && { mode: S.mode, generating: S.generating, engine: S.status?.engine?.state, mode_switch: S.status?.settings?.mode_switch, atts: S.attachments?.length }, input: document.querySelector("#input")?.value,
        hint: document.querySelector("#hint")?.innerText, toast: [...document.querySelectorAll(".toast")].map(t => t.innerText),
        messages: [...document.querySelectorAll(".msg")].map(m => m.innerText.slice(0, 200)) }, null, 1)`));
    } catch {}
    cleanup(); app.exit(1);
  });
app.on("window-all-closed", () => {});
