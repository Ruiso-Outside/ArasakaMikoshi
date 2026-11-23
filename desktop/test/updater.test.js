// Testes da atualização automática (node --test desktop/test). Sem Electron e sem rede.
"use strict";
const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const u = require("../updater-core");

const read = f => fs.readFileSync(f, "utf8");

function fakeInstall(version) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mikoshi-upd-"));
  const res = path.join(root, "resources"), staging = path.join(res, `_update_${version}`);
  for (const [base, v] of [[res, "1.0.0"], [staging, version]]) {
    fs.mkdirSync(path.join(base, "app"), { recursive: true });
    fs.mkdirSync(path.join(base, "server"), { recursive: true });
    fs.writeFileSync(path.join(base, "app", "package.json"), JSON.stringify({ version: v }));
    fs.writeFileSync(path.join(base, "app", "main.js"), `// ${v}`);
    fs.writeFileSync(path.join(base, "app", "updater-core.js"), `// ${v}`);
    fs.writeFileSync(path.join(base, "server", "server.py"), `# ${v}`);
  }
  return { root, res, staging };
}

test("cmpVer compara número a número, não como texto", () => {
  assert.ok(u.cmpVer("1.10.0", "1.9.9") > 0);
  assert.ok(u.cmpVer("1.31.1", "1.32.0") < 0);
  assert.strictEqual(u.cmpVer("1.2", "1.2.0"), 0);
});

test("planUpdate: mais nova baixa, igual ou mais velha não, Electron diferente pede o pacote completo", () => {
  const m = { version: "1.32.0", electron: "44.4.5", file: "app-patch-1.32.0.zip", sha512: "x" };
  assert.strictEqual(u.planUpdate(m, "1.31.1", "44.4.5"), "download");
  assert.strictEqual(u.planUpdate(m, "1.32.0", "44.4.5"), "latest");
  assert.strictEqual(u.planUpdate(m, "1.40.0", "44.4.5"), "latest");
  assert.strictEqual(u.planUpdate(m, "1.31.1", "45.0.0"), "manual");
  assert.strictEqual(u.planUpdate(null, "1.31.1", "44.4.5"), "latest");
});

test("planUpdate recusa um patch.json com nome de arquivo estranho", () => {
  const m = { version: "9.0.0", electron: "44.4.5", file: "../../evil.zip", sha512: "x" };
  assert.throws(() => u.planUpdate(m, "1.0.0", "44.4.5"), /inválido/);
  assert.throws(() => u.planUpdate({ ...m, file: "app-patch-9.0.0.zip", sha512: "" }, "1.0.0", "44.4.5"), /inválido/);
});

test("sha512 em base64, o mesmo formato do patch.json", () => {
  assert.strictEqual(u.sha512("abc"), "3a81oZNherrMQXNJriBBMRLm+k6JqX6iCp7u5ktV05ohkpkqJ0/BqDa6PCOj/uu9RU1EI2Q86A4qmslPpUyknw==");
});

test("checkStaged aceita o pacote certo e recusa versão errada ou arquivo faltando", () => {
  const { root, staging } = fakeInstall("1.32.0");
  try {
    u.checkStaged(staging, "1.32.0");
    assert.throws(() => u.checkStaged(staging, "1.33.0"), /incompleto/);
    fs.rmSync(path.join(staging, "app", "updater-core.js"));
    assert.throws(() => u.checkStaged(staging, "1.32.0"), /incompleto/);
    fs.rmSync(path.join(staging, "app", "package.json"));
    assert.throws(() => u.checkStaged(staging, "1.32.0"), /incompleto/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("swapFolders troca app e server e guarda a versão anterior", async () => {
  const { root, res, staging } = fakeInstall("1.32.0");
  try {
    await u.swapFolders(res, staging);
    assert.strictEqual(read(path.join(res, "app", "main.js")), "// 1.32.0");
    assert.strictEqual(read(path.join(res, "server", "server.py")), "# 1.32.0");
    assert.strictEqual(read(path.join(res, "_old_app", "main.js")), "// 1.0.0");
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("swapFolders volta tudo à versão anterior se um passo falhar", async () => {
  const { root, res, staging } = fakeInstall("1.32.0");
  try {
    let calls = 0;
    const rename = (a, b) => { // o 4º passo (server novo no lugar) falha, como uma pasta presa pelo antivírus
      if (++calls === 4) throw Object.assign(new Error("acesso negado"), { code: "ENOENT" });
      fs.renameSync(a, b);
    };
    await assert.rejects(u.swapFolders(res, staging, { rename }), /acesso negado/);
    assert.strictEqual(read(path.join(res, "app", "main.js")), "// 1.0.0");
    assert.strictEqual(read(path.join(res, "server", "server.py")), "# 1.0.0");
    assert.strictEqual(read(path.join(staging, "app", "main.js")), "// 1.32.0"); // o pacote continua lá
    assert.ok(!fs.existsSync(path.join(res, "_old_app")));
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("renameRetry espera a pasta ser liberada (EBUSY) e desiste em outros erros", async () => {
  let n = 0;
  await u.renameRetry("a", "b", { wait: 1, rename: () => { if (++n < 3) throw Object.assign(new Error("ocupada"), { code: "EBUSY" }); } });
  assert.strictEqual(n, 3);
  await assert.rejects(u.renameRetry("a", "b", { wait: 1, rename: () => { throw Object.assign(new Error("sumiu"), { code: "ENOENT" }); } }), /sumiu/);
  await assert.rejects(u.renameRetry("a", "b", { wait: 1, tries: 2, rename: () => { throw Object.assign(new Error("presa"), { code: "EPERM" }); } }), /presa/);
});
