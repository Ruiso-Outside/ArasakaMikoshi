// Partes da atualização automática que não dependem do Electron (testadas em desktop/test/updater.test.js):
// comparar versões, conferir o pacote baixado e trocar as pastas do app com volta à versão anterior se falhar.
"use strict";
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

/** Compara "1.31.1" com "1.32.0" (negativo, zero ou positivo). Partes que faltam valem 0. */
function cmpVer(a, b) {
  const pa = String(a).split(".").map(Number), pb = String(b).split(".").map(Number);
  for (let i = 0; i < 3; i++) if ((pa[i] || 0) !== (pb[i] || 0)) return (pa[i] || 0) - (pb[i] || 0);
  return 0;
}

/** O que fazer com o patch.json do Release: "latest", "manual" (o Electron muda: instalador completo) ou "download". */
function planUpdate(manifest, current, electron) {
  if (!manifest || !manifest.version || cmpVer(manifest.version, current) <= 0) return "latest";
  if (manifest.electron !== electron) return "manual";
  if (!manifest.file || !/^app-patch-[\w.-]+\.zip$/.test(manifest.file) || !manifest.sha512) throw new Error("patch.json inválido.");
  return "download";
}

const sha512 = data => crypto.createHash("sha512").update(data).digest("base64");

/** Confere a pasta extraída: a versão tem que ser a anunciada e os arquivos principais têm que existir. */
function checkStaged(staging, version) {
  let pkg;
  try {
    pkg = JSON.parse(fs.readFileSync(path.join(staging, "app", "package.json"), "utf8"));
  } catch {
    throw new Error("o pacote da atualização está incompleto.");
  }
  const needed = [["app", "main.js"], ["app", "updater-core.js"], ["server", "server.py"]];
  if (pkg.version !== version || !needed.every(f => fs.existsSync(path.join(staging, ...f))))
    throw new Error("o pacote da atualização está incompleto.");
}

async function renameRetry(from, to, { rename = fs.renameSync, tries = 20, wait = 1000 } = {}) {
  // o Windows pode segurar a pasta por alguns instantes depois que o servidor fecha
  for (let i = 0; ; i++) {
    try { return rename(from, to); } catch (err) {
      if (i >= tries || !["EPERM", "EBUSY", "EACCES"].includes(err.code)) throw err;
      await new Promise(r => setTimeout(r, wait));
    }
  }
}

/**
 * Troca resources/app e resources/server pelas da pasta staging. Se qualquer passo falhar, desfaz o que já foi
 * feito (na ordem inversa) e lança o erro: o app continua inteiro na versão atual.
 */
async function swapFolders(res, staging, opts = {}) {
  const rename = opts.rename || fs.renameSync;
  const done = [];
  try {
    for (const d of ["app", "server"]) {
      const cur = path.join(res, d), old = path.join(res, `_old_${d}`), next = path.join(staging, d);
      fs.rmSync(old, { recursive: true, force: true });
      await renameRetry(cur, old, { ...opts, rename }); done.push([d, "old"]);
      await renameRetry(next, cur, { ...opts, rename }); done.push([d, "new"]);
    }
  } catch (err) {
    for (const [d, step] of done.reverse()) {
      try {
        if (step === "new") fs.renameSync(path.join(res, d), path.join(staging, d));
        else fs.renameSync(path.join(res, `_old_${d}`), path.join(res, d));
      } catch {}
    }
    throw err;
  }
}

module.exports = { cmpVer, planUpdate, sha512, checkStaged, renameRetry, swapFolders };
