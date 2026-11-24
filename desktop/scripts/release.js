// Publica a versão do package.json a partir deste PC, sem gastar minutos do GitHub Actions:
//   npm run dist                         -> monta o .zip portátil, o pacote de atualização e o módulo da Central
//   npm run installer                    -> (opcional) instalador NSIS, só para as versões até a 1.3.0
//   npm run release -- notas.md          -> confere, envia o main e a tag e publica o Release
//   npm run verify-release               -> só confere o Release publicado
// O Release nasce como rascunho e só é publicado com todos os arquivos enviados e conferidos (tamanho e SHA-256).
// Token: GH_TOKEN ou, sem ele, o login do GitHub salvo no git (Credential Manager).
const { execFileSync } = require("child_process");
const crypto = require("crypto");
const fs = require("fs");
const https = require("https");
const path = require("path");

const DESK = path.join(__dirname, ".."), ROOT = path.join(DESK, "..");
const V = require("../package.json").version, TAG = "v" + V;
const REPO = "Ruiso-Outside/ArasakaMikoshi", API = `https://api.github.com/repos/${REPO}`;
const fail = (m) => { console.error("✗ " + m); process.exit(1); };
const git = (...a) => execFileSync("git", a, { cwd: ROOT, encoding: "utf8", env: { ...process.env, GIT_TERMINAL_PROMPT: "0", GCM_INTERACTIVE: "never" } }).trim();

const dist = (f) => path.join(DESK, "dist", f), nsis = (f) => path.join(DESK, "dist-nsis", f);
const NEEDED = [dist(`LLM-Local-${V}-win-x64.zip`), dist(`app-patch-${V}.zip`), dist("patch.json"), dist(`piphin-module-llm-${V}.zip`)];
const INSTALLER = [nsis(`LLM-Local-Setup-${V}.exe`), nsis(`LLM-Local-Setup-${V}.exe.blockmap`), nsis("latest.yml")];
// o instalador entra completo ou não entra: o latest.yml sem o .exe quebraria as versões antigas
const files = () => [...NEEDED, ...(INSTALLER.every((f) => fs.existsSync(f)) ? INSTALLER : [])];
const sha256 = (f) => crypto.createHash("sha256").update(fs.readFileSync(f)).digest("hex");

function token() {
  if (process.env.GH_TOKEN) return process.env.GH_TOKEN;
  const cred = execFileSync("git", ["credential", "fill"], { input: "protocol=https\nhost=github.com\n\n", encoding: "utf8", env: { ...process.env, GCM_INTERACTIVE: "never", GIT_TERMINAL_PROMPT: "0" } });
  const t = (cred.match(/^password=(.*)$/m) || [])[1];
  if (!t) fail("sem login do GitHub: defina GH_TOKEN ou entre no GitHub pelo git");
  return t.trim();
}

function checkLocal() {
  if (git("status", "--porcelain")) fail("há arquivos alterados sem commit");
  if (git("branch", "--show-current") !== "main") fail("publique a partir do main");
  git("fetch", "-q", "origin", "main", "--tags");
  if (git("log", "--oneline", "HEAD..origin/main")) fail("o GitHub tem commits que não estão aqui: faça git pull");
  if (require("../package-lock.json").version !== V) fail("package-lock.json em outra versão: npm install --package-lock-only");
  // um workflow que roda sozinho gastaria créditos a cada publicação
  const wf = path.join(ROOT, ".github", "workflows");
  for (const f of fs.existsSync(wf) ? fs.readdirSync(wf) : []) {
    const on = (/^on:\n([\s\S]*?)^\S/m.exec(fs.readFileSync(path.join(wf, f), "utf8").replace(/\r\n/g, "\n")) || [])[1] || "";
    if (/^ {2}(push|pull_request|schedule|release|create):/m.test(on)) fail(`.github/workflows/${f} roda sozinho no GitHub`);
  }
  for (const f of NEEDED) if (!fs.existsSync(f)) fail(`falta ${path.relative(DESK, f)} (rode npm run dist)`);
  const patch = JSON.parse(fs.readFileSync(dist("patch.json"), "utf8"));
  const data = fs.readFileSync(dist(patch.file));
  if (patch.version !== V || patch.size !== data.length || patch.sha512 !== crypto.createHash("sha512").update(data).digest("base64")) fail("patch.json não confere com o pacote: rode npm run dist");
}

async function verify() {
  for (const f of files()) {
    const n = path.basename(f);
    const r = await fetch(`https://github.com/${REPO}/releases/download/${TAG}/${n}?t=${Date.now()}`, { method: "HEAD", redirect: "follow" });
    if (!r.ok || Number(r.headers.get("content-length")) !== fs.statSync(f).size) fail(`${n} não baixa certo (HTTP ${r.status})`);
  }
  const p = await (await fetch(`https://github.com/${REPO}/releases/latest/download/patch.json?t=${Date.now()}`, { redirect: "follow", cache: "no-store" })).json();
  if (p.version !== V) fail(`o último Release ainda é o ${p.version}`);
  console.log(`✓ ${TAG} publicado e baixando: ${files().map((f) => path.basename(f)).join(", ")}`);
}

async function publish() {
  checkLocal();
  const headers = { Authorization: "Bearer " + token(), Accept: "application/vnd.github+json", "User-Agent": "mikoshi-release" };
  const api = async (method, url, body) => {
    const r = await fetch(url, { method, headers, body: body && JSON.stringify(body) });
    const t = await r.text();
    if (!r.ok) throw new Error(`${method} ${url.replace(API, "")}: HTTP ${r.status} ${t.slice(0, 200)}`);
    return t ? JSON.parse(t) : null;
  };
  // https em fluxo: o fetch do Node 26 caía no meio de envios ("terminated")
  const upload = (url, f) => new Promise((ok, no) => {
    const size = fs.statSync(f).size;
    const req = https.request(new URL(url.replace(/\{.*\}$/, "") + "?name=" + encodeURIComponent(path.basename(f))),
      { method: "POST", headers: { ...headers, "Content-Type": "application/octet-stream", "Content-Length": size } }, (res) => {
        let t = ""; res.on("data", (c) => (t += c));
        res.on("end", () => (res.statusCode < 300 ? ok(JSON.parse(t)) : no(new Error(`HTTP ${res.statusCode} ${t.slice(0, 200)}`))));
      });
    req.on("error", no);
    fs.createReadStream(f).on("error", no).pipe(req);
  });
  const find = async () => (await api("GET", `${API}/releases?per_page=30`)).find((r) => r.tag_name === TAG);
  let rel = await find();
  if (rel && !rel.draft) fail(`o Release ${TAG} já está publicado`);

  git("push", "-q", "origin", "main");
  if (!git("tag", "--list", TAG)) git("tag", "-a", TAG, "-m", `Arasaka Mikoshi ${TAG}`);
  if (git("rev-parse", TAG + "^{commit}") !== git("rev-parse", "HEAD")) fail(`a tag ${TAG} aponta para outro commit`);
  git("push", "-q", "origin", TAG);
  console.log(`✓ main e ${TAG} enviados`);

  const notes = process.argv.slice(2).find((a) => !a.startsWith("--"));
  if (!rel) rel = await api("POST", `${API}/releases`, { tag_name: TAG, name: V, body: notes ? fs.readFileSync(notes, "utf8") : "", draft: true });
  for (const f of files()) {
    const name = path.basename(f), want = "sha256:" + sha256(f);
    for (let i = 1; ; i++) {
      try {
        const old = (await api("GET", `${API}/releases/${rel.id}/assets?per_page=100`)).find((a) => a.name === name);
        if (old && old.state === "uploaded" && old.digest === want) break;
        if (old) await api("DELETE", `${API}/releases/assets/${old.id}`);
        const a = await upload(rel.upload_url, f);
        if (a.state !== "uploaded" || a.size !== fs.statSync(f).size || (a.digest && a.digest !== want)) throw new Error("envio incompleto");
        break;
      } catch (e) {
        if (i >= 3) throw new Error(`${name}: ${e.message}`);
        console.log(`  … ${name}: ${e.message}; tentando de novo`);
        await new Promise((r) => setTimeout(r, 3000 * i));
      }
    }
    console.log(`✓ enviado ${name}`);
  }
  await api("PATCH", `${API}/releases/${rel.id}`, { draft: false, make_latest: "true" });
  await new Promise((r) => setTimeout(r, 4000));
  await verify();
}

(process.argv.includes("--verify") ? verify() : publish()).catch((e) => fail(e.message));
