// Publica uma versão pela tag. Quem monta e publica o Release é o GitHub Actions (.github/workflows/release.yml):
// o .zip portátil, o pacote de atualização, o módulo da Central e o instalador, com attestation.
// O mesmo que o "Publicar" da aba Desenvolvedor da Arasaka Central.
//   node scripts/release-check.js            -> confere o repositório, cria a tag vX.Y.Z da versão atual, envia e
//                                               espera o Release ser publicado
//   node scripts/release-check.js --verify   -> só confere o Release publicado (todos os arquivos baixam)
// Sem token: usa o login do GitHub salvo no git. GH_TOKEN (opcional) só evita o limite de consultas da API.
const { execFileSync } = require("child_process");
const path = require("path");
const pkg = require("../package.json");

const { version } = pkg;
const repo = "Ruiso-Outside/ArasakaMikoshi";
const tag = `v${version}`;
const root = path.join(__dirname, "..", "..");
const fail = msg => { console.error("✗ " + msg); process.exit(1); };
const git = (...args) => execFileSync("git", args, { cwd: root, encoding: "utf8" }).trim();
const headers = { "User-Agent": "mikoshi-release", Accept: "application/vnd.github+json",
  ...(process.env.GH_TOKEN ? { Authorization: `Bearer ${process.env.GH_TOKEN}` } : {}) };

const NEEDED = ["patch.json", `app-patch-${version}.zip`, `LLM-Local-${version}-win-x64.zip`, `piphin-module-llm-${version}.zip`,
  "latest.yml", `LLM-Local-Setup-${version}.exe`, `LLM-Local-Setup-${version}.exe.blockmap`];

function pushTag() {
  if (git("status", "--porcelain")) fail("há arquivos alterados sem commit. Faça o commit antes de publicar.");
  if (git("branch", "--show-current") !== "main") fail("publique a partir do main.");
  git("fetch", "-q", "origin", "main", "--tags");
  if (git("rev-parse", "HEAD") !== git("rev-parse", "origin/main")) fail("o main local e o do GitHub estão diferentes. Faça pull/push antes.");
  if (git("tag", "--list", tag) === tag) fail(`a tag ${tag} já existe. Aumente "version" em desktop/package.json.`);
  git("tag", "-a", tag, "-m", `Arasaka Mikoshi ${tag}`);
  git("push", "-q", "origin", tag);
  console.log(`✓ tag ${tag} enviada. O GitHub Actions monta e publica o Release: https://github.com/${repo}/actions`);
}

async function waitPublished() {
  console.log("… esperando o Release ser publicado (testes + build, uns 15 min)");
  for (let i = 0; i < 180; i++) {
    const r = await fetch(`https://api.github.com/repos/${repo}/releases/tags/${tag}`, { headers });
    if (r.ok) return r.json(); // rascunhos não aparecem aqui: só o Release já completo
    if (r.status !== 404) console.log(`  (GitHub respondeu HTTP ${r.status}; tentando de novo)`);
    await new Promise(res => setTimeout(res, 10000));
  }
  fail(`o Release não foi publicado em 30 min. Veja https://github.com/${repo}/actions`);
}

async function verify() {
  const rel = await waitPublished();
  const names = rel.assets.map(a => a.name);
  const missing = NEEDED.filter(n => !names.includes(n));
  if (missing.length) fail(`faltam arquivos no Release: ${missing.join(", ")}`);
  for (const n of NEEDED) {
    const r = await fetch(`https://github.com/${repo}/releases/download/${tag}/${n}?v=${Date.now()}`, { method: "HEAD", redirect: "follow" });
    if (!r.ok) fail(`${n} não baixa (HTTP ${r.status}).`);
  }
  console.log(`✓ Release ${tag} publicado e completo: ${rel.html_url}`);
}

(async () => {
  if (!process.argv.includes("--verify")) pushTag();
  await verify();
})().catch(e => fail(e.message));
