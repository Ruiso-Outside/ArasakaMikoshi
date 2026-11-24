// Gera o pacote de atualização a partir de dist/win-unpacked:
//   dist/app-patch-X.Y.Z.zip  -> pastas resources/app e resources/server
//   dist/patch.json           -> versão, versão do Electron, tamanho e SHA-512
//   node scripts/make-patch.js
// Quem envia ao Release é o GitHub Actions (.github/workflows/release.yml).
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { spawnSync } = require("child_process");
const pkg = require("../package.json");

const { version } = pkg;
const electron = pkg.devDependencies.electron;
const dist = path.join(__dirname, "..", "dist");
const res = path.join(dist, "win-unpacked", "resources");
const file = `app-patch-${version}.zip`;
const fail = msg => { console.error("✗ " + msg); process.exit(1); };

const builtVersion = JSON.parse(fs.readFileSync(path.join(res, "app", "package.json"), "utf8")).version;
if (builtVersion !== version) fail(`dist/win-unpacked é da versão ${builtVersion}, não ${version}. Rode o build de novo.`);
if (!fs.existsSync(path.join(res, "server", "server.py"))) fail("resources/server/server.py não encontrado.");

const zip = path.join(dist, file);
fs.rmSync(zip, { force: true });
// tar.exe do Windows (o "tar" do Git Bash e o GNU tar, que nao cria .zip)
const TAR = path.join(process.env.SystemRoot || "C:\\Windows", "System32", "tar.exe");
const t = spawnSync(TAR, ["-a", "-c", "-f", zip, "-C", res, "app", "server"], { encoding: "utf8" });
if (t.status !== 0) fail("tar falhou: " + t.stderr);
const data = fs.readFileSync(zip);
const manifest = { version, electron, file, size: data.length,
  sha512: crypto.createHash("sha512").update(data).digest("base64"), date: new Date().toISOString() };
fs.writeFileSync(path.join(dist, "patch.json"), JSON.stringify(manifest, null, 2));
console.log(`✓ ${file} (${(data.length / 1e6).toFixed(1)} MB) e patch.json gerados`);
