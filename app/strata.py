"""Strata (github.com/Niko1221/Strata) como motor alternativo ao llama.cpp.

O Strata tem motor e servidor proprios (API compativel com a OpenAI, /health e /props como o llama-server) e pacotes
de modelo proprios (Qwen3.8-Flash-Next, Qwen3.6-35B-A3B). O Mikoshi:
- instala o Strata Terminal pelo instalador oficial (install.ps1 / install.sh da ultima versao, que confere a
  assinatura do pacote e nao pede administrador);
- inicia o modelo com `strata launch ... --yes --no-tui --output json`, na porta do motor do Mikoshi e so em
  127.0.0.1: o chat, os agentes e o Codex passam a falar com o Strata sem mudar nada;
- para com `strata launch --stop`.
O proprio `strata launch` baixa o motor certo para a placa e o modelo (dezenas de GB na primeira vez).
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import nettls

WIN = os.name == "nt"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
RELEASE = "https://github.com/Niko1221/Strata/releases/latest/download/"
PREFIX = "strata:"  # ENGINE_PROC.model de um modelo do Strata: "strata:<familia>/<modelo>"

# Modelos que o `strata launch` conhece (README e docs/MODELS.md do Strata). A ordem e a recomendacao do projeto:
# ISTA-DASLab, depois Unsloth, depois Gyro. Outro nome pode ser digitado: o proprio Strata confere.
CATALOG = [
    {"family": "qwen", "title": "Qwen3.8-Flash-Next · ISTA-DASLab (recomendado)", "models": ["q2_0", "iq2_xs", "iq3_xxs", "iq3_s"]},
    {"family": "coder", "title": "Qwen3.8-Flash-Next Coder · ISTA-DASLab", "models": ["iq1_m"]},
    {"family": "unsloth", "title": "Qwen3.8-Flash-Next · Unsloth", "models": ["ud-iq4_xs", "ud-q4_k_xl", "ud-q5_k_xl", "ud-q6_k_xl", "q8_0"]},
    {"family": "gyro", "title": "Qwen3.8-Flash-Next · Gyro (menos memória, mais lento)", "models": ["gyro-s", "gyro-m"]},
    {"family": "qwen36", "title": "Qwen3.6-35B-A3B · Unsloth", "models": ["ud-q4_k_xl", "ud-q6_k_xl", "q8_0"]},
]
CONTEXTS = ["32k", "64k", "128k", "256k"]
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,40}$")


def _home():
    if WIN:
        base = os.environ.get("STRATA_DATA_DIR") or os.path.join(os.environ.get("LOCALAPPDATA", ""), "Strata")
    else:
        base = os.environ.get("STRATA_DATA_DIR") or os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), "strata")
    return Path(base)


def cli_path():
    """O comando `strata` instalado, ou None."""
    p = _home() / "terminal" / "current" / "bin" / ("strata.cmd" if WIN else "strata")
    if p.is_file():
        return p
    found = shutil.which("strata")
    return Path(found) if found else None


def _run(args, timeout=60):
    exe = cli_path()
    if not exe:
        raise RuntimeError("O Strata não está instalado. Vá em Modelos → Runtime e instale o Strata.")
    return subprocess.run([str(exe), *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)


def _json_line(text):
    """Ultimo objeto JSON impresso pelo `strata ... --output json`."""
    for line in reversed((text or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


def version():
    try:
        out = _run(["--version"], timeout=30)
        m = re.search(r"\d+\.\d+\.\d+", out.stdout + out.stderr)
        return m.group(0) if m else ""
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return ""


_version_cache = {}


def status():
    exe = cli_path()
    ver = ""
    if exe:
        try:
            key = (str(exe), exe.stat().st_mtime)
        except OSError:
            key = (str(exe), 0)
        if key not in _version_cache:  # `strata --version` abre um Python: so uma vez por instalacao
            _version_cache.clear()
            _version_cache[key] = version()
        ver = _version_cache[key]
    return {"installed": bool(exe), "version": ver, "catalog": CATALOG, "contexts": CONTEXTS}


def install(task):
    """Baixa e roda o instalador oficial da ultima versao do Strata (Terminal + comando `strata`)."""
    name = "install.ps1" if WIN else "install.sh"
    task["detail"] = "Baixando o instalador do Strata..."
    tmp = Path(tempfile.mkdtemp(prefix="mikoshi-strata-"))
    try:
        script = tmp / name
        with nettls.urlopen(RELEASE + name, timeout=120) as r:
            script.write_bytes(r.read())
        cmd = (["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script)] if WIN
               else ["sh", str(script)])
        task["detail"] = "Instalando o Strata (confere a assinatura do pacote)..."
        p = subprocess.Popen(cmd, cwd=str(tmp), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             creationflags=NO_WINDOW)
        lines = []
        for raw in iter(p.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if line:
                lines.append(line)
                task["detail"] = "Strata: " + line[:160]
            if task.get("cancel"):
                p.kill()
                raise RuntimeError("cancelado")
        if p.wait() != 0 or not cli_path():
            raise RuntimeError("Falha ao instalar o Strata: " + " | ".join(lines[-4:]))
        task["detail"] = f"Strata {version()} instalado"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def model_id(family, model):
    return f"{PREFIX}{family}/{model}"


def parse_id(model_rel):
    """("qwen", "q2_0") de "strata:qwen/q2_0", ou None."""
    if not (model_rel or "").startswith(PREFIX):
        return None
    family, _, model = model_rel[len(PREFIX):].partition("/")
    return (family, model) if NAME_RE.match(family) and NAME_RE.match(model) else None


def label(model_rel):
    fm = parse_id(model_rel)
    if not fm:
        return model_rel
    title = next((c["title"].split(" · ")[0] for c in CATALOG if c["family"] == fm[0]), fm[0])
    return f"Strata · {title} {fm[1].upper()}"


def launch_args(family, model, port, context=""):
    if not (NAME_RE.match(family or "") and NAME_RE.match(model or "")):
        raise ValueError("Modelo do Strata inválido.")
    args = ["launch", "--family", family, "--model", model, "--port", str(int(port)), "--access", "local",
            "--no-api-key", "--yes", "--no-tui", "--restart", "--output", "json"]
    if context:
        if context not in CONTEXTS and not str(context).isdigit():
            raise ValueError("Contexto inválido para o Strata.")
        args += ["--context", str(context)]
    return args


def launch(family, model, port, context="", on_line=None, should_stop=None):
    """Inicia o modelo (baixa o motor e o modelo se faltar) e volta quando o servidor estiver servindo.
    Devolve o objeto JSON do Strata; erro vira RuntimeError com a mensagem dele."""
    exe = cli_path()
    if not exe:
        raise RuntimeError("O Strata não está instalado. Vá em Modelos → Runtime e instale o Strata.")
    p = subprocess.Popen([str(exe), *launch_args(family, model, port, context)], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    out = []
    for raw in iter(p.stdout.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip()
        out.append(line)
        if on_line and line and not line.startswith("{"):
            on_line(line)
        if should_stop and should_stop():
            p.kill()
            raise RuntimeError("cancelado")
    p.wait()
    res = _json_line("\n".join(out))
    if not res:
        raise RuntimeError("O Strata fechou sem responder. Últimas linhas:\n" + "\n".join(l for l in out[-8:] if l))
    if not res.get("ok"):
        raise RuntimeError("Strata: " + ((res.get("error") or {}).get("message") or "falha ao iniciar"))
    if res.get("serving") is False:
        raise RuntimeError("Strata: " + (res.get("message") or res.get("engine_problem") or "o modelo não foi iniciado"))
    return res


def stop():
    try:
        _run(["launch", "--stop"], timeout=60)
    except (OSError, RuntimeError, subprocess.SubprocessError):
        pass
