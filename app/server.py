"""LLM Local - servidor interno do app desktop.

Iniciado escondido pelo Electron (desktop/main.js); nao e usado sozinho.
Sem dependencias externas (so biblioteca padrao do Python).
Roda apenas em 127.0.0.1: nada sai do seu computador, exceto quando VOCE
pede para baixar o motor, um modelo ou faz uma pesquisa na web.
"""
import atexit
import base64
import json
import os
import random
import re
import secrets
import socket
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import agent
import agents
import kb
import mcp
import modelsettings
import train
import gateway
import gguf
import rea
import catalog
import media
import nettls
import promptcache
import sprites
import strata
import tune
import websearch
import winrt

# LLM_HOME: pasta de dados (modelos, motor, conversas). O app desktop define isso.
ROOT = Path(os.environ.get("LLM_HOME") or Path(__file__).resolve().parent.parent)
WEB = Path(__file__).resolve().parent / "web"
MODELS = ROOT / "models"
ENGINE = ROOT / "engine"
DATA = ROOT / "data"
CHATS = DATA / "chats"
SETTINGS_FILE = DATA / "settings.json"
STATS_FILE = DATA / "stats.json"
MMPROJ_FILE = DATA / "mmproj.json"
MTP_FILE = DATA / "mtp.json"
MODEL_INFO_FILE = DATA / "model_info.json"
CPU_COUNT = os.cpu_count() or 4

HOST = "127.0.0.1"
PORT = int(os.environ.get("LLM_PORT", "8765"))
ENGINE_PORT = PORT + 1
# Chave do motor (llama-server): nova a cada execucao. Sem ela, qualquer site aberto no navegador podia usar
# o modelo pela porta do motor (o llama-server libera CORS para todos). Outros programas usam a API local (gateway).
ENGINE_KEY = secrets.token_urlsafe(24)
ENGINE_HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {ENGINE_KEY}"}
UA = "LLM-Local/1.0"
BACKENDS = ("cuda", "vulkan", "cpu")
EMBED_CTX = 8192  # contexto do motor de embeddings (bases de conhecimento)
EXTRA_CTX = 16384  # contexto automatico dos modelos extras e dos que nao cabem na placa
BENCH_FILE = DATA / "bench.json"  # medidas do Otimizar, por modelo

for d in (MODELS, ENGINE, CHATS):
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_SETTINGS = {
    "system_prompt": "",
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "min_p": 0.05,
    "repeat_penalty": 1.0,
    "max_tokens": 0,  # 0 = sem limite
    "ctx_size": 8192,  # 0 = automatico: o maior em que cada modelo cabe inteiro na placa
    "gpu_layers": 999,  # 999 = tudo o que couber na GPU (o resto vai para a RAM, escolhido pelo llama.cpp)
    "kv_cache": "f16",  # cache do contexto: f16 (normal) | q8_0 (8 bits: quase metade da memoria)
    "ubatch": 0,  # lote de leitura do prompt (0 = padrao do motor, 512); o Otimizar ajusta por modelo
    "threads": 0,  # (antigo) numero fixo de threads; 0 = usa cpu_percent
    "cpu_percent": 0,  # 0 = automatico (llama.cpp usa os nucleos fisicos, ~50%)
    "backend": "auto",
    "extra_args": "",
    "hf_token": "",
    "last_model": "",
    "auto_load": True,  # carrega o ultimo modelo ao iniciar
    "auto_summarize": True,  # resume a conversa antiga quando o contexto passa de ~80%
    "use_recommended": True,  # aplica a amostragem recomendada da familia de cada modelo
    "quick_model": "",  # modelo da pergunta rapida ("" = o principal)
    "quick_device": "gpu",  # onde roda o modelo da pergunta rapida: gpu | cpu ("auto" antigo = gpu)
    "quick_keep_loaded": True,
    "vram_guard": True,
    "rea_codex": True,
    "mode_models": {},  # modelo predefinido de cada modo: {"chat": ..., "codex": ..., "agents": ...} ("" = o carregado)
    "mode_switch": True,
    # API local para outros programas (Visual Studio, VS Code, Unreal...): compativel com Ollama e OpenAI
    "api_enabled": False,
    "api_port": 11434,  # a porta do Ollama: as IDEs ja procuram nela
    "api_require_key": False,
    "api_key": "",  # ao entrar num modo, descarrega a IA ativa e carrega o modelo predefinido dele  # o Codex pode usar o REA (engenharia reversa) instalado no motor, pelo terminal  # VRAM cheia por ~40 s sem nada gerando: descarrega os modelos para o PC nao travar  # deixa o modelo da pergunta rapida sempre carregado (responde na hora)
    "language": "",  # idioma da interface: pt-BR | en | es ("" = o do sistema)
    "reply_language": "auto",  # auto = a IA responde no idioma da pergunta | app = sempre no idioma do app
    "extra_idle_minutes": 10,  # modelos extras (agentes/pergunta rapida) sao descarregados apos X min parados
    "search_provider": "duckduckgo",  # duckduckgo | google | searxng
    "search_region": "br-pt",
    "search_results": 4,
    "google_api_key": "",
    "google_cx": "",
    "searxng_url": "",
    # Codex
    "workspaces": [],
    "agent_confirm_commands": True,
    "agent_confirm_writes": False,
    "agent_max_steps": 40,
    # midia
    "whisper_model": "",
    "whisper_language": "auto",
    "image_preset": "z-image-turbo",
    "image_width": 0,
    "image_height": 0,
    "image_steps": 0,
    "image_offload": True,
    "image_enhance": True,
    "image_unload_llm": False,
}
SECRETS = ("hf_token", "google_api_key", "api_key")

_settings_lock = threading.Lock()


def load_settings():
    s = dict(DEFAULT_SETTINGS)
    try:
        s.update(json.loads(SETTINGS_FILE.read_text("utf-8")))
    except (OSError, ValueError):
        pass
    return s


def save_settings(new):
    with _settings_lock:
        s = load_settings()
        for k, v in new.items():
            if k in DEFAULT_SETTINGS:
                s[k] = v
        SETTINGS_FILE.write_text(json.dumps(s, indent=2, ensure_ascii=False), "utf-8")
        return s


REPLY_RULES = {"pt-BR": "Responda sempre em português do Brasil.", "en": "Always answer in English.",
               "es": "Responde siempre en español.", "th": "ตอบเป็นภาษาไทยเสมอ", "zh": "请始终用简体中文回答。",
               "ja": "常に日本語で答えてください。", "ko": "항상 한국어로 답하세요.", "id": "Selalu jawab dalam bahasa Indonesia.",
               "vi": "Luôn trả lời bằng tiếng Việt.", "fr": "Réponds toujours en français.", "de": "Antworte immer auf Deutsch.",
               "ru": "Всегда отвечай по-русски."}


def reply_rule(s=None):
    """Instrucao extra quando o usuario quer que a IA responda sempre no idioma do app."""
    s = s or load_settings()
    return REPLY_RULES.get(s.get("language") or "pt-BR", "") if s.get("reply_language") == "app" else ""


def with_reply_rule(messages, s=None):
    rule = reply_rule(s)
    if not rule:
        return messages
    if messages and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str):
        return [{**messages[0], "content": messages[0]["content"] + "\n\n" + rule}] + messages[1:]
    return [{"role": "system", "content": rule}] + messages


def masked(s):
    s = dict(s)
    for k in SECRETS:
        s[k] = "***" if s.get(k) else ""
    return s


# ---------------------------------------------------------------- contador de tokens

_stats_lock = threading.Lock()


def load_stats():
    try:
        return json.loads(STATS_FILE.read_text("utf-8"))
    except (OSError, ValueError):
        return {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "requests": 0, "searches": 0,
                "since": time.time()}


def add_stats(timings=None, searches=0):
    with _stats_lock:
        st = load_stats()
        if timings:
            st["prompt_tokens"] += int(timings.get("prompt_n") or 0)
            st["cached_tokens"] = st.get("cached_tokens", 0) + int(timings.get("cache_n") or 0)
            st["completion_tokens"] += int(timings.get("predicted_n") or 0)
            st["requests"] += 1
        st["searches"] = st.get("searches", 0) + searches
        STATS_FILE.write_text(json.dumps(st), "utf-8")


# ---------------------------------------------------------------- rede

class _Redirect(urllib.request.HTTPRedirectHandler):
    """Remove o token do Hugging Face quando o redirect vai para outro dominio (CDN)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


class _Opener:
    """Downloads da internet: HTTPS com a lista de certificados do nettls (corrige o CERTIFICATE_VERIFY_FAILED)."""

    def open(self, req, timeout=30):
        return nettls.urlopen(req, timeout=timeout, handlers=(_Redirect,))


OPENER = _Opener()


def make_request(url, headers=None):
    h = {"User-Agent": UA}
    if "huggingface.co" in url:
        token = load_settings().get("hf_token", "").strip()
        if token:
            h["Authorization"] = f"Bearer {token}"
    h.update(headers or {})
    return urllib.request.Request(url, headers=h)


def get_json(url):
    with OPENER.open(make_request(url), timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def http_error_text(e):
    """Texto do erro HTTP para a pessoa: o 403 do GitHub (limite de consultas) nao e o do Hugging Face (modelo restrito)."""
    url = getattr(e, "url", "") or getattr(e, "filename", "") or ""
    hdrs = getattr(e, "headers", None) or {}
    if "api.github.com" in url and e.code in (403, 429):
        reset = hdrs.get("X-RateLimit-Reset") if hasattr(hdrs, "get") else None
        mins = ""
        try:
            mins = f" Libera em cerca de {max(1, round((int(reset) - time.time()) / 60))} min."
        except (TypeError, ValueError):
            pass
        return "o GitHub limitou as consultas deste PC (60 por hora sem login)." + mins + " Tente de novo depois."
    if "huggingface.co" in url and e.code in (401, 403):
        return "modelo restrito, coloque seu token do Hugging Face nas Configuracoes"
    return e.reason or "falha no download"


def llama_releases(tagos):
    """Versoes recentes do llama.cpp. Pela API do GitHub; se ela recusar (limite de 60 consultas por hora sem login),
    pelo feed de releases e pela lista de arquivos de cada versao, que nao contam no limite."""
    try:
        return get_json("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15")
    except urllib.error.HTTPError as e:
        if e.code not in (403, 429):
            raise
        first = e
    try:
        with OPENER.open(make_request("https://github.com/ggml-org/llama.cpp/releases.atom"), timeout=30) as r:
            atom = r.read().decode("utf-8", "replace")
        tags = list(dict.fromkeys(re.findall(r'/ggml-org/llama\.cpp/releases/tag/([^"<>\s]+)"', atom)))
        for tag in tags[:6]:
            with OPENER.open(make_request(f"https://github.com/ggml-org/llama.cpp/releases/expanded_assets/{tag}"), timeout=30) as r:
                html = r.read().decode("utf-8", "replace")
            names = sorted(set(re.findall(r'/ggml-org/llama\.cpp/releases/download/[^/"]+/([^"/?#]+)"', html)))
            if any(tagos in n for n in names):
                base = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/"
                return [{"tag_name": tag, "assets": [{"name": n, "browser_download_url": base + n, "size": 0} for n in names]}]
    except Exception:  # noqa: BLE001
        pass
    raise first


# ---------------------------------------------------------------- tarefas (downloads)

TASKS = {}
_tasks_lock = threading.Lock()


def new_task(kind, label):
    t = {"id": uuid.uuid4().hex[:10], "kind": kind, "label": label, "done": 0, "total": 0,
         "status": "running", "error": "", "detail": "", "cancel": False, "started": time.time()}
    with _tasks_lock:
        TASKS[t["id"]] = t
    return t


def public_tasks():
    with _tasks_lock:
        # remove tarefas concluidas ha mais de 10 min
        now = time.time()
        for tid in [k for k, v in TASKS.items() if v["status"] != "running" and now - v.get("ended", now) > 600]:
            del TASKS[tid]
        return [{k: v for k, v in t.items() if k != "cancel"} for t in TASKS.values()]


class Cancelled(Exception):
    pass


def download(url, dest: Path, task, base_done=0):
    """Baixa com suporte a retomada (.part)."""
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    with OPENER.open(make_request(url, headers), timeout=60) as r:
        if start and r.status != 206:
            start = 0
        length = int(r.headers.get("Content-Length") or 0)
        file_total = start + length
        with open(part, "ab" if start else "wb") as f:
            got = start
            while True:
                if task["cancel"]:
                    raise Cancelled()
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                task["done"] = base_done + got
    if length and part.stat().st_size < file_total:
        raise IOError("Download incompleto, tente de novo (ele continua de onde parou).")
    part.replace(dest)
    return dest.stat().st_size


def run_task(task, fn):
    def wrapper():
        try:
            fn(task)
            task["status"] = "done"
        except Cancelled:
            task["status"] = "cancelled"
        except urllib.error.HTTPError as e:
            task["status"] = "error"
            task["error"] = f"HTTP {e.code}: " + http_error_text(e)
        except Exception as e:  # noqa: BLE001
            task["status"] = "error"
            task["error"] = str(e)
        task["ended"] = time.time()

    threading.Thread(target=wrapper, daemon=True).start()


media.configure(ROOT, lambda url, dest, task, base=0: download(url, dest, task, base), get_json)
rea.configure(ENGINE, lambda url, dest, task, base=0: download(url, dest, task, base), get_json)
agents.configure(DATA)
kb.configure(DATA)
modelsettings.configure(DATA)
mcp.configure(DATA)
SPRITE_LOCK = threading.Lock()  # uma geracao de sprites por vez (a GPU e uma so)


def connect_construct_later(delay=2.0):
    """Conecta ao Arasaka Construct em segundo plano, sem configuracao: acha o editor e abre `demonic-editor --mcp`."""
    def run():
        time.sleep(delay)
        try:
            mcp.MANAGER.sync()
            srv = mcp.MANAGER.get(mcp.CONSTRUCT_ID)
            if srv and srv.cfg.get("enabled", True):
                srv.connect()
        except mcp.MCPError as e:
            print("Arasaka Construct:", e, flush=True)
    threading.Thread(target=run, daemon=True).start()
CHECKPOINTS = DATA / "checkpoints"


# ---------------------------------------------------------------- motor (llama.cpp)

def detect_gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
        if out:
            name, mem = out.splitlines()[0].rsplit(",", 1)
            return {"vendor": "nvidia", "name": name.strip(), "vram_mb": int(float(mem))}
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return {"vendor": "other", "name": "", "vram_mb": 0}


GPU = detect_gpu()


def nvidia_driver_major():
    """Versao principal do driver da NVIDIA (ex.: 580), ou None sem driver."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
        return int(out.splitlines()[0].split(".")[0]) if out else None
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------- diagnostico
# Amostra a VRAM real (nvidia-smi) e a RAM a cada 20 s. Quando a VRAM enche, o Windows passa a usar a RAM
# como memoria de video ("sysmem fallback") e o PC inteiro pode travar: isso fica registrado e a interface avisa.
DIAG_FILE = DATA / "diagnostico.log"
DIAG = {"vram_used": 0, "vram_total": 0, "gpu_util": 0, "at": 0, "samples": deque(maxlen=90), "full_since": 0,
        "full_count": 0, "guard": None}


def diag_log(msg):
    try:
        if DIAG_FILE.exists() and DIAG_FILE.stat().st_size > 1024 * 1024:
            DIAG_FILE.replace(DIAG_FILE.with_suffix(".old.log"))
        with open(DIAG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")
    except OSError:
        pass


def gpu_sample():
    """Uso real da placa NVIDIA: (MB usados, MB total, % de uso) ou None."""
    if GPU.get("vendor") != "nvidia":
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=8, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        used, total, util = (int(float(x)) for x in out.strip().splitlines()[0].split(","))
        return used, total, util
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def diag_snapshot():
    total, avail = ram_info()
    return {"vram_used": DIAG["vram_used"], "vram_total": DIAG["vram_total"], "gpu_util": DIAG["gpu_util"],
            "vram_full": bool(DIAG["full_since"]), "ram_total": total, "ram_avail": avail,
            "engine": {"state": ENGINE_PROC.state, "model": ENGINE_PROC.model, "device": ENGINE_PROC.device},
            "extras": [{"model": m, "device": d, "state": e.state} for (m, d), e in list(POOL.extra.items())],
            "threads": threading.active_count()}


def vram_guard():
    """Protecao contra travar o PC: com a VRAM cheia e nada sendo gerado, descarrega extras e depois o principal."""
    busy = ENGINE_PROC.busy > 0 or any(e.busy > 0 for e in list(POOL.extra.values()))
    if busy:
        return
    gpu_extras = [m for (m, d), e in list(POOL.extra.items()) if d == "gpu" and e.state in ("ready", "loading")]
    if gpu_extras:
        POOL.stop_device("gpu")
        what = "os modelos extras da GPU (" + ", ".join(model_display(m) for m in gpu_extras) + ")"
    elif ENGINE_PROC.state in ("ready", "loading") and ENGINE_PROC.device == "gpu" and ENGINE_PROC.backend != "strata":
        # o Strata ocupa a placa de proposito (e usa a RAM e o SSD pelo proprio plano de memoria): nao e travamento
        what = "o modelo principal (" + model_display(ENGINE_PROC.model) + ")"
        ENGINE_PROC.stop()
    else:
        return
    POOL.quick_error = "Descarregado pela proteção de VRAM (a placa de vídeo encheu). Carregue de novo quando liberar memória."
    DIAG["guard"] = {"at": time.time(), "what": what}
    DIAG["full_count"] = 0
    diag_log(f"PROTEÇÃO DE VRAM: descarreguei {what}")


def diag_sampler():
    diag_log(f"servidor iniciado | GPU: {GPU.get('name', '-')} {GPU.get('vram_mb', 0)} MB")
    while True:
        g = gpu_sample()
        if g:
            used, total, util = g
            DIAG.update(vram_used=used, vram_total=total, gpu_util=util, at=time.time())
            DIAG["samples"].append((round(time.time()), used, util))
            full = total and used >= total * 0.97
            if full and not DIAG["full_since"]:
                DIAG["full_since"] = time.time()
                diag_log(f"VRAM CHEIA: {used}/{total} MB (o Windows pode começar a usar a RAM e travar) | {json.dumps(diag_snapshot(), ensure_ascii=False)}")
            elif not full and DIAG["full_since"]:
                diag_log(f"VRAM voltou ao normal: {used}/{total} MB (ficou cheia por {time.time() - DIAG['full_since']:.0f}s)")
                DIAG["full_since"] = 0
            DIAG["full_count"] = DIAG["full_count"] + 1 if full else 0
            if full and DIAG["full_count"] >= 2 and load_settings().get("vram_guard", True):
                vram_guard()
        time.sleep(20)


def ram_info():
    """Memoria RAM total e disponivel (MB), pelo Windows (no Linux, pelo /proc/meminfo)."""
    if os.name != "nt":
        try:
            info = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                k, _, v = line.partition(":")
                info[k] = int(v.split()[0]) // 1024
            return info.get("MemTotal", 0), info.get("MemAvailable", info.get("MemFree", 0))
        except (OSError, ValueError):
            return 0, 0
    import ctypes

    class MEMSTAT(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    st = MEMSTAT()
    st.dwLength = ctypes.sizeof(MEMSTAT)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return 0, 0
    return st.ullTotalPhys // 2 ** 20, st.ullAvailPhys // 2 ** 20


def installed_backends():
    found = {}
    for b in BACKENDS:
        d = ENGINE / b
        if d.is_dir():
            exe = next(d.rglob("llama-server.exe" if os.name == "nt" else "llama-server"), None)
            if exe:
                ver = (d / "version.txt").read_text("utf-8").strip() if (d / "version.txt").exists() else ""
                found[b] = {"exe": str(exe), "version": ver}
    return found


def install_engine(task, backend):
    task["detail"] = "Consultando versao mais recente do llama.cpp..."
    # os builds "bNNNN" sao marcados como prerelease; pega o mais novo que tenha binarios do sistema
    # (Windows: -bin-win-; Linux: -bin-ubuntu-, que tambem roda no Zorin, Mint, Pop!_OS e outros)
    linux = os.name != "nt"
    tagos = "-bin-ubuntu-" if linux else "-bin-win-"
    releases = llama_releases(tagos)
    rel = next((r for r in releases if any(tagos in a["name"] for a in r.get("assets", []))), None)
    if not rel:
        raise RuntimeError("Nenhuma versao do llama.cpp com binarios para " + ("Linux" if linux else "Windows") + " encontrada.")
    assets = {a["name"]: (a["browser_download_url"], a.get("size", 0)) for a in rel.get("assets", [])}
    tag = rel.get("tag_name", "")
    files = []
    if linux and backend == "cuda":
        # pacotes ubuntu-cuda-<versao>: o CUDA 13 pede o driver 580+, o 12.x o 570+ (Linux). Escolhe o mais novo que o driver roda.
        drv = nvidia_driver_major()
        if drv is None:
            raise RuntimeError("Nenhum driver da NVIDIA encontrado. Instale o driver em Drivers adicionais (Zorin/Ubuntu), reinicie o PC "
                               "e tente de novo. Sem a placa NVIDIA, use o motor Vulkan.")
        cands = []
        for n in assets:
            m = re.match(r"llama-.*-bin-ubuntu-cuda-([\d.]+)-x64\.tar\.gz$", n)
            if m:
                ver = tuple(int(x) for x in m.group(1).split("."))
                if drv >= (580 if ver[0] >= 13 else 570 if ver >= (12, 8) else 525):
                    cands.append((ver, n, m.group(1)))
        if not cands:
            raise RuntimeError(f"O driver da NVIDIA instalado ({drv}) e antigo para os pacotes CUDA desta versao. "
                               "Atualize o driver em Drivers adicionais ou use o motor Vulkan.")
        _, name, ver = max(cands)
        files.append(name)
        cudart = next((n for n in assets if re.match(rf"cudart-llama-.*-bin-ubuntu-cuda-{re.escape(ver)}-x64\.tar\.gz$", n)), None)
        if cudart:
            files.append(cudart)
    elif linux:
        pat = {"vulkan": r"llama-.*-bin-ubuntu-vulkan-x64\.(zip|tar\.gz)$", "cpu": r"llama-.*-bin-ubuntu-x64\.(zip|tar\.gz)$"}[backend]
        name = next((n for n in assets if re.match(pat, n)), None)
        if not name:
            raise RuntimeError(f"Pacote '{backend}' para Linux nao encontrado na versao {tag}.")
        files.append(name)
    elif backend == "cuda":
        cands = []
        for n in assets:
            m = re.match(r"llama-.*-bin-win-cuda-([\d.]+)-x64\.zip$", n)
            if m:
                cands.append((tuple(int(x) for x in m.group(1).split(".")), n, m.group(1)))
        if not cands:
            raise RuntimeError("Nenhum pacote CUDA encontrado nesta versao. Tente Vulkan.")
        _, name, ver = max(cands)
        files.append(name)
        cudart = f"cudart-llama-bin-win-cuda-{ver}-x64.zip"
        if cudart in assets:
            files.append(cudart)
    else:
        pat = {"vulkan": r"llama-.*-bin-win-vulkan-x64\.zip$", "cpu": r"llama-.*-bin-win-cpu-x64\.zip$"}[backend]
        name = next((n for n in assets if re.match(pat, n)), None)
        if not name:
            raise RuntimeError(f"Pacote '{backend}' nao encontrado na versao {tag}.")
        files.append(name)

    task["total"] = sum(assets[f][1] for f in files)
    tmp = ENGINE / "_download"
    tmp.mkdir(exist_ok=True)
    done = 0
    for f in files:
        task["detail"] = f"Baixando {f}"
        done += download(assets[f][0], tmp / f, task, done)

    task["detail"] = "Extraindo..."
    if ENGINE_PROC.backend == backend:
        ENGINE_PROC.stop()
    dest = ENGINE / backend
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir()
    for f in files:
        if f.endswith(".tar.gz"):
            import tarfile
            with tarfile.open(tmp / f) as t:
                t.extractall(dest, filter="data") if hasattr(tarfile, "data_filter") else t.extractall(dest)  # noqa: S202
        else:
            with zipfile.ZipFile(tmp / f) as z:
                z.extractall(dest)
    if linux:  # o .zip nao guarda a permissao de executar: programas e bibliotecas
        for x in dest.rglob("*"):
            if x.is_file() and (x.suffix in ("", ".so") or ".so." in x.name):
                x.chmod(0o755)
    # bibliotecas do CUDA precisam ficar na mesma pasta do executavel
    exe = next(dest.rglob("llama-server.exe" if not linux else "llama-server"), None)
    if exe and exe.parent != dest:
        for dll in dest.glob("*.dll"):
            if not (exe.parent / dll.name).exists():
                shutil.move(str(dll), exe.parent / dll.name)
    if linux and exe:  # Linux: o cudart vem numa pasta propria (cudart-llama-...); o motor procura as .so ao lado do executavel
        for so in [x for x in dest.rglob("*.so*") if x.parent != exe.parent and x.parent.name.startswith("cudart-")]:
            if not (exe.parent / so.name).exists():
                shutil.move(str(so), exe.parent / so.name)
    (dest / "version.txt").write_text(f"{tag} ({files[0]})", "utf-8")
    shutil.rmtree(tmp, ignore_errors=True)
    task["detail"] = f"Motor {backend} instalado ({tag})"


def _job_object():
    """No Windows, garante que o llama-server morra junto com este app."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = ctypes.c_void_p
        k32.SetInformationJobObject.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32)
        k32.AssignProcessToJobObject.argtypes = (ctypes.c_void_p, ctypes.c_void_p)

        class Basic(ctypes.Structure):
            _fields_ = [("a", ctypes.c_longlong), ("b", ctypes.c_longlong), ("LimitFlags", ctypes.c_uint32),
                        ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("e", ctypes.c_uint32),
                        ("f", ctypes.c_size_t), ("g", ctypes.c_uint32), ("h", ctypes.c_uint32)]

        class Ext(ctypes.Structure):
            _fields_ = [("Basic", Basic), ("Io", ctypes.c_ulonglong * 6), ("i", ctypes.c_size_t),
                        ("j", ctypes.c_size_t), ("k", ctypes.c_size_t), ("l", ctypes.c_size_t)]

        job = k32.CreateJobObjectW(None, None)
        info = Ext()
        info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
        return (k32, job)
    except Exception:  # noqa: BLE001
        return None


def gguf_context_length(path):
    """Contexto maximo de treino gravado no GGUF (<arquitetura>.context_length); 0 se nao der para ler."""
    return gguf.context_length(path)


# erro de memoria ao criar o cache do contexto (CUDA, Vulkan, Metal): da para tentar de novo com menos contexto
KV_OOM_RE = re.compile(r"failed to allocate buffer for kv cache|cudaMalloc failed: out of memory|ErrorOutOfDeviceMemory|"
                       r"failed to allocate .* buffer of size", re.I)
MIN_FIT_CTX = 4096


class EngineProcess:
    def __init__(self, port=ENGINE_PORT):
        self.port = port
        self.device = "gpu"  # gpu | cpu
        self.busy = 0
        self.last_used = time.time()
        self.proc = None
        self.model = ""
        self.backend = ""
        self.state = "stopped"  # stopped | loading | ready | error
        self.error = ""
        self.n_ctx = 0
        self.ctx_requested = 0  # o contexto pedido nas configuracoes (o usado pode ser menor: n_ctx)
        self.notice = ""  # aviso para a tela (ex.: contexto reduzido para caber na placa)
        self.modalities = {}
        self.mtp = self.mtp_error = False  # acelerador em uso / recusado pelo motor
        self.accel = ""  # tipo do acelerador em uso: mtp, draft ou ngram
        self.log = deque(maxlen=400)
        self.lock = threading.Lock()
        self.job = _job_object()
        self._strata_gen = 0

    def stop(self):
        with self.lock:
            p, self.proc = self.proc, None
            was_strata = self.backend == "strata" and self.state != "stopped"
            self.state, self.model, self.error = "stopped", "", ""
            self.n_ctx = 0  # nao mostrar o contexto da carga anterior
            self._strata_gen = getattr(self, "_strata_gen", 0) + 1  # cancela um `strata launch` em andamento
        if was_strata:
            strata.stop()  # o servidor do Strata roda separado: `strata launch --stop`
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()

    def start(self, model_rel, s, device=None, embed=False, no_mtp=False, ctx=None, kv=None):
        """ctx: contexto ja ajustado (nova tentativa depois de faltar memoria); None = o das configuracoes.
        kv: tipo do cache do contexto ja ajustado (o motor recusou o de 8 bits); None = o das configuracoes."""
        if strata.parse_id(model_rel):
            return self.start_strata(model_rel, s.get("strata_context", ""))
        self._args = (model_rel, dict(s), device, embed, no_mtp)
        if not no_mtp:
            self.mtp_error = False
        model_path = (MODELS / model_rel).resolve()
        if not embed:
            s = model_effective(model_rel, s)  # ajustes proprios do modelo (contexto, GPU)
        s = dict(s)
        self._kv = kv
        kv_type = kv or (s.get("kv_cache") if s.get("kv_cache") in modelsettings.KV_TYPES else "f16")
        if MODELS.resolve() not in model_path.parents or not model_path.is_file():
            raise ValueError("Modelo nao encontrado.")
        backs = installed_backends()
        if not backs:
            raise ValueError("Nenhum motor instalado. Va em Modelos > Motor e instale.")
        backend = s.get("backend", "auto")
        if backend not in backs:
            backend = next(b for b in BACKENDS if b in backs)
        self.stop()
        if self is ENGINE_PROC:
            POOL.stop(model_rel)  # se ja estava carregado como extra, libera antes
        on_cpu = device == "cpu"
        if self is ENGINE_PROC and not on_cpu:
            POOL.make_room_for_main(model_rel)  # o principal tem prioridade na VRAM
        if ctx is not None:
            s["ctx_size"] = ctx
        elif embed:
            # os trechos da base tem ~1200 caracteres: o contexto das configuracoes (ex.: 64000) so gastava VRAM (4 GB ou mais)
            self.notice, self.ctx_requested = "", EMBED_CTX
            s["ctx_size"] = min(EMBED_CTX, gguf_context_length(model_path) or EMBED_CTX)
        else:
            self.notice = ""
            if int(s["ctx_size"]) <= 0:  # automatico: o maior que cabe inteiro na VRAM livre agora
                s["ctx_size"] = auto_context(model_rel, kv_type, self, on_cpu or backend == "cpu")
            self.ctx_requested = int(s["ctx_size"])
            # mais contexto que o de treino nao serve para nada e so gasta VRAM (ex.: 720000 num modelo de 262144)
            train = gguf_context_length(model_path)
            if train and int(s["ctx_size"]) > train:
                s["ctx_size"] = train
                self.notice = (f"Contexto reduzido de {self.ctx_requested} para {train} tokens: é o máximo deste modelo. "
                               "Ajuste em Configurações → Modelo e runtime.")
        ngl = "0" if on_cpu else str(int(s["gpu_layers"]))
        if not on_cpu and int(s["gpu_layers"]) >= 999:
            ngl = "auto"  # tudo que couber na VRAM livre de verdade; o resto fica na RAM (um numero fixo desliga esse encaixe)
        cmd = [backs[backend]["exe"], "-m", str(model_path), "--host", "127.0.0.1", "--port", str(self.port),
               "-c", str(int(s["ctx_size"])), "-ngl", ngl, "--jinja", "--api-key", ENGINE_KEY]
        if not embed and kv_type != "f16":
            cmd += ["-ctk", kv_type, "-ctv", kv_type]
        if not embed and int(s.get("ubatch") or 0) > 0:
            cmd += ["-ub", str(int(s["ubatch"])), "-b", str(max(2048, int(s["ubatch"])))]
        if ngl == "auto" and backend != "cpu" and ctx is None and not embed:
            self.notice = " ".join(filter(None, [self.notice, vram_notice(model_rel, int(s["ctx_size"]), kv_type)]))
        if on_cpu:
            cmd += ["--device", "none"]  # CPU e RAM apenas, sem tocar na GPU
        if embed:
            # motor so de embeddings (bases de conhecimento); o Qwen3-Embedding usa o ultimo token
            cmd += ["--embedding", "-b", "4096", "-ub", "4096"]
            if "qwen3-embedding" in model_rel.lower():
                cmd += ["--pooling", "last"]
        mm = mmproj_for(model_rel)
        if mm:
            cmd += ["--mmproj", str(mm)]
        kind, mt = ("", None) if (embed or no_mtp) else accel_for(model_rel)
        if on_cpu and kind != "ngram":
            kind, mt = "", None  # rascunho e MTP precisam da GPU; o n-gram funciona em qualquer lugar
        cmd += accel_args(kind, mt)  # decodificacao especulativa
        threads = cpu_threads(s)
        if threads:
            cmd += ["-t", str(threads), "-tb", str(threads)]
        if s.get("extra_args", "").strip():
            cmd += [a.strip('"') for a in shlex.split(s["extra_args"], posix=False)]
        self.log.clear()
        self.log.append("> " + " ".join("***" if c == ENGINE_KEY else f'"{c}"' if " " in c else c for c in cmd))
        env = None
        if os.name != "nt":  # as bibliotecas .so do llama.cpp ficam ao lado do programa
            env = {**os.environ, "LD_LIBRARY_PATH": os.pathsep.join(filter(None, [str(Path(cmd[0]).parent), os.environ.get("LD_LIBRARY_PATH", "")]))}
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=str(Path(cmd[0]).parent), env=env,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if self.job:
            try:
                self.job[0].AssignProcessToJobObject(self.job[1], int(proc._handle))
            except Exception:  # noqa: BLE001
                pass
        with self.lock:
            self.proc, self.model, self.backend, self.state, self.error = proc, model_rel, backend, "loading", ""
            self.mtp, self.accel = bool(kind), kind
            self.ctx_used = int(s["ctx_size"])
            self.kv_type = "f16" if embed else kv_type
            self.device = "cpu" if on_cpu or backend == "cpu" or int(s["gpu_layers"]) == 0 else "gpu"
        threading.Thread(target=self._read_output, args=(proc,), daemon=True).start()
        threading.Thread(target=self._wait_ready, args=(proc,), daemon=True).start()

    def start_strata(self, model_rel, context=""):
        """Motor Strata: `strata launch` sobe o servidor dele nesta mesma porta; o resto do app nao percebe a troca."""
        if self is not ENGINE_PROC:
            raise ValueError("O Strata roda só como modelo principal.")
        family, model = strata.parse_id(model_rel)
        strata.launch_args(family, model, self.port, context)  # valida antes de parar o modelo atual
        if not strata.cli_path():
            raise ValueError("O Strata não está instalado. Vá em Modelos → Runtime e instale o Strata.")
        self.stop()
        POOL.stop()  # o Strata usa a placa toda: libera os modelos extras do llama.cpp
        with self.lock:
            gen = self._strata_gen
            self.model, self.backend, self.state, self.error = model_rel, "strata", "loading", ""
            self.mtp, self.mtp_error, self.accel, self.device = False, False, "", "gpu"
            self.notice, self.ctx_requested, self.modalities = "", 0, {}
        self.log.clear()
        self.log.append(f"> strata {' '.join(strata.launch_args(family, model, self.port, context))}")
        self.log.append("[Arasaka Mikoshi] Na primeira vez o Strata baixa o motor da placa e o modelo (pode levar bastante tempo).")

        def run():
            try:
                res = strata.launch(family, model, self.port, context, on_line=self.log.append,
                                    should_stop=lambda: self._strata_gen != gen)
                for n in res.get("notices") or []:
                    self.log.append(f"[Strata] {n}")
                if self._strata_gen != gen:
                    return
                self.n_ctx, self.modalities = self._read_props()
                self.state = "ready"
                diag_log(f"Strata pronto: {model_rel} na porta {self.port}")
            except Exception as e:  # noqa: BLE001
                if self._strata_gen != gen:
                    return
                self.log.append(str(e))
                self.state, self.error = "error", str(e)
                diag_log(f"Strata falhou ao iniciar {model_rel}: {e}")
        threading.Thread(target=run, daemon=True).start()

    def _read_output(self, proc):
        for raw in iter(proc.stdout.readline, b""):
            self.log.append(raw.decode("utf-8", "replace").rstrip())

    def _wait_ready(self, proc):
        started = time.monotonic()
        while proc is self.proc and proc.poll() is None:
            try:
                with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{self.port}/health", headers=ENGINE_HEADERS), timeout=2) as r:
                    if r.status == 200:
                        self.n_ctx, self.modalities = self._read_props()
                        self.state = "ready"
                        return
            except (urllib.error.URLError, OSError):
                pass
            # modelo pequeno ou ja no cache do disco fica pronto em instantes: confere mais vezes no comeco
            time.sleep(0.1 if time.monotonic() - started < 3 else 0.5)
        model_rel, s, device, embed, no_mtp = self._args
        used = getattr(self, "ctx_used", 0)
        hint = winrt.crash_hint(proc.poll()) if proc is self.proc else ""
        if hint:  # o Visual C++ do Windows e antigo: tentar de novo nao adianta
            diag_log(f"motor nao abriu ({self.backend}): Visual C++ antigo {winrt.vc_runtime_version()}")
            self.log.append(hint)
            self.state, self.error = "error", hint
            return
        if proc is self.proc and used > MIN_FIT_CTX and any(KV_OOM_RE.search(l) for l in list(self.log)[-40:]):
            # o cache do contexto nao coube na placa: tenta de novo com metade (sem mudar as configuracoes)
            new = max(MIN_FIT_CTX, (used // 2) // 1024 * 1024)
            asked = self.ctx_requested or used
            try:
                self.start(model_rel, s, device, embed, no_mtp=no_mtp, ctx=new, kv=self._kv)
                self.notice = (f"Contexto reduzido de {asked} para {new} tokens para caber na memória da placa de vídeo. "
                               "Para não esperar as tentativas, ajuste em Configurações → Modelo e runtime.")
                self.log.append(f"[Arasaka Mikoshi] Faltou memória para {used} tokens de contexto; tentando com {new}.")
                diag_log(f"contexto reduzido ao carregar {model_rel}: {used} -> {new} (faltou VRAM)")
                return
            except (ValueError, OSError):
                pass
        if proc is self.proc and getattr(self, "kv_type", "f16") != "f16":
            # ha modelos e placas em que o motor recusa o cache em 8 bits: volta para o normal
            try:
                self.start(model_rel, s, device, embed, no_mtp=no_mtp, ctx=used or None, kv="f16")
                self.notice = "O motor não aceitou o cache de contexto em 8 bits com este modelo; carregado com o normal."
                self.log.append("[Arasaka Mikoshi] " + self.notice)
                return
            except (ValueError, OSError):
                pass
        if proc is self.proc and getattr(self, "mtp", False):
            # o motor nao aceitou o acelerador MTP (versao antiga ou arquivo incompativel): tenta sem ele
            note = "[Arasaka Mikoshi] O runtime falhou com o speculative decoding; carregando sem ele."
            try:
                self.start(model_rel, s, device, embed, no_mtp=True, ctx=used or None, kv=self._kv)
                self.log.append(note)
                self.mtp_error = True
                return
            except (ValueError, OSError):
                pass
        if proc is self.proc:
            tail = [l for l in list(self.log)[-15:] if l]
            diag_log(f"motor fechou ao carregar {self.model}: " + " | ".join(tail[-4:]))
            self.state = "error"
            self.error = "O motor fechou. Ultimas linhas do log:\n" + "\n".join(tail)

    def _read_props(self):
        try:
            with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{self.port}/props", headers=ENGINE_HEADERS), timeout=3) as r:
                p = json.loads(r.read().decode("utf-8"))
            n_ctx = int((p.get("default_generation_settings") or {}).get("n_ctx") or p.get("n_ctx") or 0)
            return n_ctx, {k: bool(v) for k, v in (p.get("modalities") or {}).items()}
        except (OSError, ValueError):
            return 0, {}

    def status(self):
        if self.state == "ready" and self.proc and self.proc.poll() is not None:
            self.state, self.error = "error", "O motor fechou inesperadamente (veja o log)."
        return {"state": self.state, "model": self.model, "backend": self.backend, "error": self.error,
                "label": strata.label(self.model) if self.backend == "strata" else "",
                "kv_cache": getattr(self, "kv_type", "f16"),
                "port": self.port, "n_ctx": self.n_ctx, "ctx_requested": self.ctx_requested, "notice": self.notice,
                "mtp": self.mtp, "mtp_error": self.mtp_error, "accel": self.accel,
                "vision": self.state == "ready" and self.modalities.get("vision", False),
                "audio": self.state == "ready" and self.modalities.get("audio", False)}


ENGINE_PROC = EngineProcess()


def model_display(rel):
    if strata.parse_id(rel):
        return strata.label(rel)
    return Path(rel).stem if rel else "modelo principal"


def model_bytes(rel):
    """Tamanho em disco do modelo (com partes e arquivo de visao)."""
    p = MODELS / rel
    if not p.is_file():
        return 0
    m = SPLIT_RE.match(p.name)
    size = sum(q.stat().st_size for q in p.parent.glob(f"{m.group(1)}-*-of-{m.group(3)}.gguf")) if m else p.stat().st_size
    mm, mt = mmproj_for(rel), mtp_for(rel)
    return size + (mm.stat().st_size if mm else 0) + (mt.stat().st_size if mt else 0)


class EnginePool:
    """Modelos extras: cada agente ou a pergunta rapida pode usar um modelo diferente do principal.
    Cada modelo extra roda num llama-server proprio (porta propria) enquanto couber na VRAM;
    os menos usados sao descarregados quando falta memoria ou depois de ficarem parados."""

    def __init__(self):
        self.extra = {}  # caminho do modelo -> EngineProcess
        self.lock = threading.Lock()
        threading.Thread(target=self._reaper, daemon=True).start()

    def _free_port(self):
        used = {ENGINE_PORT} | {e.port for e in self.extra.values()}
        for port in range(ENGINE_PORT + 1, ENGINE_PORT + 60):
            if port in used:
                continue
            with socket.socket() as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError("Sem portas livres para outro modelo.")

    OVERHEAD = 1.3 * 1024 ** 3  # contexto + buffers por modelo (aproximado)

    def _used(self, device):
        """Memoria ocupada pelos modelos carregados no dispositivo (bytes)."""
        alive = ("ready", "loading")
        total = model_bytes(ENGINE_PROC.model) + self.OVERHEAD if ENGINE_PROC.state in alive and ENGINE_PROC.device == device else 0
        return total + sum(model_bytes(e.model) + self.OVERHEAD for e in self.extra.values() if e.state in alive and e.device == device)

    @staticmethod
    def _capacity(device):
        if device == "gpu":
            return GPU.get("vram_mb", 0) * 2 ** 20 * 0.95
        total, _ = ram_info()
        return total * 2 ** 20 * 0.70  # deixa RAM para o Windows e outros programas

    def _fits(self, device, need):
        return self._used(device) + need + self.OVERHEAD <= self._capacity(device)

    @staticmethod
    def kept_model():
        """Modelo da pergunta rapida que deve ficar sempre carregado ("" se nenhum)."""
        s = load_settings()
        return s.get("quick_model", "") if s.get("quick_keep_loaded") else ""

    def make_room_for_main(self, model_rel):
        """Antes de carregar o principal na GPU, descarrega extras da GPU se ele nao couber junto."""
        need = model_bytes(model_rel)
        with self.lock:
            for key, e in sorted(self.extra.items(), key=lambda kv: (kv[0][0] == self.kept_model(), kv[1].last_used)):
                if self._used("gpu") + need + self.OVERHEAD <= self._capacity("gpu"):
                    break
                if e.busy == 0 and e.device == "gpu" and key[0] != model_rel:
                    e.stop()
                    self.extra.pop(key, None)

    def _make_room(self, device, need):
        """Descarrega extras do dispositivo (do menos usado para o mais usado) ate o novo modelo caber.
        O modelo mantido da pergunta rapida e o ultimo a sair."""
        kept = self.kept_model()
        for key, e in sorted(self.extra.items(), key=lambda kv: (kv[0][0] == kept, kv[1].last_used)):
            if self._fits(device, need):
                break
            if e.busy == 0 and e.device == device:
                e.stop()
                self.extra.pop(key, None)

    def _resolve(self, device, need):
        """Escolhe onde rodar: 'auto' = GPU se couber, senao CPU (RAM), senao libera espaco na GPU.
        'gpu' = sempre GPU; se nao couber inteiro, o llama.cpp divide entre VRAM e RAM (-ngl auto)."""
        if not GPU.get("vram_mb"):
            return "cpu"
        if device in ("gpu", "cpu"):
            self._make_room(device, need)
            return device
        if self._fits("gpu", need):
            return "gpu"
        if self._fits("cpu", need):
            return "cpu"
        self._make_room("gpu", need)
        return "gpu"

    def use(self, model_rel, s, on_wait=None, device="auto", embed=False):
        return _PoolLease(self, model_rel, s, on_wait, device, embed)

    def acquire(self, model_rel, s, on_wait=None, device="auto", embed=False):
        """Garante o modelo pronto e devolve o motor (o principal se model_rel for vazio ou igual a ele).
        device: 'auto' (GPU se couber, senao CPU), 'gpu' ou 'cpu'."""
        device = device if device in ("gpu", "cpu") else "auto"
        if not model_rel or (model_rel == ENGINE_PROC.model and device in ("auto", ENGINE_PROC.device)):
            eng = ENGINE_PROC
            if not model_rel and eng.state != "ready":
                raise RuntimeError("Nenhum modelo carregado. Carregue um modelo no topo do app.")
        else:
            if not (MODELS / model_rel).is_file():
                raise RuntimeError(f"O modelo \"{model_display(model_rel)}\" não foi encontrado. Escolha outro.")
            with self.lock:
                def alive(e):
                    return e.state in ("ready", "loading") and not (e.proc and e.proc.poll() is not None)
                cands = [e for (m, d), e in self.extra.items() if m == model_rel and alive(e) and device in ("auto", d)]
                eng = sorted(cands, key=lambda e: e.state != "ready")[0] if cands else None
                if not eng:
                    where = self._resolve(device, model_bytes(model_rel))
                    old = self.extra.pop((model_rel, where), None)
                    if old:
                        old.stop()
                    eng = EngineProcess(self._free_port())
                    eng.start(model_rel, s, device=where, embed=embed)
                    self.extra[(model_rel, where)] = eng
        eng.busy += 1
        eng.last_used = time.time()
        announced = False
        t0 = time.time()
        while eng.state == "loading":
            if on_wait and not on_wait(model_display(model_rel or eng.model), not announced):
                eng.busy -= 1
                raise ConnectionError("cliente desconectou")
            announced = True
            if time.time() - t0 > 600:
                break
            time.sleep(0.5)
        if eng.state != "ready":
            eng.busy -= 1
            err = eng.error or "não carregou"
            if "out of memory" in err.lower() or "failed to allocate" in err.lower():
                err = "Sem memória de vídeo para carregar este modelo junto com o principal. Escolha um modelo menor ou descarregue o principal."
            raise RuntimeError(f"Falha ao carregar \"{model_display(model_rel)}\": {err}")
        return eng

    def release(self, eng):
        eng.busy = max(0, eng.busy - 1)
        eng.last_used = time.time()

    def stop_except(self, keep_model=""):
        with self.lock:
            for key in [k for k in list(self.extra) if k[0] != keep_model and self.extra[k].busy == 0]:
                e = self.extra.pop(key, None)
                if e:
                    e.stop()

    def stop_device(self, device):
        with self.lock:
            for key in [k for k in list(self.extra) if k[1] == device and self.extra[k].busy == 0]:
                e = self.extra.pop(key, None)
                if e:
                    e.stop()

    def stop(self, model_rel=None):
        with self.lock:
            for key in [k for k in list(self.extra) if not model_rel or k[0] == model_rel]:
                e = self.extra.pop(key, None)
                if e:
                    e.stop()

    def status(self):
        return [{"model": m, "name": model_display(m), "device": d, "state": e.status()["state"], "port": e.port,
                 "busy": e.busy, "idle": round(time.time() - e.last_used)} for (m, d), e in self.extra.items()]

    def memory(self):
        total, avail = ram_info()
        return {"vram_used": round(self._used("gpu") / 2 ** 20), "vram_total": GPU.get("vram_mb", 0),
                "ram_used": round(self._used("cpu") / 2 ** 20), "ram_total": total, "ram_avail": avail}

    def quick_status(self):
        s = load_settings()
        model = s.get("quick_model", "")
        e = next((e for (m, d), e in self.extra.items() if m == model and e.state in ("ready", "loading")), None)
        if not e and model and model == ENGINE_PROC.model and ENGINE_PROC.state in ("ready", "loading"):
            e = ENGINE_PROC
        loading = getattr(self, "_quick_loading", False) and not getattr(self, "_quick_cancel", False)
        state = e.state if e else ("loading" if loading else "stopped")
        return {"model": model, "name": model_display(model) if model else "", "keep": bool(s.get("quick_keep_loaded")),
                "device": quick_device(s), "state": state, "on": e.device if e else "",
                "error": getattr(self, "quick_error", "")}

    def load_quick(self):
        """Carrega o modelo da pergunta rapida em segundo plano (se houver um e ainda nao estiver carregado)."""
        s = load_settings()
        model = s.get("quick_model", "")
        if not model or getattr(self, "_quick_loading", False):
            return
        if not (MODELS / model).is_file() or not installed_backends():
            return
        if self.quick_status()["state"] in ("ready", "loading"):
            return

        def run():
            self._quick_loading, self._quick_cancel, self.quick_error = True, False, ""
            for _ in range(600):  # o principal carrega primeiro (prioridade na VRAM)
                if ENGINE_PROC.state != "loading":
                    break
                time.sleep(0.5)
            try:
                self.release(self.acquire(model, s, None, quick_device(s)))
            except (RuntimeError, ValueError, OSError) as e:
                if not self._quick_cancel:  # descarregado de proposito no meio da carga: nao e erro
                    self.quick_error = str(e)
            finally:
                self._quick_loading = False
        threading.Thread(target=run, daemon=True).start()

    def _reaper(self):
        while True:
            time.sleep(30)
            if self.kept_model() and not getattr(self, "quick_error", ""):
                self.load_quick()  # volta a carregar se foi descarregado (ex.: para abrir espaco ao principal)
            limit = int(load_settings().get("extra_idle_minutes") or 0) * 60
            if limit <= 0:
                continue
            kept = self.kept_model()
            with self.lock:
                for key, e in list(self.extra.items()):
                    if e.busy == 0 and time.time() - e.last_used > limit and key[0] != kept:
                        e.stop()
                        self.extra.pop(key, None)


def quick_device(s):
    """Onde roda o modelo da pergunta rapida: GPU, a menos que o usuario escolha CPU."""
    return "cpu" if s.get("quick_device") == "cpu" else "gpu"


class _PoolLease:
    def __init__(self, pool, model_rel, s, on_wait, device="auto", embed=False):
        self.pool, self.model_rel, self.s, self.on_wait, self.device, self.eng = pool, model_rel, s, on_wait, device, None
        self.embed = embed

    def __enter__(self):
        self.eng = self.pool.acquire(self.model_rel, self.s, self.on_wait, self.device, self.embed)
        return self.eng

    def __exit__(self, *exc):
        self.pool.release(self.eng)


POOL = EnginePool()
atexit.register(ENGINE_PROC.stop)
atexit.register(POOL.stop)
atexit.register(mcp.MANAGER.stop)


# ---------------------------------------------------------------- modelos

SPLIT_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.I)


MEDIA_DIRS = ("image", "whisper", "mmproj", "embed", "mtp")


def model_recommended(rel):
    info = load_model_info().get(rel) or catalog.model_caps({"id": Path(rel).name})
    rec, family = modelsettings.recommended(rel, [c["key"] for c in info.get("caps", [])], info.get("ctx_n", 0),
                                            model_bytes(rel), GPU.get("vram_mb", 0) * 2 ** 20)
    fit = fit_context(rel)
    if fit is not None:  # conta de verdade, pela arquitetura do modelo
        rec["ctx_size"] = min(gguf_context_length(MODELS / rel) or EXTRA_CTX, max(fit, EXTRA_CTX))
    return rec, family


def fit_context(rel, kv_type="", free=None):
    """Maior contexto em que o modelo cabe inteiro na placa (0 = nao cabe; None = nao da para estimar).
    model_bytes ja inclui o projetor de visao e o acelerador."""
    if not rel or strata.parse_id(rel) or not GPU.get("vram_mb"):
        return None
    if not kv_type:
        kv_type = modelsettings.get(rel).get("kv_cache") or load_settings().get("kv_cache") or "f16"
    return gguf.best_context(gguf.read_meta(MODELS / rel), model_bytes(rel), 0, GPU["vram_mb"] * 2 ** 20, kv_type, free)


def free_vram(engine=None):
    """VRAM livre (bytes) para `engine`: medida na placa ou, sem a medida, o total menos os outros modelos carregados."""
    others = [e for e in [ENGINE_PROC, *list(POOL.extra.values())] if e is not engine and e.device == "gpu"]
    size = lambda e: model_bytes(e.model) + POOL.OVERHEAD
    sample = gpu_sample()
    if not sample:
        return GPU.get("vram_mb", 0) * 2 ** 20 - gguf.DESKTOP - sum(size(e) for e in others if e.state in ("ready", "loading"))
    for _ in range(4):  # o driver leva um instante para devolver a memoria do modelo que acabou de sair
        time.sleep(0.3)
        again = gpu_sample() or sample
        if again[0] >= sample[0] - 64:
            break
        sample = again
    # um modelo ainda carregando vai ocupar mais do que a medida mostra
    return (sample[1] - sample[0]) * 2 ** 20 - sum(size(e) for e in others if e.state == "loading")


def auto_context(rel, kv_type, engine=None, on_cpu=False):
    """Contexto automatico (ctx_size 0): o maior em que o modelo cabe inteiro na VRAM livre agora. Os modelos extras
    (agentes, pergunta rapida) ficam em ate 16 mil para nao tomar a placa do principal; um modelo que nao cabe tambem:
    parte dele roda na RAM de qualquer jeito."""
    train = gguf_context_length(MODELS / rel) or gguf.CTX_STEPS[-1]
    fit = None if on_cpu else fit_context(rel, kv_type, free_vram(engine) if engine else None)
    if fit is None:
        return min(train, 16384 if on_cpu and gguf_context_length(MODELS / rel) else 8192)
    if engine is not None and engine is not ENGINE_PROC:
        return min(train, fit or 8192, EXTRA_CTX)
    # nunca menos de 16 mil no principal: a conta tem folga, e no limite so um pedaco pequeno vai para a RAM
    return min(train, max(fit, EXTRA_CTX))


def vram_notice(rel, ctx, kv_type):
    """Aviso quando o modelo nao vai caber inteiro na VRAM livre agora (contexto grande ou um jogo aberto)."""
    sample = gpu_sample()
    meta = gguf.read_meta(MODELS / rel)
    need = gguf.needed(meta, model_bytes(rel), 0, ctx, kv_type)
    if not sample or not need:
        return ""
    free = (sample[1] - sample[0]) * 2 ** 20
    if need + gguf.MARGIN <= free:
        return ""
    fit = fit_context(rel, kv_type, free)
    tip = (f" Com contexto de até {fit} tokens ele caberia inteiro." if fit and fit < ctx else
           " Feche programas que usam a placa (jogos) ou use um modelo menor.")
    moe = bool(meta.get(f"{meta.get('general.architecture', '')}.expert_count"))  # so os especialistas pouco usados saem da placa
    return (f"Parte do modelo pode ficar na RAM ({'neste tipo de modelo (MoE) perde pouca velocidade' if moe else 'mais lento'}): com {ctx} tokens de contexto ele precisa de cerca de "
            f"{need / gguf.GIB:.1f} GB e a placa tem {free / gguf.GIB:.1f} GB livres.{tip}")


def load_bench():
    try:
        return json.loads(BENCH_FILE.read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def tune_model(task, rel):
    """Mede o modelo com o llama-bench e guarda o ajuste mais rapido (ver tune.py)."""
    s = model_effective(rel, load_settings())
    backs = installed_backends()
    backend = s.get("backend", "auto")
    if backend not in backs:
        backend = next((b for b in BACKENDS if b in backs), "")
    exe = tune.bench_exe(backs[backend]["exe"]) if backend else None
    if not exe:
        raise ValueError("O motor instalado não traz o llama-bench. Atualize o motor em Modelos → Runtime.")
    if ENGINE_PROC.backend == "strata" and ENGINE_PROC.state != "stopped":
        raise ValueError("Descarregue o Strata antes de otimizar.")
    if ENGINE_PROC.busy or any(e.busy for e in POOL.extra.values()):
        raise ValueError("Espere a resposta atual terminar antes de otimizar.")
    before = ENGINE_PROC.model if ENGINE_PROC.state in ("ready", "loading") else ""
    rows, code, tail = [], 0, ""
    try:
        ENGINE_PROC.stop()
        POOL.stop()
        gpu = backend != "cpu" and int(s["gpu_layers"]) > 0
        if gpu:
            time.sleep(1.5)  # o driver devolve a VRAM do modelo descarregado
            sample = gpu_sample()
            if sample and sample[1] and sample[0] > sample[1] * tune.BUSY_SHARE:
                raise ValueError(f"A placa já tem {sample[0] / 1024:.1f} GB ocupados por outros programas (um jogo aberto?). "
                                 "Feche-os e tente de novo: a medida sairia errada.")
        ctx = int(s["ctx_size"]) or auto_context(rel, "f16")
        ctx = min(ctx, gguf_context_length(MODELS / rel) or ctx)
        plan = [(u, "f16") for u in tune.UBATCHES] + [(512, "q8_0")]
        task["total"] = len(plan)

        def should_stop():
            if task["cancel"]:
                raise Cancelled()
            if ENGINE_PROC.state in ("ready", "loading"):
                raise ValueError("Um modelo foi carregado durante o teste. Tente de novo sem usar a IA.")

        def own(p):
            if ENGINE_PROC.job:  # fecha junto com o app
                try:
                    ENGINE_PROC.job[0].AssignProcessToJobObject(ENGINE_PROC.job[1], int(p._handle))
                except Exception:  # noqa: BLE001
                    pass

        for n, (ub, kv) in enumerate(plan):
            task["detail"] = f"lote {ub}" + (", cache em 8 bits" if kv != "f16" else "")
            res, code, tail = tune.run_one(tune.command(exe, MODELS / rel, ctx, ub, kv, cpu_threads(s), gpu), should_stop, own)
            task["done"] = n + 1
            if res:
                rows.append({"ubatch": ub, "kv": kv, **res})
            elif not rows:
                break  # nem o padrao rodou
        if not rows:
            raise ValueError(winrt.crash_hint(code) or "O teste não rodou com este modelo. " + tail.strip()[-300:])
        best, values = tune.choose(rows)
        saved = {k: v for k, v in modelsettings.get(rel).items() if k not in ("ubatch", "kv_cache")}
        modelsettings.save(rel, {**saved, **values} or None)
        doc = load_bench()
        doc[rel] = {"at": time.time(), "backend": backend, "version": backs[backend].get("version", ""), "ctx": ctx, "rows": rows, "best": best}
        BENCH_FILE.write_text(json.dumps(doc, ensure_ascii=False, indent=1), "utf-8")
        task["detail"] = f"{best['tg']:.0f} tok/s"
    finally:
        if before and ENGINE_PROC.state == "stopped":
            try:
                ENGINE_PROC.start(before, load_settings())
            except (ValueError, OSError):
                pass


def model_effective(rel, s):
    rec, _ = model_recommended(rel)
    return modelsettings.effective(s, rel, rec, s.get("use_recommended", True))


def apply_model_sampling(payload, rel, agent_temperature=None):
    """Aplica a amostragem do modelo (recomendada/salva) sobre o pedido; a temperatura do agente vence."""
    if rel:
        s = load_settings()
        rec, _ = model_recommended(rel)
        payload.update(modelsettings.sampling_for(rel, rec, s.get("use_recommended", True)))
    if agent_temperature is not None:
        payload["temperature"] = agent_temperature
    return payload


def embed_texts(texts, on_wait=None):
    """Vetores (embeddings) dos textos, com o modelo de embeddings rodando num motor proprio."""
    path = media.embed_model_path()
    if not path:
        raise RuntimeError("Modelo de embeddings não baixado. Baixe em RAG.")
    rel = path.relative_to(MODELS).as_posix()
    with POOL.use(rel, load_settings(), on_wait, "auto", embed=True) as eng:
        req = urllib.request.Request(f"http://127.0.0.1:{eng.port}/v1/embeddings",
                                     data=json.dumps({"input": texts, "model": "embed"}).encode("utf-8"),
                                     headers=ENGINE_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError("Falha ao gerar embeddings: " + e.read().decode("utf-8", "replace")[:200]) from None
    return [d["embedding"] for d in sorted(data["data"], key=lambda d: d.get("index", 0))]


def cleanup_checkpoints(days=30):
    if not CHECKPOINTS.exists():
        return
    limit = time.time() - days * 86400
    for chat_dir in CHECKPOINTS.iterdir():
        for run in list(chat_dir.iterdir()) if chat_dir.is_dir() else []:
            if run.stat().st_mtime < limit:
                shutil.rmtree(run, ignore_errors=True)
        if chat_dir.is_dir() and not any(chat_dir.iterdir()):
            chat_dir.rmdir()


def cpu_threads(s):
    """Threads da CPU pela porcentagem escolhida (0 = deixa o motor decidir)."""
    if int(s.get("threads") or 0) > 0:
        return int(s["threads"])
    pct = int(s.get("cpu_percent") or 0)
    return max(1, round(CPU_COUNT * min(pct, 100) / 100)) if pct > 0 else 0


def load_model_info():
    try:
        return json.loads(MODEL_INFO_FILE.read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def save_model_info(model_rel, info):
    d = load_model_info()
    d[model_rel] = info
    MODEL_INFO_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=1), "utf-8")


def hf_model_info(repo):
    m = get_json(f"https://huggingface.co/api/models/{urllib.parse.quote(repo, safe='/')}?{catalog.HF_EXPAND}")
    return {"repo": repo, **catalog.model_caps(m)}


def list_models():
    out = []
    mm = load_mmproj_map()
    mtps = load_mtp_map()
    infos = load_model_info()
    for p in sorted(MODELS.rglob("*.gguf"), key=lambda x: x.name.lower()):
        rel_parts = p.relative_to(MODELS).parts
        if is_aux_file(p.relative_to(MODELS).as_posix(), p.stat().st_size) or (len(rel_parts) > 1 and rel_parts[0] in MEDIA_DIRS):
            continue
        m = SPLIT_RE.match(p.name)
        size = p.stat().st_size
        if m:
            if m.group(2) != "00001":
                continue
            size = sum(q.stat().st_size for q in p.parent.glob(f"{m.group(1)}-*-of-{m.group(3)}.gguf"))
        rel = p.relative_to(MODELS).as_posix()
        info = infos.get(rel) or {"repo": "", **catalog.model_caps({"id": p.name})}
        out.append({"name": p.stem if not m else m.group(1), "path": rel, "size": size,
                    "mmproj": mm.get(rel, "") if (MODELS / mm.get(rel, "_")).is_file() else "",
                    "mtp": mtps.get(rel, "") if mtps.get(rel) == ACCEL_NGRAM or (MODELS / mtps.get(rel, "_")).is_file() else "",
                    "custom": bool(modelsettings.get(rel)),
                    "info": info, "quant": catalog.quant_info(p.name)})
    return out


def load_mmproj_map():
    try:
        return json.loads(MMPROJ_FILE.read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def set_mmproj(model_rel, mm_rel):
    m = load_mmproj_map()
    if mm_rel:
        m[model_rel] = mm_rel
    else:
        m.pop(model_rel, None)
    MMPROJ_FILE.write_text(json.dumps(m, indent=1), "utf-8")


def mmproj_for(model_rel):
    rel = load_mmproj_map().get(model_rel)
    p = (MODELS / rel).resolve() if rel else None
    return p if p and p.is_file() and MODELS.resolve() in p.parents else None


def list_mmproj():
    return sorted(p.relative_to(MODELS).as_posix() for p in MODELS.rglob("*.gguf") if "mmproj" in p.name.lower())


MTP_RE = re.compile(r"(^|[-_.])(fast)?mtp([-_.]|$)", re.I)


MTP_MAX_BYTES = 4 * 1024 ** 3  # cabecas MTP tem de 0,4 a ~1,5 GB; um "-mtp.gguf" maior e um modelo inteiro com MTP embutido


def is_mtp(path, size=None):
    """Cabeca MTP (Multi-Token Prediction): acelerador do modelo grande, nao funciona sozinho.
    Arquivos grandes com "mtp" no nome sao modelos completos (com o acelerador embutido), nao cabecas."""
    if size is not None and size > MTP_MAX_BYTES:
        return False
    p = path.replace("\\", "/").lower()
    return p.startswith("mtp/") or "/mtp/" in p or bool(MTP_RE.search(p.rsplit("/", 1)[-1][:-5]))


def is_aux_file(path, size=None):
    """Arquivos .gguf que nao sao modelos de conversa (visao, MTP, matriz de importancia)."""
    name = path.rsplit("/", 1)[-1].lower()
    return "mmproj" in name or "imatrix" in name or is_mtp(path, size)


def pick_mtp(files, model_path=""):
    """Melhor arquivo MTP do repositorio: mesma quantizacao do modelo, senao Q8_0, Q4_0 e o menor."""
    mtp = [f for f in files if is_mtp(f["path"], f.get("size"))]
    if not mtp:
        return None
    q = (catalog.quant_info(model_path.rsplit("/", 1)[-1]) or {}).get("quant", "").lower() if model_path else ""

    def score(f):
        n = f["path"].lower()
        return (0 if q and q in n else 1, 0 if "q8_0" in n else 1 if "q4_0" in n else 3 if re.search(r"bf16|f16|f32", n) else 2,
                f["size"])
    return min(mtp, key=score)


def load_mtp_map():
    try:
        return json.loads(MTP_FILE.read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def set_mtp(model_rel, mtp_rel):
    m = load_mtp_map()
    if mtp_rel:
        m[model_rel] = mtp_rel
    else:
        m.pop(model_rel, None)
    MTP_FILE.write_text(json.dumps(m, indent=1), "utf-8")


ACCEL_NGRAM = "ngram"  # acelerador sem modelo extra: reaproveita trechos que ja apareceram no texto


def mtp_for(model_rel):
    """Arquivo do acelerador (cabeca MTP ou modelo rascunho) ligado ao modelo, se houver."""
    rel = load_mtp_map().get(model_rel)
    if not rel or rel == ACCEL_NGRAM:
        return None
    p = (MODELS / rel).resolve()
    return p if p.is_file() and MODELS.resolve() in p.parents else None


def accel_for(model_rel):
    """(tipo, arquivo) do acelerador: mtp (cabeca MTP), draft (modelo rascunho pequeno da mesma familia) ou ngram."""
    if load_mtp_map().get(model_rel) == ACCEL_NGRAM:
        return "ngram", None
    p = mtp_for(model_rel)
    if not p:
        return "", None
    rel = p.relative_to(MODELS.resolve()).as_posix()
    return ("mtp" if is_mtp(rel, p.stat().st_size) else "draft"), p


def accel_args(kind, path):
    if kind == "mtp":  # a cabeca MTP adivinha os proximos tokens e o modelo confere
        return ["--spec-type", "draft-mtp", "-md", str(path), "-ngld", "all"]
    if kind == "draft":  # um modelo pequeno do mesmo vocabulario escreve um rascunho e o grande confere de uma vez
        return ["--spec-type", "draft-simple", "-md", str(path), "-ngld", "all", "--spec-draft-n-max", "8"]
    if kind == "ngram":  # sem modelo extra: bom quando a resposta repete o pedido (editar codigo, reescrever)
        return ["--spec-type", "ngram-map-k"]
    return []


def valid_accel(model_rel, value):
    """Aceita: vazio, n-gram, uma cabeca MTP ou outro modelo de conversa (rascunho) menor que o principal."""
    if not value or value == ACCEL_NGRAM or value in list_mtp():
        return True
    models = {m["path"]: m["size"] for m in list_models()}
    return value in models and value != model_rel and models[value] < models.get(model_rel, 0)


def list_mtp():
    return sorted(p.relative_to(MODELS).as_posix() for p in MODELS.rglob("*.gguf")
                  if is_mtp(p.relative_to(MODELS).as_posix(), p.stat().st_size))


def hf_tree(repo):
    items = get_json(f"https://huggingface.co/api/models/{urllib.parse.quote(repo, safe='/')}/tree/main?recursive=true")
    return [{"path": it["path"], "size": (it.get("lfs") or {}).get("size") or it.get("size", 0)}
            for it in items if it.get("type") == "file" and it["path"].lower().endswith(".gguf")]


def hf_files(repo, tree=None):
    files = [f for f in (tree if tree is not None else hf_tree(repo)) if not is_aux_file(f["path"], f.get("size"))]
    # agrupa arquivos divididos
    groups = {}
    for f in files:
        m = SPLIT_RE.match(f["path"])
        key = m.group(1) + ".gguf" if m else f["path"]
        g = groups.setdefault(key, {"path": f["path"], "size": 0, "parts": 0})
        g["size"] += f["size"]
        g["parts"] += 1
        if m and m.group(2) == "00001":
            g["path"] = f["path"]
    return sorted(groups.values(), key=lambda g: g["size"])


def pick_mmproj(files):
    mm = [f for f in files if "mmproj" in f["path"].lower()]
    for pref in ("f16", "bf16", "q8", ""):
        for f in mm:
            if pref in f["path"].lower():
                return f
    return None


def download_model(task, repo, path, with_mmproj=False, with_mtp=False):
    files = hf_tree(repo)
    try:
        info = hf_model_info(repo)
    except Exception:  # noqa: BLE001
        info = None
    m = SPLIT_RE.match(path)
    if m:
        parts = sorted((f["path"], f["size"]) for f in files
                       if SPLIT_RE.match(f["path"]) and SPLIT_RE.match(f["path"]).group(1) == m.group(1))
    else:
        parts = [(path, next((f["size"] for f in files if f["path"] == path), 0))]
    task["total"] = sum(s for _, s in parts)
    folder = MODELS / repo.split("/")[-1] if m else MODELS
    if info:
        save_model_info((folder / Path(parts[0][0]).name).relative_to(MODELS).as_posix(), info)
    folder.mkdir(parents=True, exist_ok=True)
    done = 0
    for p, _ in parts:
        dest = folder / Path(p).name
        task["detail"] = Path(p).name
        if dest.exists():
            done += dest.stat().st_size
            task["done"] = done
            continue
        url = f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(p)}"
        done += download(url, dest, task, done)
    mm = pick_mmproj(files) if with_mmproj else None
    if mm:
        mm_dest = MODELS / "mmproj" / f"{repo.split('/')[-1]}__{Path(mm['path']).name}"
        mm_dest.parent.mkdir(parents=True, exist_ok=True)
        task["detail"] = "arquivo de visao (mmproj)"
        task["total"] += mm["size"]
        if not mm_dest.exists():
            download(f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(mm['path'])}", mm_dest, task, done)
        set_mmproj((folder / Path(parts[0][0]).name).relative_to(MODELS).as_posix(),
                   mm_dest.relative_to(MODELS).as_posix())
    mt = pick_mtp(files, path) if with_mtp else None
    if mt:
        mt_dest = MODELS / "mtp" / f"{repo.split('/')[-1]}__{Path(mt['path']).name}"
        mt_dest.parent.mkdir(parents=True, exist_ok=True)
        task["detail"] = "cabeça MTP (speculative decoding)"
        task["total"] += mt["size"]
        if not mt_dest.exists():
            download(f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(mt['path'])}", mt_dest, task, done)
        set_mtp((folder / Path(parts[0][0]).name).relative_to(MODELS).as_posix(), mt_dest.relative_to(MODELS).as_posix())


# ---------------------------------------------------------------- conversas

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


_chats_cache = {}  # arquivo -> ((data, tamanho), resumo): a tela pede a lista a cada 1,5 s


def list_chats():
    out, seen = [], set()
    try:
        entries = [e for e in os.scandir(CHATS) if e.name.endswith(".json")]
    except OSError:
        return []
    for e in entries:
        try:
            st = e.stat()
            sig = (st.st_mtime_ns, st.st_size)
            hit = _chats_cache.get(e.name)
            if not hit or hit[0] != sig:
                c = json.loads(Path(e.path).read_text("utf-8"))
                hit = (sig, {"id": c["id"], "title": c.get("title", ""), "updated": c.get("updated", 0)})
                _chats_cache[e.name] = hit
            seen.add(e.name)
            out.append(dict(hit[1]))
        except (OSError, ValueError, KeyError, TypeError):
            pass
    for name in [n for n in list(_chats_cache) if n not in seen]:
        _chats_cache.pop(name, None)
    return sorted(out, key=lambda c: c["updated"], reverse=True)


# ---------------------------------------------------------------- anexos -> mensagens do modelo

IMG_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}


def _image_part(fid):
    p = media.upload_path(fid)
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{IMG_MIME.get(p.suffix.lower(), 'image/png')};base64,{b64}"}}


def _audio_part(fid):
    p = media.upload_path(fid)
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(p.read_bytes()).decode("ascii"),
                                                   "format": "mp3" if p.suffix.lower() == ".mp3" else "wav"}}


def build_content(m, mod):
    atts = m.get("attachments") or []
    if not atts:
        return m.get("content", "")
    texts, parts = [], []
    for a in atts:
        kind, name = a.get("kind"), a.get("name", "arquivo")
        try:
            if kind == "text":
                texts.append(f"[Conteudo do arquivo {name}]\n{a.get('text', '')}\n[Fim do arquivo {name}]")
            elif kind == "image":
                if mod.get("vision"):
                    parts.append(_image_part(a["id"]))
                else:
                    texts.append(f"[Imagem anexada '{name}': o modelo carregado nao tem visao e nao pode ve-la]")
            elif kind == "audio":
                if mod.get("audio") and a.get("id"):
                    parts.append(_audio_part(a["id"]))
                elif a.get("transcript"):
                    texts.append(f"[Transcricao do audio {name}]\n{a['transcript']}")
                else:
                    texts.append(f"[Audio anexado '{name}' sem transcricao disponivel]")
            elif kind == "video":
                frames = a.get("frames") or []
                if frames and mod.get("vision"):
                    stamps = ", ".join(f"{f.get('t', 0):.0f}s" for f in frames)
                    texts.append(f"[Video '{name}' ({a.get('duration', 0):.0f}s): {len(frames)} quadros anexados em ordem, "
                                 f"nos tempos {stamps}]")
                    parts += [_image_part(f["id"]) for f in frames]
                if a.get("transcript"):
                    texts.append(f"[Transcricao do audio do video {name}]\n{a['transcript']}")
                if not (frames and mod.get("vision")) and not a.get("transcript"):
                    texts.append(f"[Video anexado '{name}': o modelo nao tem visao e nao ha transcricao]")
        except ValueError:
            texts.append(f"[Anexo '{name}' nao encontrado]")
    text = "\n\n".join(texts + ([m["content"]] if m.get("content") else []))
    return parts + [{"type": "text", "text": text or "(veja os anexos)"}] if parts else text


def text_of(content):
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
    return content or ""


def build_history(msgs, mod, agent_mode=False, budget=0):
    """budget (Codex): enquanto o historico couber nestes caracteres, nada e encurtado. Encurtar a resposta antiga a cada
    pedido mudava o comeco da conversa e fazia o motor reler tudo."""
    out, n = [], len(msgs)
    if agent_mode and budget:
        full = promptcache.text_len([x for m in msgs for x in (m.get("trace") or [m])])
        if full <= budget:
            n = 0  # todas as respostas entram inteiras
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role == "system":
            out.append({"role": "system", "content": m.get("content", "")})
        elif role == "user":
            out.append({"role": "user", "content": build_content(m, mod)})
        elif role == "assistant":
            if m.get("trace") and agent_mode:
                out += agent.compact_history(m["trace"], keep_full=i >= n - 2)
            elif m.get("image"):
                out.append({"role": "assistant", "content": f"[Imagem gerada com o prompt: {m['image'].get('prompt', '')}]"})
            elif m.get("content"):
                out.append({"role": "assistant", "content": m["content"]})
    return out


# ---------------------------------------------------------------- backup das conversas

BACKUP_FORMAT = 1
BACKUP_APP = "Arasaka Mikoshi"
BACKUP_APPS = (BACKUP_APP, "LLM Local")  # backups feitos ate a 1.32.0 dizem "LLM Local"


def backup_info():
    chats = list(CHATS.glob("*.json"))
    ids = set()
    for f in chats:
        try:
            ids |= upload_ids(json.loads(f.read_text("utf-8")))
        except (OSError, ValueError):
            pass
    files = [media.UPLOADS / i for i in ids if (media.UPLOADS / i).is_file()]
    return {"chats": len(chats), "chats_size": sum(f.stat().st_size for f in chats),
            "files": len(files), "files_size": sum(f.stat().st_size for f in files)}


def build_backup(dest, with_files):
    chats = sorted(CHATS.glob("*.json"))
    ids = set()
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in chats:
            z.write(f, f"chats/{f.name}")
            try:
                ids |= upload_ids(json.loads(f.read_text("utf-8")))
            except (OSError, ValueError):
                pass
        n_files = 0
        if with_files:
            for i in sorted(ids):
                fp = media.UPLOADS / i
                if fp.is_file():
                    # midia ja comprimida nao precisa de nova compressao
                    z.write(fp, f"uploads/{i}", compress_type=zipfile.ZIP_STORED if fp.suffix in (".png", ".jpg", ".jpeg", ".webp", ".mp3") else zipfile.ZIP_DEFLATED)
                    n_files += 1
        s = {k: v for k, v in load_settings().items() if k not in SECRETS}
        z.writestr("settings.json", json.dumps(s, ensure_ascii=False, indent=1))
        if agents.FILE.exists():
            z.write(agents.FILE, "agents.json")
        z.writestr("manifest.json", json.dumps({"app": BACKUP_APP, "format": BACKUP_FORMAT, "created": time.time(),
                                                "chats": len(chats), "files": n_files}, indent=1))


def restore_backup(src, with_settings=False):
    added = updated = skipped = files = 0
    with zipfile.ZipFile(src) as z:
        try:
            manifest = json.loads(z.read("manifest.json"))
        except (KeyError, ValueError):
            raise ValueError("Este arquivo nao e um backup do Arasaka Mikoshi.")
        if manifest.get("app") not in BACKUP_APPS:
            raise ValueError("Este arquivo nao e um backup do Arasaka Mikoshi.")
        for name in z.namelist():
            m = re.match(r"^chats/([A-Za-z0-9_-]{1,64})\.json$", name)
            if m:
                try:
                    chat = json.loads(z.read(name).decode("utf-8"))
                except ValueError:
                    skipped += 1
                    continue
                chat["id"] = m.group(1)
                dest = CHATS / f"{m.group(1)}.json"
                if dest.exists():
                    try:
                        cur = json.loads(dest.read_text("utf-8"))
                    except (OSError, ValueError):
                        cur = {}
                    if float(chat.get("updated") or 0) <= float(cur.get("updated") or 0):
                        skipped += 1
                        continue
                    updated += 1
                else:
                    added += 1
                dest.write_text(json.dumps(chat, ensure_ascii=False, indent=1), "utf-8")
                continue
            m = re.match(r"^uploads/([a-f0-9]{20}\.[a-z0-9]{1,6})$", name)
            if m and not (media.UPLOADS / m.group(1)).exists():
                with z.open(name) as fsrc, open(media.UPLOADS / m.group(1), "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
                files += 1
        if with_settings and "settings.json" in z.namelist():
            try:
                s = json.loads(z.read("settings.json"))
                s.pop("workspaces", None)  # pastas do outro PC podem nao existir aqui
                save_settings({k: v for k, v in s.items() if k not in SECRETS})
            except ValueError:
                pass
        n_agents = 0
        if "agents.json" in z.namelist():
            try:
                incoming, cur = json.loads(z.read("agents.json")), agents.load()
                have_a, have_g = {a["id"] for a in cur["agents"]}, {g["id"] for g in cur["groups"]}
                new_a = [a for a in incoming.get("agents", []) if a.get("id") not in have_a]
                new_g = [g for g in incoming.get("groups", []) if g.get("id") not in have_g]
                if new_a or new_g:
                    agents.save({"agents": cur["agents"] + new_a, "groups": cur["groups"] + new_g})
                n_agents = len(new_a) + len(new_g)
            except (ValueError, KeyError, TypeError):
                pass
    return {"added": added, "updated": updated, "skipped": skipped, "files": files, "agents": n_agents}


def upload_ids(obj):
    return set(re.findall(r'"([a-f0-9]{20}\.[a-z0-9]{1,6})"', json.dumps(obj)))


# ---------------------------------------------------------------- HTTP

# As rotas e os metodos grandes ficam em routes.py, chat_routes.py e media_routes.py (mixins da classe Handler).
# Eles chamam este modulo como "server"; rodando como programa (python server.py) ele e o "__main__".
sys.modules.setdefault("server", sys.modules[__name__])
import chat_routes  # noqa: E402
import media_routes  # noqa: E402
import routes  # noqa: E402


MIME = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
        ".mjs": "application/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon", ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif", ".wav": "audio/wav",
        ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".json": "application/json", ".txt": "text/plain; charset=utf-8"}


class Handler(routes.ApiRoutes, media_routes.MediaRoutes, chat_routes.ChatRoutes, BaseHTTPRequestHandler):
    server_version = "LLMLocal"

    def log_message(self, fmt, *args):
        pass

    # helpers
    def send_json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    MAX_JSON = 256 << 20  # 256 MB: cabe uma conversa longa com imagens; arquivos grandes vao por /api/upload (em partes)

    def body(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ValueError("Content-Length inválido") from None
        if not n:
            return {}
        if n < 0 or n > self.MAX_JSON:
            self.close_connection = True  # o corpo nao e lido; a conexao nao pode ser reaproveitada
            raise ValueError(f"pedido grande demais ({n // (1 << 20)} MB; o limite é {self.MAX_JSON >> 20} MB)")
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def check_origin(self):
        # bloqueia sites externos de usarem seu servidor local
        origin = self.headers.get("Origin")
        return not origin or origin in (f"http://{HOST}:{PORT}", f"http://localhost:{PORT}")

    def check_host(self):
        """Protege contra DNS rebinding: um site que faz o proprio dominio apontar para 127.0.0.1 manda o Host
        dele (e, num GET, nenhum Origin). So aceita os nomes locais."""
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].lower()
        if host in (HOST, "localhost"):
            return True
        self.send_json({"error": "host bloqueado"}, 403)
        return False

    # rotas
    def do_GET(self):
        if not self.check_host():
            return
        url = urllib.parse.urlparse(self.path)
        p, q = url.path, urllib.parse.parse_qs(url.query)
        try:
            route = routes.GET_ROUTES.get(p)  # rotas exatas: tabela no fim da classe
            if route:
                return route(self, p, q)
            if p.startswith("/api/uploads/"):
                try:
                    return self.send_file(media.upload_path(p.rsplit("/", 1)[-1]))
                except ValueError:
                    return self.send_json({"error": "nao encontrado"}, 404)
            if p.startswith("/api/chats/"):
                cid = p.rsplit("/", 1)[-1]
                f = CHATS / f"{cid}.json"
                if not ID_RE.match(cid) or not f.exists():
                    return self.send_json({"error": "nao encontrado"}, 404)
                return self.send_json(json.loads(f.read_text("utf-8")))
            return self.static(p)
        except urllib.error.HTTPError as e:
            self.send_json({"error": f"HTTP {e.code}: {e.reason}"}, 502)
        except (urllib.error.URLError, OSError) as e:
            self.send_json({"error": f"Sem conexao: {e}"}, 502)

    def static(self, p):
        if p == "/":
            p = "/index.html"
        f = (WEB / p.lstrip("/")).resolve()
        if WEB.resolve() not in f.parents or not f.is_file():
            return self.send_json({"error": "nao encontrado"}, 404)
        self.send_file(f)

    def send_file(self, f):
        data = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(f.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if not self.check_host():
            return
        if not self.check_origin():
            return self.send_json({"error": "origem bloqueada"}, 403)
        p = urllib.parse.urlparse(self.path).path
        if p == "/api/upload":
            return self.handle_upload()
        if p == "/mcp":
            return self.mikoshi_mcp()
        if p == "/api/restore":
            return self.handle_restore(urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query))
        try:
            b = self.body()
            route = routes.POST_ROUTES.get(p)  # rotas exatas: tabela no fim da classe
            if route:
                return route(self, p, b)
            if p.startswith("/api/train/"):
                return self.train_post(p, b)
            return self.send_json({"error": "rota desconhecida"}, 404)
        except ValueError as e:
            self.send_json({"error": str(e)}, 400)

    def do_PUT(self):
        if not self.check_host():
            return
        if not self.check_origin():
            return self.send_json({"error": "origem bloqueada"}, 403)
        p = urllib.parse.urlparse(self.path).path
        try:
            b = self.body()
        except ValueError as e:
            return self.send_json({"error": str(e)}, 400)
        if p == "/api/agents":
            return self.send_json(agents.save(b))
        if p == "/api/mcp":
            try:
                mcp.save(b.get("servers", []))
            except mcp.MCPError as e:
                return self.send_json({"error": str(e)}, 400)
            return self.send_json(self.mcp_status())
        if p == "/api/models/settings":
            return self.send_json({"settings": modelsettings.save(b.get("model", ""), b.get("settings"))})
        if p.startswith("/api/kb/"):
            try:
                return self.send_json(kb.update(p.rsplit("/", 1)[-1], b.get("name"), b.get("folders")))
            except (ValueError, OSError) as e:
                return self.send_json({"error": str(e)}, 400)
        if p == "/api/settings":
            for k in SECRETS:
                if b.get(k) == "***":
                    b.pop(k)
            if "workspaces" in b:
                b["workspaces"] = sorted({str(Path(w).resolve()) for w in b["workspaces"] if w and Path(w).is_dir()})
            if b.get("api_require_key") and not load_settings().get("api_key"):
                import secrets
                b["api_key"] = "pk-" + secrets.token_urlsafe(24)
            saved = save_settings(b)
            if any(k.startswith("api_") for k in b):
                gateway.GATEWAY.apply()  # liga, desliga ou muda a porta da API local
            return self.send_json(masked(saved))
        if p.startswith("/api/chats/"):
            cid = p.rsplit("/", 1)[-1]
            if not ID_RE.match(cid):
                return self.send_json({"error": "id invalido"}, 400)
            b["id"] = cid
            b["updated"] = time.time()
            (CHATS / f"{cid}.json").write_text(json.dumps(b, ensure_ascii=False, indent=1), "utf-8")
            return self.send_json({"ok": True, "updated": b["updated"]})
        self.send_json({"error": "rota desconhecida"}, 404)

    def do_DELETE(self):
        if not self.check_host():
            return
        if not self.check_origin():
            return self.send_json({"error": "origem bloqueada"}, 403)
        p = urllib.parse.urlparse(self.path).path
        if p.startswith("/api/kb/"):
            try:
                kb.delete(p.rsplit("/", 1)[-1])
            except ValueError:
                pass
            return self.send_json({"ok": True})
        if p.startswith("/api/chats/"):
            cid = p.rsplit("/", 1)[-1]
            f = CHATS / f"{cid}.json"
            if ID_RE.match(cid):
                shutil.rmtree(CHECKPOINTS / cid, ignore_errors=True)
            if ID_RE.match(cid) and f.exists():
                try:
                    for fid in upload_ids(json.loads(f.read_text("utf-8"))):
                        (media.UPLOADS / fid).unlink(missing_ok=True)
                except (OSError, ValueError):
                    pass
                f.unlink()
            return self.send_json({"ok": True})
        self.send_json({"error": "rota desconhecida"}, 404)

# ---------------------------------------------------------------- estudio de treino

def train_chat(prompt):
    """Pergunta ao modelo carregado (usado para gerar exemplos de treino)."""
    with POOL.use("", load_settings()) as eng:
        req = urllib.request.Request(f"http://127.0.0.1:{eng.port}/v1/chat/completions", headers=ENGINE_HEADERS,
                                     data=json.dumps({"messages": [{"role": "user", "content": prompt}], "stream": False,
                                                      "temperature": 0.7, "max_tokens": 3000,
                                                      "chat_template_kwargs": {"enable_thinking": False}}).encode("utf-8"))
        with urllib.request.urlopen(req, timeout=900) as r:
            data = json.loads(r.read().decode("utf-8"))
    add_stats(data.get("timings"))
    return data["choices"][0]["message"].get("content") or ""


def train_free_gpu():
    POOL.stop()
    ENGINE_PROC.stop()


def train_llama_dir():
    b = installed_backends()
    for k in ("cuda", "vulkan", "cpu"):
        if k in b:
            return str(Path(b[k]["exe"]).parent)
    return ""


def train_engine_version():
    b = installed_backends()
    return next((b[k]["version"] for k in ("cuda", "vulkan", "cpu") if k in b), "")


def train_on_model(path, run):
    """Modelo exportado: registra as informacoes para aparecer em Meus modelos com a etiqueta de treinado."""
    rel = Path(path).relative_to(MODELS).as_posix()
    info = {"repo": run.cfg.get("base_repo", ""), **catalog.model_caps({"id": run.cfg.get("base_repo", "")}),
            "trained": {"run": run.id, "dataset": run.cfg.get("dataset_name"), "base": run.cfg.get("base_repo"),
                        "draft_for": run.cfg.get("draft_for", "")}}
    save_model_info(rel, info)
    try:
        link_trained_accel(run, rel)
    except (OSError, ValueError) as e:
        diag_log(f"acelerador treinado: {e}")


def _run_export_rel(run_id):
    """Modelo .gguf exportado por uma execucao do Estudio (relativo a models), ou ''."""
    f = DATA / "train" / "runs" / run_id / "run.json"
    out = (json.loads(f.read_text("utf-8")).get("export") or {}).get("out_path", "") if f.is_file() else ""
    p = Path(out) if out else None
    return p.resolve().relative_to(MODELS.resolve()).as_posix() if p and p.is_file() and MODELS.resolve() in p.resolve().parents else ""


def link_trained_accel(run, rel):
    """Acelerador treinado (rascunho) exportado: liga ao modelo da execucao de origem; e o contrario."""
    parent = run.cfg.get("draft_for")
    if parent:
        target = _run_export_rel(parent)
        if target:
            set_mtp(target, rel)
        return
    runs = DATA / "train" / "runs"
    for d in runs.iterdir() if runs.is_dir() else []:
        try:
            cfg = json.loads((d / "config.json").read_text("utf-8"))
        except (OSError, ValueError):
            continue
        if cfg.get("draft_for") == run.id and _run_export_rel(d.name):
            set_mtp(rel, _run_export_rel(d.name))


def configure_train():
    train.configure(ENGINE, DATA, MODELS, download=lambda url, dest, task: download(url, dest, task), get_json=get_json,
                    llama_dir=train_llama_dir, engine_version=train_engine_version, chat=train_chat, free_gpu=train_free_gpu,
                    diag=diag_log, on_model=train_on_model)


def watch_parent():
    """Encerra o servidor quando o app desktop que o abriu fecha (mesmo se fechar a forca)."""
    pid = int(os.environ.get("LLM_PARENT_PID") or 0)
    if not pid:
        return
    if os.name != "nt":  # Linux: confere a cada 2 s se o app ainda existe
        def poll():
            while True:
                time.sleep(2)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                except PermissionError:
                    continue
            mcp.MANAGER.stop()
            POOL.stop()
            ENGINE_PROC.stop()
            os._exit(0)
        threading.Thread(target=poll, daemon=True).start()
        return
    import ctypes
    k32 = ctypes.WinDLL("kernel32")
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    handle = k32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        return

    def wait():
        k32.WaitForSingleObject(handle, 0xFFFFFFFF)
        mcp.MANAGER.stop()
        POOL.stop()
        ENGINE_PROC.stop()
        os._exit(0)

    threading.Thread(target=wait, daemon=True).start()


def auto_load():
    s = load_settings()
    if s.get("quick_keep_loaded") and s.get("quick_model"):
        def later():
            for _ in range(240):  # espera o principal terminar de carregar (ele tem prioridade na VRAM)
                if ENGINE_PROC.state != "loading":
                    break
                time.sleep(0.5)
            time.sleep(2)
            POOL.load_quick()
        threading.Thread(target=later, daemon=True).start()
    if s.get("auto_load") and s.get("last_model") and installed_backends() and (MODELS / s["last_model"]).is_file():
        try:
            ENGINE_PROC.start(s["last_model"], s)
        except (ValueError, OSError) as e:
            print("Nao foi possivel carregar o ultimo modelo:", e)


def main():
    url = f"http://{HOST}:{PORT}"
    try:
        httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError:
        sys.exit(f"A porta {PORT} ja esta em uso (o Arasaka Mikoshi ja esta aberto?).")
    httpd.daemon_threads = True
    watch_parent()
    threading.Thread(target=auto_load, daemon=True).start()
    threading.Thread(target=cleanup_checkpoints, daemon=True).start()
    threading.Thread(target=diag_sampler, daemon=True).start()
    if os.environ.get("MIKOSHI_NO_CONSTRUCT") != "1":
        connect_construct_later()
    configure_train()
    gateway.GATEWAY.configure(sys.modules[__name__])
    gateway.GATEWAY.apply()
    print(f"Servidor em {url} | motor em http://127.0.0.1:{ENGINE_PORT} (com chave; outros programas usam a API local)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        gateway.GATEWAY.stop()
        mcp.MANAGER.stop()
        POOL.stop()
        ENGINE_PROC.stop()


if __name__ == "__main__":
    main()
