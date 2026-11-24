"""Cabecalho do GGUF e estimativa da memoria do contexto (cache KV) de cada modelo.

O mesmo contexto custa muito diferente conforme a arquitetura: 64 mil tokens ocupam ~7,8 GB no Llama 3.1 8B e menos de
2 GB no Gemma 4 12B (janela deslizante) ou num Qwen hibrido (so 1 em cada 4 camadas tem cache de atencao).
"""
import os
import struct
from pathlib import Path

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
KV_BYTES = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}  # bytes por valor no cache
GIB = 1024 ** 3
COMPUTE = 0.7 * GIB  # buffers de calculo do motor (aproximado)
MARGIN = 1.0 * GIB  # folga que o proprio llama.cpp deixa (--fit-target)
DESKTOP = 1.0 * GIB  # o Windows, o navegador e a area de trabalho
CTX_STEPS = (4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536)
_cache = {}


def _read(path, meta):
    """Preenche meta com o que der para ler (um arquivo cortado no meio ainda pode ter as chaves que interessam)."""
    with open(path, "rb") as f:
        def unpack(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))

        if f.read(4) != b"GGUF":
            return
        version, = unpack("I")
        if version < 2:
            return
        _tensors, count = unpack("QQ")

        def text(limit):
            n, = unpack("Q")
            if n > limit:
                raise ValueError("texto grande demais")
            return f.read(n).decode("utf-8", "replace")

        def skip(t):
            if t in SIZES:
                f.seek(SIZES[t], 1)
            elif t == 8:
                f.seek(unpack("Q")[0], 1)
            elif t == 9:
                it, n = unpack("IQ")
                if it in SIZES:
                    f.seek(SIZES[it] * n, 1)
                else:
                    for _ in range(n):
                        skip(it)
            else:
                raise ValueError("tipo desconhecido")

        for _ in range(min(count, 100000)):
            key = text(65536)
            t, = unpack("I")
            # as listas do tokenizador sao enormes e vem depois das chaves da arquitetura
            if key.startswith("tokenizer.") and any(k.endswith(".context_length") for k in meta):
                break
            if t in SIZES:
                meta[key] = unpack(FMT[t])[0]
            elif t == 8 and key.startswith("general."):
                meta[key] = text(1 << 20)
            elif t == 9:
                pos = f.tell()
                it, n = unpack("IQ")
                if it in SIZES and n <= 4096:
                    meta[key] = list(unpack(f"{n}{FMT[it]}"))
                else:
                    f.seek(pos)
                    skip(t)
            else:
                skip(t)


def read_meta(path):
    """Chaves do cabecalho ({} se o arquivo nao for GGUF ou estiver corrompido). Lido uma vez por arquivo."""
    try:
        st = os.stat(path)
        key = (str(path), st.st_size, st.st_mtime_ns)
        if key not in _cache:
            meta = {}
            try:
                _read(path, meta)
            except (ValueError, struct.error, MemoryError):
                pass
            _cache[key] = meta
        return _cache[key]
    except OSError:
        return {}


def context_length(path):
    return int(next((v for k, v in read_meta(path).items() if k.endswith(".context_length") and isinstance(v, int)), 0))


def kv_bytes(meta, ctx, kv_type="f16", slots=4, ubatch=512):
    """Memoria do cache do contexto, em bytes (0 = nao da para estimar esta arquitetura)."""
    arch = meta.get("general.architecture", "")

    def g(key, default=0):
        return meta.get(f"{arch}.{key}", default)

    blocks = int(g("block_count") or 0) - int(g("nextn_predict_layers") or 0)
    heads = g("attention.head_count")
    if blocks <= 0 or not heads or g("attention.key_length_mla"):
        return 0

    def per_layer(v, i, default):
        if isinstance(v, list):
            return v[i] if i < len(v) else default
        return v if v else default

    embd = int(g("embedding_length") or 0)
    kv_heads = g("attention.head_count_kv", heads)
    window = int(g("attention.sliding_window") or 0)
    pattern = g("attention.sliding_window_pattern", None)
    interval = int(g("full_attention_interval") or 0)
    shared = int(g("attention.shared_kv_layers") or 0)
    swa_cells = min(ctx, -(-(window * slots + ubatch) // 256) * 256)
    total = 0
    for i in range(blocks - shared):
        if interval and (i + 1) % interval:
            continue  # camada recorrente dos modelos hibridos: estado pequeno e fixo
        h = per_layer(heads, i, 1)
        hk = kv_heads[i] if isinstance(kv_heads, list) and i < len(kv_heads) else per_layer(kv_heads, i, h)
        swa = False
        if window and isinstance(pattern, list):
            swa = bool(i < len(pattern) and pattern[i])
        elif window and isinstance(pattern, int) and pattern > 1:
            swa = bool((i + 1) % pattern)
        default = embd // h if h else 0
        key = int(g("attention.key_length_swa") if swa and g("attention.key_length_swa") else g("attention.key_length") or default)
        val = int(g("attention.value_length_swa") if swa and g("attention.value_length_swa") else g("attention.value_length") or default)
        total += hk * (key + val) * (swa_cells if swa else ctx)
    return int(total * KV_BYTES.get(kv_type, 2.0))


def needed(meta, model_size, extra, ctx, kv_type="f16"):
    """Memoria para o modelo rodar inteiro na placa (0 = nao da para estimar)."""
    kv = kv_bytes(meta, ctx, kv_type)
    return int(model_size + extra + kv + COMPUTE) if kv else 0


def best_context(meta, model_size, extra, vram, kv_type="f16", free=None):
    """Maior contexto em que o modelo cabe inteiro na placa (0 = nem com 4096; None = nao da para estimar).
    free: VRAM livre medida agora; sem ela, usa o total menos o que o Windows costuma ocupar."""
    if not vram or not kv_bytes(meta, 4096, kv_type):
        return None
    room = (free if free is not None else vram - DESKTOP) - MARGIN
    top = int(meta.get(f"{meta.get('general.architecture', '')}.context_length") or CTX_STEPS[-1])
    fits = [c for c in CTX_STEPS if c <= max(top, CTX_STEPS[0]) and needed(meta, model_size, extra, c, kv_type) <= room]
    return min(max(fits), top) if fits else 0


def size_of(path):
    try:
        return Path(path).stat().st_size
    except (OSError, TypeError):
        return 0
