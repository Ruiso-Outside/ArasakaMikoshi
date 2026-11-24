"""Otimizar um modelo para este PC: mede com o llama-bench (vem com o motor) e guarda o ajuste mais rapido do modelo.

Testa o tamanho do lote (`-ub`, acelera a leitura do prompt e usa um pouco mais de VRAM) e o cache do contexto em 8 bits
(so muda algo na velocidade quando libera memoria para por mais do modelo na placa). O teste encaixa o modelo na placa
como o motor faz ao carregar (`--fit-target`), com o contexto configurado.
"""
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

N_PROMPT, N_GEN, REPS = 1536, 64, 2
TIMEOUT = 420  # segundos por configuracao
UBATCHES = (512, 1024, 2048)
BUSY_SHARE = 0.25  # acima disso, outro programa (um jogo) esta usando a placa e a medida nao vale


def bench_exe(engine_exe):
    """O llama-bench ao lado do llama-server (None se o pacote nao o trouxer)."""
    p = Path(engine_exe)
    f = p.with_name("llama-bench" + p.suffix)
    return f if f.is_file() else None


def command(exe, model, ctx, ubatch=512, kv="f16", threads=0, gpu=True):
    cmd = [str(exe), "-m", str(model), "-p", str(N_PROMPT), "-n", str(N_GEN), "-r", str(REPS), "-o", "jsonl",
           "-ub", str(ubatch), "-b", str(max(2048, ubatch)), "-ctk", kv, "-ctv", kv]
    if gpu:
        cmd += ["-fitt", "1024", "-fitc", str(ctx)]
    else:
        cmd += ["-dev", "none"]
    if threads:
        cmd += ["-t", str(threads)]
    return cmd


def parse(text):
    """Tokens por segundo lendo o prompt (pp) e escrevendo (tg), da saida jsonl do llama-bench."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            j = json.loads(line)
            if j.get("n_prompt") and not j.get("n_gen"):
                out["pp"] = round(float(j["avg_ts"]), 1)
            elif j.get("n_gen") and not j.get("n_prompt"):
                out["tg"] = round(float(j["avg_ts"]), 1)
        except (ValueError, KeyError, TypeError):
            continue
    return out if "pp" in out and "tg" in out else None


def choose(rows):
    """Melhor ajuste entre as medidas. A primeira linha e o padrao. So troca com ganho claro, sem perder velocidade de escrita."""
    base = rows[0]
    best = base
    for r in rows[1:]:
        if r["kv"] == "f16" and r["tg"] >= base["tg"] * 0.97 and r["pp"] >= base["pp"] * 1.05 and r["pp"] > best["pp"]:
            best = r
    for r in rows[1:]:
        # o cache em 8 bits pode perder um pouco de qualidade: so vale se a escrita ficar bem mais rapida
        if r["kv"] != "f16" and r["tg"] >= best["tg"] * 1.10:
            best = r
    values = {}
    if best["ubatch"] != 512:
        values["ubatch"] = best["ubatch"]
    if best["kv"] != "f16":
        values["kv_cache"] = best["kv"]
    return best, values


def run_one(cmd, should_stop, on_start=None):
    """Roda o llama-bench. Devolve (medida ou None, codigo de saida, fim do log). should_stop() pode lancar excecao."""
    env = None
    if os.name != "nt":
        env = {**os.environ, "LD_LIBRARY_PATH": os.pathsep.join(filter(None, [str(Path(cmd[0]).parent), os.environ.get("LD_LIBRARY_PATH", "")]))}
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        p = subprocess.Popen(cmd, stdout=out, stderr=err, stdin=subprocess.DEVNULL, cwd=str(Path(cmd[0]).parent), env=env,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            if on_start:
                on_start(p)
            end = time.monotonic() + TIMEOUT
            while p.poll() is None:
                should_stop()
                if time.monotonic() > end:
                    return None, -1, "tempo esgotado"
                time.sleep(0.2)
        finally:
            if p.poll() is None:
                p.kill()
                p.wait(10)
        out.seek(0)
        err.seek(0)
        return (parse(out.read().decode("utf-8", "replace")) if p.returncode == 0 else None, p.returncode,
                err.read().decode("utf-8", "replace")[-600:])
