"""Arquivos enviados, backup, geracao de imagens, setor Sprites, servidor MCP da Mikoshi e estudio de treino.

Metodos da classe Handler (server.py), que herda deste mixin. Os nomes do server aparecem como
S.<nome>: o modulo e consultado na hora da chamada.
"""
import json
import os
import random
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
import train
import media
import sprites

import server as S  # noqa: E402  (importado pelo proprio server; usado so dentro dos metodos)


class MediaRoutes:
    # ---- arquivos enviados ----
    def handle_upload(self):
        if not self.check_origin():
            return self.send_json({"error": "origem bloqueada"}, 403)
        n = int(self.headers.get("Content-Length") or 0)
        name = urllib.parse.unquote(self.headers.get("X-Filename") or "arquivo.bin")

        def chunks():
            left = n
            while left > 0:
                data = self.rfile.read(min(1 << 20, left))
                if not data:
                    break
                left -= len(data)
                yield data
        try:
            return self.send_json(media.save_upload(name, chunks()))
        except ValueError as e:
            return self.send_json({"error": str(e)}, 400)

    # ---- backup ----
    def send_backup(self, with_files):
        tmp = S.DATA / f"_backup_{uuid.uuid4().hex[:8]}.zip"
        try:
            S.build_backup(tmp, with_files)
            name = f"LLM-Local-backup-{datetime.now():%Y-%m-%d_%H%M}.zip"
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(tmp.stat().st_size))
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.end_headers()
            with open(tmp, "rb") as f:
                shutil.copyfileobj(f, self.wfile, 1 << 20)
        except OSError:
            pass
        finally:
            tmp.unlink(missing_ok=True)

    def handle_restore(self, q):
        if not self.check_origin():
            return self.send_json({"error": "origem bloqueada"}, 403)
        n = int(self.headers.get("Content-Length") or 0)
        tmp = S.DATA / f"_restore_{uuid.uuid4().hex[:8]}.zip"
        try:
            with open(tmp, "wb") as f:
                left = n
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            return self.send_json(S.restore_backup(tmp, q.get("settings", ["0"])[0] == "1"))
        except (ValueError, zipfile.BadZipFile) as e:
            return self.send_json({"error": str(e) if isinstance(e, ValueError) else "Arquivo de backup invalido ou corrompido."}, 400)
        finally:
            tmp.unlink(missing_ok=True)

    # ---- geracao de imagem ----
    def enhance_prompt(self, prompt):
        """Usa o LLM carregado para transformar o pedido num prompt detalhado em ingles."""
        req = ("Rewrite the user's image request as ONE detailed English prompt for an image generation model: "
               "subject, style, composition, lighting, colors, details. Keep any text that must appear in the image "
               "in its original language, inside quotes. Answer with the prompt only.\n\nRequest: " + prompt)
        try:
            with self.engine_call({"messages": [{"role": "user", "content": req}], "stream": False, "temperature": 0.6,
                                   "max_tokens": 400, "chat_template_kwargs": {"enable_thinking": False}}, False) as r:
                data = json.loads(r.read().decode("utf-8"))
            S.add_stats(data.get("timings"))
            out = data["choices"][0]["message"].get("content") or ""
            out = re.sub(r"<think>[\s\S]*?(</think>|$)", "", out).strip().strip('"')
            return out if len(out) > 8 else prompt
        except (RuntimeError, ValueError, KeyError, IndexError, OSError):
            return prompt

    def image_generate(self, b):
        s = S.load_settings()
        preset = b.get("preset") or s.get("image_preset") or "z-image-turbo"
        prompt = (b.get("prompt") or "").strip()
        if not prompt:
            return self.send_json({"error": "Descreva a imagem."}, 400)
        self.start_sse()
        alive = {"ok": True}

        def emit(obj):
            if alive["ok"]:
                alive["ok"] = self.sse_safe(obj)
            return alive["ok"]

        reload_model = None
        try:
            final = prompt
            if b.get("enhance", s.get("image_enhance")) and S.ENGINE_PROC.state == "ready":
                emit({"image": {"stage": "enhance"}})
                final = self.enhance_prompt(prompt)
            if s.get("image_unload_llm") and S.ENGINE_PROC.state in ("ready", "loading"):
                reload_model = S.ENGINE_PROC.model
                emit({"image": {"stage": "unload"}})
                S.ENGINE_PROC.stop()
            seed = int(b.get("seed", -1))
            if seed < 0:
                seed = random.randint(0, 2 ** 31 - 1)
            emit({"image": {"stage": "loading", "prompt": final, "seed": seed}})
            t0 = time.time()
            fid, out = media.generate_image(
                preset, final, b.get("negative", ""), b.get("width") or s.get("image_width"),
                b.get("height") or s.get("image_height"), b.get("steps") or s.get("image_steps"), 0, seed,
                s.get("image_offload", True), threads=S.cpu_threads(s),
                on_progress=lambda pr: emit({"image": {"stage": "progress", **pr}}),
                cancel=lambda: not alive["ok"])
            emit({"image": {"stage": "done", "id": fid, "path": str(out), "prompt": prompt, "final_prompt": final,
                            "seed": seed, "preset": preset, "seconds": round(time.time() - t0, 1)}})
        except RuntimeError as e:
            emit({"error": {"message": str(e)}})
        finally:
            if reload_model:
                try:
                    S.ENGINE_PROC.start(reload_model, S.load_settings())
                except (ValueError, OSError):
                    pass

    def image_for_agent(self, prompt, width, height, dest):
        s = S.load_settings()
        fid, out = media.generate_image(s.get("image_preset") or "z-image-turbo", prompt, "", width or 0, height or 0,
                                        0, 0, random.randint(0, 2 ** 31 - 1), s.get("image_offload", True))
        shutil.copy2(out, dest)
        return fid

    # ---- Sprites (pixel art) ----
    def sprite_status(self):
        img = media.image_status()
        key = media.pixel_default()
        models = [{"key": k, **img["presets"][k], "bytes": media.IMAGE_PRESETS[k].get("bytes", 0),
                   "dir": str(S.MODELS / "image" / k)} for k in media.PIXEL_PRESETS]
        return {"engine": {k: img[k] for k in ("installed", "version", "problem")}, "preset": key, "models": models,
                "model": img["presets"][key], "download_bytes": media.IMAGE_PRESETS[key].get("bytes", 0),
                "construct": sprites.construct_status(), "mcp_url": f"http://{S.HOST}:{S.PORT}/mcp",
                "sizes": list(sprites.pixel.SIZES), "views": list(sprites.VIEWS), "os": "windows" if os.name == "nt" else "linux",
                "models_dir": str(S.MODELS / "image" / key),
                "tasks": [t for t in S.public_tasks() if any(k in t["label"] for k in media.PIXEL_PRESETS)
                          or t["label"].startswith("Motor de imagem")]}

    def sprite_gen(self, alive=None, key=None):
        """Funcao gen() do sprites.py: gera uma imagem com um preset de pixel art e devolve o caminho."""
        s = S.load_settings()
        key = key if key in media.PIXEL_PRESETS else media.pixel_default()
        if not media.find_exe("sd", media.SD_EXE):
            raise RuntimeError("Instale o motor de imagem primeiro (setor Sprites > Instalar motor).")
        prob = media.engine_problem("sd", media.SD_EXE)
        if prob:
            raise RuntimeError(prob)
        if not media.preset_installed(key):
            raise RuntimeError(f"Baixe o modelo {media.IMAGE_PRESETS[key]['name']} primeiro (setor Sprites > Baixar modelo).")
        preset = media.IMAGE_PRESETS[key]

        def gen(prompt, negative, seed, mode, on_progress=None):
            m = preset["modes"].get(mode) or preset["modes"]["rapido"]
            _, out = media.generate_image(key, prompt, negative, 0, 0, m["steps"], m["cfg"], seed,
                                          s.get("image_offload", True), threads=S.cpu_threads(s),
                                          on_progress=on_progress, cancel=(lambda: not alive["ok"]) if alive else None,
                                          loras=m["loras"], sampler=m["sampler"])
            return out
        return gen

    def sprites_generate(self, b):
        try:
            params = sprites.clean_params(b)
        except ValueError as e:
            return self.send_json({"error": str(e)}, 400)
        if not (b.get("prompt") or "").strip():
            return self.send_json({"error": "Descreva o sprite."}, 400)
        self.start_sse()
        alive = {"ok": True}

        def emit(obj):
            if alive["ok"]:
                alive["ok"] = self.sse_safe(obj)
            return alive["ok"]

        s = S.load_settings()
        reload_model = None
        try:
            self.sprite_gen(alive, params["preset"])  # confere motor e modelo antes de descarregar o LLM
            if s.get("image_unload_llm") and S.ENGINE_PROC.state in ("ready", "loading"):
                reload_model = S.ENGINE_PROC.model
                emit({"sprite": {"stage": "unload"}})
                S.ENGINE_PROC.stop()
            with S.SPRITE_LOCK:
                res = sprites.generate(b["prompt"], params, lambda k: self.sprite_gen(alive, k), lambda ev: emit({"sprite": ev}),
                                       lambda: not alive["ok"], int(b.get("seed", -1)))
            emit({"sprite": {"stage": "done", **res}})
        except (RuntimeError, ValueError, OSError) as e:
            emit({"error": {"message": str(e)}})
        finally:
            if reload_model:
                try:
                    S.ENGINE_PROC.start(reload_model, S.load_settings())
                except (ValueError, OSError):
                    pass

    def mikoshi_mcp(self):
        """Servidor MCP da Mikoshi (HTTP streamable, respostas em JSON): generate_sprite, pixelate_image, sprite_status."""
        try:  # o Host ja foi conferido em do_POST (DNS rebinding)
            msg = self.body()
        except ValueError:
            return self.send_json({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "JSON inválido"}}, 400)

        def gen_for(key):
            gen = self.sprite_gen(None, key)

            def locked(*a, **k):
                with S.SPRITE_LOCK:
                    return gen(*a, **k)
            return locked
        status = self.sprite_status
        if isinstance(msg, list):
            out = [r for r in (sprites.handle_rpc(m, gen_for, status) for m in msg) if r]
        else:
            out = sprites.handle_rpc(msg, gen_for, status)
        if not out:
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        return self.send_json(out)

    def train_post(self, p, b):
        try:
            if p == "/api/train/install":
                t = S.new_task("engine", "Ambiente de treino (Python + PyTorch CUDA)")
                S.run_task(t, train.install)
                return self.send_json({"id": t["id"]})
            if p == "/api/train/uninstall":
                train.uninstall()
                return self.send_json(train.env_status())
            if p == "/api/train/dataset/create":
                return self.send_json(train.ds_create(b.get("name", ""), b.get("system", "")))
            if p == "/api/train/dataset/update":
                return self.send_json(train.ds_update(b.get("id", ""), b.get("name"), b.get("system")))
            if p == "/api/train/dataset/delete":
                train.ds_delete(b.get("id", ""))
                return self.send_json({"ok": True})
            if p == "/api/train/dataset/example":
                return self.send_json(train.ds_save_example(b.get("id", ""), b.get("example") or {}))
            if p == "/api/train/dataset/review":
                return self.send_json(train.ds_review(b.get("id", ""), b.get("ids") or [], b.get("status", "")))
            if p == "/api/train/dataset/import_chats":
                ids = b.get("chat_ids") or [c["id"] for c in S.list_chats()]
                chats = []
                for cid in ids:
                    f = S.CHATS / f"{cid}.json"
                    if S.ID_RE.match(cid) and f.exists():
                        chats.append(json.loads(f.read_text("utf-8")))
                return self.send_json({"added": train.ds_import_chats(b.get("id", ""), chats)})
            if p == "/api/train/dataset/import_jsonl":
                return self.send_json({"added": train.ds_import_jsonl(b.get("id", ""), b.get("text", ""))})
            if p == "/api/train/dataset/generate":
                folders = [f for f in b.get("folders") or [] if Path(f).is_dir()]
                if not folders:
                    return self.send_json({"error": "Escolha pelo menos uma pasta."}, 400)
                did = b.get("id", "")
                t = S.new_task("model", f"Gerando exemplos de treino ({train.ds_meta(did)['name']})")
                S.run_task(t, lambda task: train.ds_generate(task, did, folders, int(b.get("per_chunk") or 3), int(b.get("max_chunks") or 40),
                                                           b.get("topic") or "Unreal Engine 5 e C++", b.get("extra", "")))
                return self.send_json({"id": t["id"]})
            if p == "/api/train/start":
                b.setdefault("hf_token", S.load_settings().get("hf_token", ""))
                return self.send_json(train.run_start(b))
            if p == "/api/train/stop":
                return self.send_json(train.run_stop(b.get("id", ""), bool(b.get("force"))))
            if p == "/api/train/export":
                return self.send_json(train.run_export(b.get("id", ""), b.get("name", ""), b.get("quant", "Q4_K_M")))
            if p == "/api/train/run/delete":
                train.run_delete(b.get("id", ""))
                return self.send_json({"ok": True})
            return self.send_json({"error": "rota desconhecida"}, 404)
        except (ValueError, RuntimeError, OSError, KeyError) as e:
            return self.send_json({"error": str(e)}, 400)
