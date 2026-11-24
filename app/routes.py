"""Rotas exatas da API (GET e POST): um metodo por rota e as tabelas GET_ROUTES e POST_ROUTES, usadas por Handler.do_GET e Handler.do_POST.

Metodos da classe Handler (server.py), que herda deste mixin. Os nomes do server aparecem como
S.<nome>: o modulo e consultado na hora da chamada.
"""
import os
import re
import secrets
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
import agent
import agents
import kb
import mcp
import modelsettings
import train
import gateway
import rea
import catalog
import media
import sprites
import strata

import server as S  # noqa: E402  (importado pelo proprio server; usado so dentro dos metodos)


class ApiRoutes:
    # ---- rotas exatas (tabelas GET_ROUTES e POST_ROUTES, logo abaixo da classe) ----
    def get_api_status(self, p, q):
        return self.send_json({"engine": S.ENGINE_PROC.status() | {"device": S.ENGINE_PROC.device}, "extras": S.POOL.status(),
                               "hub_tools": bool(mcp.hub_server()),
                               "memory": S.POOL.memory(), "backends": S.installed_backends(), "os": "windows" if os.name == "nt" else "linux", "quick": S.POOL.quick_status(),
                               "gpu_live": {k: S.DIAG[k] for k in ("vram_used", "vram_total", "gpu_util", "guard")} | {"full": bool(S.DIAG["full_since"])},
                               "gpu": S.GPU | {"cpu_count": S.CPU_COUNT},
                               "models": S.list_models(), "tasks": S.public_tasks(),
                               "settings": S.masked(S.load_settings()), "chats": S.list_chats(),
                               "stats": S.load_stats()})

    def get_api_train_status(self, p, q):
        act = train.active()
        return self.send_json({"env": train.env_status(), "datasets": train.ds_list(), "runs": train.runs_list(),
                               "active": act.id if act else "", "base_models": train.BASE_MODELS, "gpu": train.gpu_live()})

    def get_api_train_examples(self, p, q):
        return self.send_json(train.ds_examples(q.get("id", [""])[0], q.get("status", [""])[0], int(q.get("offset", ["0"])[0]),
                                                int(q.get("limit", ["50"])[0]), q.get("q", [""])[0]))

    def get_api_train_run(self, p, q):
        try:
            return self.send_json(train.run_get(q.get("id", [""])[0], int(q.get("since", ["0"])[0])))
        except ValueError as e:
            return self.send_json({"error": str(e)}, 404)

    def get_api_train_export_dataset(self, p, q):
        data = train.ds_export_jsonl(q.get("id", [""])[0]).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/jsonl; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="dataset.jsonl"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        return self.wfile.write(data)

    def get_api_integration(self, p, q):
        return self.send_json(gateway.GATEWAY.status())

    def get_api_diag(self, p, q):
        return self.send_json(S.diag_snapshot())

    def get_api_log(self, p, q):
        return self.send_json({"log": list(S.ENGINE_PROC.log)})

    def get_api_strata(self, p, q):
        return self.send_json(strata.status())

    def get_api_agents(self, p, q):
        return self.send_json(agents.load())

    def get_api_mcp(self, p, q):
        return self.send_json(self.mcp_status())

    def get_api_construct(self, p, q):
        mcp.MANAGER.sync()
        return self.send_json(sprites.construct_status())

    def get_api_sprites(self, p, q):
        return self.send_json(self.sprite_status())

    def get_mcp(self, p, q):
        return self.send_json({"error": "use POST (MCP streamable HTTP, respostas em JSON)"}, 405)

    def get_api_models_settings(self, p, q):
        rel = q.get("model", [""])[0]
        rec, family = S.model_recommended(rel)
        return self.send_json({"settings": modelsettings.get(rel), "recommended": rec, "family": family,
                               "fit_ctx": S.fit_context(rel), "bench": S.load_bench().get(rel)})

    def post_api_models_tune(self, p, b):
        rel = b.get("model", "")
        if rel not in {m["path"] for m in S.list_models()}:
            return self.send_json({"error": "Modelo nao encontrado."}, 404)
        if any(t["kind"] == "tune" and t["status"] == "running" for t in S.public_tasks()):
            return self.send_json({"error": "Já há um teste em andamento."}, 409)
        t = S.new_task("tune", "Otimizar " + S.model_display(rel))
        S.run_task(t, lambda task: S.tune_model(task, rel))
        return self.send_json({"id": t["id"]})

    def get_api_codex_diff(self, p, q):
        d = self.checkpoint_dir(q.get("chat_id", [""])[0], q.get("run_id", [""])[0])
        if not d:
            return self.send_json({"error": "ponto de restauração não encontrado"}, 404)
        return self.send_json(agent.diff_checkpoint(d))

    def get_api_kb(self, p, q):
        ep = media.embed_model_path()
        return self.send_json({"bases": kb.list_kbs(), "embed_model": ep.name if ep else ""})

    def get_api_kb_pending(self, p, q):
        return self.send_json(kb.load_meta(q.get("id", [""])[0]).get("pending_pdfs", []))

    def get_api_kb_file(self, p, q):
        meta = kb.load_meta(q.get("id", [""])[0])
        path = Path(q.get("path", [""])[0])
        if path.suffix.lower() != ".pdf" or not kb.inside_folders(meta, path) or not path.is_file():
            return self.send_json({"error": "arquivo inválido"}, 400)
        return self.send_file(path)

    def get_api_backup_info(self, p, q):
        return self.send_json(S.backup_info())

    def get_api_backup(self, p, q):
        return self.send_backup(q.get("attachments", ["1"])[0] == "1")

    def get_api_media(self, p, q):
        return self.send_json({"whisper": media.whisper_status(), "image": media.image_status(),
                               "mmproj": S.list_mmproj(), "mtp": S.list_mtp(), "rea": rea.status()})

    def get_api_chats(self, p, q):
        return self.send_json(S.list_chats())

    def get_api_hf_search(self, p, q):
        term = q.get("q", [""])[0]
        res = S.get_json("https://huggingface.co/api/models?" + urllib.parse.urlencode(
            {"search": term, "filter": "gguf", "sort": "downloads", "direction": "-1", "limit": "40"})
            + "&" + catalog.HF_EXPAND)
        return self.send_json([{"id": r["id"], "downloads": r.get("downloads", 0), "likes": r.get("likes", 0),
                                "updated": r.get("lastModified", ""), **catalog.model_caps(r)} for r in res])

    def get_api_hf_files(self, p, q):
        repo = q.get("repo", [""])[0]
        tree = S.hf_tree(repo)
        mm = S.pick_mmproj(tree)
        files = [f | catalog.quant_info(f["path"].rsplit("/", 1)[-1]) for f in S.hf_files(repo, tree)]
        mt = S.pick_mtp(tree)
        try:
            info = S.hf_model_info(repo)
        except (urllib.error.URLError, OSError, ValueError):
            info = None
        return self.send_json({"files": files, "mmproj": mm["path"] if mm else "", "info": info,
                               "mtp": mt["path"] if mt else "", "mtp_size": mt["size"] if mt else 0})

    def post_api_chat(self, p, b):
        return self.proxy_chat(b)

    def post_api_agent(self, p, b):
        return self.run_agent(b)

    def post_api_summarize(self, p, b):
        return self.summarize(b)

    def post_api_codex_undo(self, p, b):
        d = self.checkpoint_dir(b.get("chat_id", ""), b.get("run_id", ""))
        if not d:
            return self.send_json({"error": "ponto de restauração não encontrado"}, 404)
        try:
            return self.send_json({"restored": agent.undo_checkpoint(d)})
        except agent.ToolError as e:
            return self.send_json({"error": str(e)}, 400)

    def post_api_kb_create(self, p, b):
        return self.send_json(kb.create(b.get("name", ""), b.get("folders", [])))

    def post_api_kb_index(self, p, b):
        kid = b.get("id", "")
        meta = kb.load_meta(kid)
        t = S.new_task("kb", f"Indexando \"{meta['name']}\"")
        S.run_task(t, lambda task: kb.index(kid, task, lambda texts: S.embed_texts(texts), media.extract_text_path))
        return self.send_json({"id": t["id"]})

    def post_api_kb_add_text(self, p, b):
        try:
            return self.send_json({"chunks": kb.add_text(b.get("id", ""), b.get("path", ""), b.get("text", ""), S.embed_texts)})
        except RuntimeError as e:
            return self.send_json({"error": str(e)}, 400)

    def post_api_kb_search(self, p, b):
        try:
            return self.send_json(kb.search(b.get("id", ""), b.get("query", ""), S.embed_texts))
        except RuntimeError as e:
            return self.send_json({"error": str(e)}, 400)

    def post_api_group_run(self, p, b):
        return self.run_group(b)

    def post_api_mcp_test(self, p, b):
        srv = mcp.MANAGER.get(b.get("id", ""))
        if not srv:
            return self.send_json({"error": "servidor não encontrado"}, 404)
        srv.close()
        try:
            srv.connect()
        except mcp.MCPError as e:
            return self.send_json({"error": str(e), **self.mcp_status()}, 400)
        return self.send_json(self.mcp_status())

    def post_api_mcp_stop(self, p, b):
        srv = mcp.MANAGER.get(b.get("id", ""))
        if srv:
            srv.close()
        return self.send_json(self.mcp_status())

    def post_api_mcp_preset(self, p, b):
        try:
            cfg = mcp.preset(b.get("kind", ""))
        except mcp.MCPError as e:
            return self.send_json({"error": str(e)}, 400)
        servers = mcp.load()
        if any(s.get("id") == cfg["id"] or (s.get("command") == cfg["command"] and s.get("args") == cfg["args"]) for s in servers):
            return self.send_json({"error": "Esse servidor já foi adicionado."}, 400)
        mcp.save(servers + [cfg])
        return self.send_json(self.mcp_status())

    def post_api_agents_template(self, p, b):
        doc, gid = agents.template(b.get("kind", ""))
        return self.send_json({"doc": doc, "group_id": gid})

    def post_api_agent_approve(self, p, b):
        return self.send_json({"ok": agent.AgentRun.answer(b.get("id"), bool(b.get("ok")), bool(b.get("always")))})

    def post_api_image_generate(self, p, b):
        return self.image_generate(b)

    def post_api_sprites_generate(self, p, b):
        return self.sprites_generate(b)

    def post_api_sprites_pixelate(self, p, b):
        try:
            srcs = [media.upload_path(x) for x in (b.get("ids") or [])][:32]
            if not srcs:
                return self.send_json({"error": "Escolha uma imagem."}, 400)
            return self.send_json(sprites.process(srcs, sprites.clean_params(b), int(b.get("rows") or 1)))
        except ValueError as e:
            return self.send_json({"error": str(e)}, 400)

    def post_api_sprites_construct(self, p, b):
        try:
            return self.send_json(sprites.send_to_construct(
                b.get("sheet", ""), b.get("name", ""), int(b.get("size") or 32), int(b.get("cols") or 1),
                int(b.get("rows") or 1), b.get("views"), int(b.get("fps") or 8)))
        except (ValueError, mcp.MCPError) as e:
            return self.send_json({"error": str(e)}, 400)

    def post_api_construct_connect(self, p, b):
        mcp.MANAGER.sync()
        srv = mcp.MANAGER.get(mcp.CONSTRUCT_ID)
        if not srv:
            return self.send_json({"error": "Arasaka Construct não encontrado neste PC.", **sprites.construct_status()}, 404)
        srv.close()
        try:
            srv.connect()
        except mcp.MCPError as e:
            return self.send_json({"error": str(e), **sprites.construct_status()}, 400)
        return self.send_json(sprites.construct_status())

    def post_api_construct_project(self, p, b):
        try:
            mcp.set_construct_project(b.get("path", ""))
        except mcp.MCPError as e:
            return self.send_json({"error": str(e)}, 400)
        S.connect_construct_later(0)
        return self.send_json(sprites.construct_status())

    def post_api_transcribe(self, p, b):
        s = S.load_settings()
        try:
            text = media.transcribe(b.get("id", ""), s.get("whisper_model", ""), s.get("whisper_language", "auto"),
                                    S.cpu_threads(s))
        except RuntimeError as e:
            return self.send_json({"error": str(e)}, 400)
        return self.send_json({"text": text})

    def post_api_extract(self, p, b):
        try:
            return self.send_json({"text": media.extract_text(b.get("id", ""))})
        except (KeyError, zipfile.BadZipFile) as e:
            return self.send_json({"error": f"Nao foi possivel ler o documento: {e}"}, 400)

    def post_api_media_install(self, p, b):
        what, value = b.get("what"), b.get("value", "")
        jobs = {
            "whisper": (("cuda", "cpu"), "Motor de audio Whisper", media.install_whisper),
            "whisper_model": (tuple(media.WHISPER_MODELS), "Modelo Whisper", media.install_whisper_model),
            "sd": (("vulkan", "cuda", "cpu"), "Motor de imagem", media.install_sd),
            "image_preset": (tuple(media.IMAGE_PRESETS), "Modelo de imagem", media.install_image_preset),
            "embed_model": (("qwen3-embedding",), "Modelo de embeddings", media.install_embed_model),
            "rea": (("install",), "REA (engenharia reversa)", lambda task, _v: rea.install(task)),
        }
        if what not in jobs or value not in jobs[what][0]:
            return self.send_json({"error": "opcao invalida"}, 400)
        t = S.new_task("engine" if what in ("whisper", "sd", "rea") else "model",
                     jobs[what][1] if what == "rea" else f"{jobs[what][1]} ({value})")
        S.run_task(t, lambda task: jobs[what][2](task, value))
        return self.send_json({"id": t["id"]})

    def post_api_quick_load(self, p, b):
        upd = {k: b[k] for k in ("quick_model", "quick_device", "quick_keep_loaded") if k in b}
        if upd:
            S.save_settings(upd)
        S.POOL.quick_error = ""
        s = S.load_settings()
        S.POOL.stop(s.get("quick_model") or None) if "quick_device" in upd and s.get("quick_model") else None
        S.POOL.load_quick()
        return self.send_json(S.POOL.quick_status())

    def post_api_quick_unload(self, p, b):
        s = S.load_settings()
        S.save_settings({"quick_keep_loaded": False})
        S.POOL._quick_cancel = True
        if s.get("quick_model") and s["quick_model"] != S.ENGINE_PROC.model:
            S.POOL.stop(s["quick_model"])
        return self.send_json(S.POOL.quick_status())

    def post_api_integration_key(self, p, b):
        import secrets
        S.save_settings({"api_key": "pk-" + secrets.token_urlsafe(24)})
        return self.send_json(gateway.GATEWAY.status())

    def post_api_rea_uninstall(self, p, b):
        srv = mcp.MANAGER.get("rea")
        if srv:
            srv.close()
        rea.uninstall()
        return self.send_json(rea.status())

    def post_api_models_mtp(self, p, b):
        mt = b.get("mtp") or ""
        if not S.valid_accel(b.get("model", ""), mt):
            return self.send_json({"error": "speculative decoding inválido: use MTP, n-gram ou um draft model menor da mesma família"}, 400)
        S.set_mtp(b.get("model", ""), mt)
        return self.send_json({"ok": True})

    def post_api_models_mmproj(self, p, b):
        mm = b.get("mmproj") or ""
        if mm and mm not in S.list_mmproj():
            return self.send_json({"error": "arquivo de visao invalido"}, 400)
        S.set_mmproj(b.get("model", ""), mm)
        return self.send_json({"ok": True})

    def post_api_load(self, p, b):
        s = S.load_settings()
        s.update({k: v for k, v in b.items() if k in S.DEFAULT_SETTINGS})
        S.save_settings({"last_model": b.get("model", "")})
        if b.get("free_extras"):  # troca de modo: libera tambem os extras (menos o da pergunta rapida mantido)
            S.POOL.stop_except(S.POOL.kept_model())
        S.ENGINE_PROC.start(b.get("model", ""), s)
        return self.send_json({"ok": True})

    def post_api_unload(self, p, b):
        if b.get("extra"):
            S.POOL.stop(None if b["extra"] == "all" else b["extra"])
        else:
            S.ENGINE_PROC.stop()
            if b.get("all"):
                S.POOL.stop()
        return self.send_json({"ok": True})

    def post_api_engine_install(self, p, b):
        backend = b.get("backend")
        if backend not in S.BACKENDS:
            return self.send_json({"error": "backend invalido"}, 400)
        t = S.new_task("engine", f"Motor llama.cpp ({backend.upper()})")
        S.run_task(t, lambda task: S.install_engine(task, backend))
        return self.send_json({"id": t["id"]})

    def post_api_strata_install(self, p, b):
        if any(t["kind"] == "engine" and t["status"] == "running" for t in S.public_tasks()):
            return self.send_json({"error": "espere a instalação em andamento terminar"}, 409)
        t = S.new_task("engine", "Motor Strata")
        S.run_task(t, strata.install)
        return self.send_json({"id": t["id"]})

    def post_api_strata_launch(self, p, b):
        rel = strata.model_id(str(b.get("family", "")).strip().lower(), str(b.get("model", "")).strip().lower())
        if not strata.parse_id(rel):
            return self.send_json({"error": "modelo do Strata inválido"}, 400)
        S.ENGINE_PROC.start_strata(rel, str(b.get("context", "") or ""))
        return self.send_json({"ok": True, "model": rel})

    def post_api_models_download(self, p, b):
        repo, path = b.get("repo", ""), b.get("path", "")
        if not re.match(r"^[\w.-]+/[\w.-]+$", repo) or ".." in path:
            return self.send_json({"error": "repositorio invalido"}, 400)
        t = S.new_task("model", f"{repo} / {Path(path).name}")
        with_mm, with_mtp = bool(b.get("mmproj")), bool(b.get("mtp"))
        S.run_task(t, lambda task: S.download_model(task, repo, path, with_mm, with_mtp))
        return self.send_json({"id": t["id"]})

    def post_api_models_delete(self, p, b):
        f = (S.MODELS / b.get("path", "")).resolve()
        if S.MODELS.resolve() not in f.parents or not f.is_file():
            return self.send_json({"error": "modelo nao encontrado"}, 404)
        if S.ENGINE_PROC.model == b.get("path"):
            S.ENGINE_PROC.stop()
        m = S.SPLIT_RE.match(f.name)
        targets = list(f.parent.glob(f"{m.group(1)}-*-of-{m.group(3)}.gguf")) if m else [f]
        for t in targets:
            t.unlink()
        if f.parent != S.MODELS.resolve() and not any(f.parent.iterdir()):
            f.parent.rmdir()
        return self.send_json({"ok": True})

    def post_api_stats_reset(self, p, b):
        with S._stats_lock:
            S.STATS_FILE.unlink(missing_ok=True)
        return self.send_json(S.load_stats())

    def post_api_tasks_cancel(self, p, b):
        t = S.TASKS.get(b.get("id"))
        if t:
            t["cancel"] = True
        return self.send_json({"ok": True})

    def post_api_open_folder(self, p, b):
        folder = {"models": S.MODELS, "data": S.DATA, "engine": S.ENGINE}.get(b.get("which"), S.ROOT)
        if os.name == "nt":
            os.startfile(folder)  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(folder)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603,S607
        return self.send_json({"ok": True})


GET_ROUTES = {
    "/api/status": ApiRoutes.get_api_status,
    "/api/train/status": ApiRoutes.get_api_train_status,
    "/api/train/examples": ApiRoutes.get_api_train_examples,
    "/api/train/run": ApiRoutes.get_api_train_run,
    "/api/train/export_dataset": ApiRoutes.get_api_train_export_dataset,
    "/api/integration": ApiRoutes.get_api_integration,
    "/api/diag": ApiRoutes.get_api_diag,
    "/api/log": ApiRoutes.get_api_log,
    "/api/strata": ApiRoutes.get_api_strata,
    "/api/agents": ApiRoutes.get_api_agents,
    "/api/mcp": ApiRoutes.get_api_mcp,
    "/api/construct": ApiRoutes.get_api_construct,
    "/api/sprites": ApiRoutes.get_api_sprites,
    "/mcp": ApiRoutes.get_mcp,
    "/api/models/settings": ApiRoutes.get_api_models_settings,
    "/api/codex/diff": ApiRoutes.get_api_codex_diff,
    "/api/kb": ApiRoutes.get_api_kb,
    "/api/kb/pending": ApiRoutes.get_api_kb_pending,
    "/api/kb/file": ApiRoutes.get_api_kb_file,
    "/api/backup/info": ApiRoutes.get_api_backup_info,
    "/api/backup": ApiRoutes.get_api_backup,
    "/api/media": ApiRoutes.get_api_media,
    "/api/chats": ApiRoutes.get_api_chats,
    "/api/hf/search": ApiRoutes.get_api_hf_search,
    "/api/hf/files": ApiRoutes.get_api_hf_files,
}

POST_ROUTES = {
    "/api/chat": ApiRoutes.post_api_chat,
    "/api/agent": ApiRoutes.post_api_agent,
    "/api/summarize": ApiRoutes.post_api_summarize,
    "/api/codex/undo": ApiRoutes.post_api_codex_undo,
    "/api/kb/create": ApiRoutes.post_api_kb_create,
    "/api/kb/index": ApiRoutes.post_api_kb_index,
    "/api/kb/add_text": ApiRoutes.post_api_kb_add_text,
    "/api/kb/search": ApiRoutes.post_api_kb_search,
    "/api/group/run": ApiRoutes.post_api_group_run,
    "/api/mcp/test": ApiRoutes.post_api_mcp_test,
    "/api/mcp/stop": ApiRoutes.post_api_mcp_stop,
    "/api/mcp/preset": ApiRoutes.post_api_mcp_preset,
    "/api/agents/template": ApiRoutes.post_api_agents_template,
    "/api/agent/approve": ApiRoutes.post_api_agent_approve,
    "/api/image/generate": ApiRoutes.post_api_image_generate,
    "/api/sprites/generate": ApiRoutes.post_api_sprites_generate,
    "/api/sprites/pixelate": ApiRoutes.post_api_sprites_pixelate,
    "/api/sprites/construct": ApiRoutes.post_api_sprites_construct,
    "/api/construct/connect": ApiRoutes.post_api_construct_connect,
    "/api/construct/project": ApiRoutes.post_api_construct_project,
    "/api/transcribe": ApiRoutes.post_api_transcribe,
    "/api/extract": ApiRoutes.post_api_extract,
    "/api/media/install": ApiRoutes.post_api_media_install,
    "/api/quick/load": ApiRoutes.post_api_quick_load,
    "/api/quick/unload": ApiRoutes.post_api_quick_unload,
    "/api/integration/key": ApiRoutes.post_api_integration_key,
    "/api/rea/uninstall": ApiRoutes.post_api_rea_uninstall,
    "/api/models/mtp": ApiRoutes.post_api_models_mtp,
    "/api/models/mmproj": ApiRoutes.post_api_models_mmproj,
    "/api/load": ApiRoutes.post_api_load,
    "/api/models/tune": ApiRoutes.post_api_models_tune,
    "/api/unload": ApiRoutes.post_api_unload,
    "/api/engine/install": ApiRoutes.post_api_engine_install,
    "/api/strata/install": ApiRoutes.post_api_strata_install,
    "/api/strata/launch": ApiRoutes.post_api_strata_launch,
    "/api/models/download": ApiRoutes.post_api_models_download,
    "/api/models/delete": ApiRoutes.post_api_models_delete,
    "/api/stats/reset": ApiRoutes.post_api_stats_reset,
    "/api/tasks/cancel": ApiRoutes.post_api_tasks_cancel,
    "/api/open-folder": ApiRoutes.post_api_open_folder,
}
