"""Chat, Codex, agentes e grupos: streaming (SSE), pesquisa na web, base de conhecimento, ferramentas (MCP e modulos da Central) e resumo.

Metodos da classe Handler (server.py), que herda deste mixin. Os nomes do server aparecem como
S.<nome>: o modulo e consultado na hora da chamada.
"""
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
import agent
import agents
import ctxfit
import kb
import mcp
import media
import promptcache
import websearch

import server as S  # noqa: E402  (importado pelo proprio server; usado so dentro dos metodos)


class ChatRoutes:
    # ---- chat (stream SSE) ----
    def sse(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n")
        self.wfile.flush()

    def engine_call(self, payload, stream):
        port = getattr(self, "engine_port", S.ENGINE_PORT)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=json.dumps(payload).encode("utf-8"), headers=S.ENGINE_HEADERS)
        try:
            return urllib.request.urlopen(req, timeout=3600)
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")
            try:
                msg = json.loads(msg)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                pass
            raise RuntimeError(msg) from None
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"Motor nao respondeu: {e}") from None

    def make_query(self, messages):
        """Pede ao proprio modelo uma consulta curta para o buscador."""
        convo = [m for m in messages if m["role"] in ("user", "assistant")][-5:]
        hist = "\n".join(f"{'Usuario' if m['role'] == 'user' else 'Assistente'}: {S.text_of(m['content'])[:600]}"
                         for m in convo[:-1])
        last = S.text_of(convo[-1]["content"]) if convo else ""
        prompt = (f"Data de hoje: {datetime.now():%d/%m/%Y}.\n"
                  + (f"Conversa anterior:\n{hist}\n\n" if hist else "")
                  + f"Mensagem atual do usuario:\n{last[:1500]}\n\n"
                  "Escreva UMA consulta curta para um buscador da web que encontre as informacoes necessarias "
                  "para responder a mensagem atual. Responda somente com a consulta, sem aspas e sem explicacao.")
        try:
            with self.engine_call({"messages": [{"role": "user", "content": prompt}], "stream": False,
                                   "temperature": 0.2, "max_tokens": 600,
                                   "chat_template_kwargs": {"enable_thinking": False}}, False) as r:
                data = json.loads(r.read().decode("utf-8"))
            S.add_stats(data.get("timings"))
            q = data["choices"][0]["message"].get("content") or ""
            q = re.sub(r"<think>[\s\S]*?(</think>|$)", "", q).strip().splitlines()
            q = q[0].strip().strip('"\'`').strip() if q else ""
            if q.lower().startswith(("consulta:", "query:")):
                q = q.split(":", 1)[1].strip()
            if 2 <= len(q) <= 200:
                return q
        except (RuntimeError, ValueError, KeyError, IndexError, OSError):
            pass
        return re.sub(r"\s+", " ", last)[:200]

    def web_augment(self, messages):
        s = S.load_settings()
        self.sse({"llmlocal": {"stage": "query"}})
        query = self.make_query(messages)
        self.sse({"llmlocal": {"stage": "search", "query": query}})
        try:
            results = websearch.search(query, s)
        except Exception as e:  # noqa: BLE001
            self.sse({"llmlocal": {"stage": "error", "query": query, "text": f"Falha na pesquisa: {e}"}})
            return messages
        S.add_stats(searches=1)
        if not results:
            self.sse({"llmlocal": {"stage": "error", "query": query, "text": "A pesquisa nao retornou resultados."}})
            return messages
        self.sse({"llmlocal": {"stage": "read", "query": query,
                               "sources": [{"title": r["title"], "url": r["url"]} for r in results]}})
        n_ctx = S.ENGINE_PROC.n_ctx or int(s["ctx_size"]) or 8192
        budget = max(2500, min(24000, int(n_ctx * 0.45 * 3.2)))  # ~3,2 caracteres por token
        texts = websearch.fetch_pages(results, budget // len(results))
        blocks = []
        for i, (r, t) in enumerate(zip(results, texts), 1):
            blocks.append(f"[{i}] {r['title']}\nURL: {r['url']}\n{t or r['snippet']}")
        self.sse({"llmlocal": {"stage": "done", "query": query,
                               "sources": [{"title": r["title"], "url": r["url"], "snippet": r["snippet"]} for r in results]}})
        ctx = (f"[Resultados de pesquisa na web, feita agora ({datetime.now():%d/%m/%Y %H:%M}) com a consulta \"{query}\"]\n\n"
               + "\n\n---\n\n".join(blocks)
               + "\n\n[Fim dos resultados]\n\nUse os resultados acima para responder com informacoes atuais. "
               "Cite as fontes no texto com o numero entre colchetes, ex.: [1], [2]. "
               "Se os resultados nao forem suficientes, diga isso.\n\nMensagem do usuario:\n")
        out = [dict(m) for m in messages]
        for m in reversed(out):
            if m["role"] == "user":
                if isinstance(m["content"], list):
                    m["content"] = [dict(x) for x in m["content"]]
                    for part in m["content"]:
                        if part.get("type") == "text":
                            part["text"] = ctx + part["text"]
                            break
                else:
                    m["content"] = ctx + m["content"]
                break
        return out

    def sse_safe(self, obj):
        """Envia um evento SSE; obj=None manda so um ping. Retorna False se o cliente desconectou."""
        try:
            if obj is None:
                self.wfile.write(b": ping\n\n")
                self.wfile.flush()
            else:
                self.sse(obj)
            return True
        except OSError:
            return False

    def start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    # ---- grupos de agentes ----
    def reply_rule(self):
        return S.reply_rule()

    def apply_model_sampling(self, payload, rel, agent_temperature=None):
        return S.apply_model_sampling(payload, rel, agent_temperature)

    def split_content(self, m, modalities=None):
        """Separa o texto e as partes de midia (imagens/audio) de uma mensagem do usuario."""
        c = S.build_content(m, S.ENGINE_PROC.modalities if modalities is None else modalities)
        if isinstance(c, list):
            return S.text_of(c), [x for x in c if x.get("type") != "text"]
        return c, []

    def run_group(self, b):
        self.start_sse()
        sampling = {k: b[k] for k in ("temperature", "top_p", "top_k", "min_p", "repeat_penalty") if k in b}
        if int(b.get("max_tokens") or 0) > 0:
            sampling["max_tokens"] = int(b["max_tokens"])
        msgs = [m for m in b.get("messages", []) if m.get("role") != "system"]
        if not msgs or msgs[-1].get("role") != "user":
            return self.sse_safe({"error": {"message": "Mensagem vazia."}})
        history = S.build_history(msgs[:-1], S.ENGINE_PROC.modalities)
        try:
            timings = agents.run_group(self, b.get("group_id", ""), msgs[-1], history, sampling)
            if timings:
                S.add_stats(timings)
        except RuntimeError as e:
            self.sse_safe({"error": {"message": str(e)}})
        except ConnectionError:
            pass

    # ---- Codex ----
    def run_agent(self, b):
        if S.ENGINE_PROC.state != "ready":
            return self.send_json({"error": "Nenhum modelo carregado."}, 503)
        s = S.load_settings()
        mcp_ids = [x.cfg["id"] for x in mcp.MANAGER.all() if x.cfg.get("enabled")]
        if b.get("mcp_only"):
            if not mcp_ids:
                return self.send_json({"error": "Nenhum servidor MCP ativo. Adicione um em Configurações > Ferramentas MCP."}, 400)
            ws = None
        else:
            ws = str(Path(b.get("workspace") or "_").resolve())
            if ws not in s.get("workspaces", []):
                return self.send_json({"error": "Pasta nao liberada. Adicione-a nas Configuracoes > Codex."}, 403)
        self.start_sse()
        sampling = {k: b[k] for k in ("temperature", "top_p", "top_k", "min_p", "repeat_penalty") if k in b}
        if int(b.get("max_tokens") or 0) > 0:
            sampling["max_tokens"] = int(b["max_tokens"])
        history = S.build_history(b.get("messages", []), S.ENGINE_PROC.modalities, agent_mode=True,
                                  budget=promptcache.budget(S.ENGINE_PROC.n_ctx))
        s["_n_ctx"] = S.ENGINE_PROC.n_ctx
        image_fn = self.image_for_agent if (media.find_exe("sd", media.SD_EXE)
                                            and media.preset_installed(s.get("image_preset") or "z-image-turbo")) else None
        s["_reply_rule"] = S.reply_rule(s)
        S.apply_model_sampling(sampling, S.ENGINE_PROC.model)
        chat_id = b.get("chat_id", "") if S.ID_RE.match(b.get("chat_id", "")) else "sem-conversa"
        ckpt = S.CHECKPOINTS / chat_id / uuid.uuid4().hex[:10]
        kb_fn, kb_name = self.codex_kb(b.get("kb", ""))
        run = agent.AgentRun(lambda payload: self.engine_call(payload, True), self.sse_safe, s, image_fn if ws else None,
                             ckpt, mcp_ids, self.save_tool_image, kb_fn, kb_name)
        try:
            trace, timings = run.run(ws, history, b.get("chat_id", ""), sampling)
            run.send({"agent": {"type": "trace", "trace": trace}})
            if timings:
                S.add_stats(timings)
        except (RuntimeError, agent.ToolError) as e:
            run.send({"error": {"message": str(e)}})

    def codex_kb(self, kid):
        """Funcao de busca na base de conhecimento para o Codex (ferramenta search_knowledge)."""
        if not kid:
            return None, ""
        try:
            meta = kb.load_meta(kid)
        except (ValueError, OSError):
            return None, ""

        def search(query):
            if not query.strip():
                return "Erro: informe o que procurar."
            try:
                hits = kb.search(kid, query[:2000], S.embed_texts, k=6)
            except RuntimeError as e:
                return f"Erro ao buscar na base: {e}"
            if not hits:
                return "Nada relevante encontrado na base para essa consulta. Tente outras palavras."
            return "\n\n---\n\n".join(f"[D{i}] {h['path']} (linhas {h['lines']}, relevancia {h['score']})\n{h['text']}"
                                         for i, h in enumerate(hits, 1))
        return search, meta.get("name", "")

    def save_tool_image(self, data, mime):
        """Guarda uma imagem devolvida por uma ferramenta MCP (ex.: print da tela) e devolve o id do arquivo."""
        ext = {"image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}.get(mime, "png")
        fid, path = media.new_media_path(ext)
        path.write_bytes(data)
        return fid

    def mcp_status(self):
        mcp.MANAGER.sync()  # o Construct pode ter sido instalado depois de a Mikoshi abrir
        return {"servers": [x.public() for x in mcp.MANAGER.all()], "roblox_detected": bool(mcp.roblox_bat())}

    def checkpoint_dir(self, chat_id, run_id):
        if not re.match(r"^[A-Za-z0-9_-]{1,64}$", chat_id or "") or not re.match(r"^[a-f0-9]{10}$", run_id or ""):
            return None
        d = S.CHECKPOINTS / chat_id / run_id
        return d if (d / "manifest.json").is_file() else None

    def summarize(self, b):
        """Resume a parte antiga de uma conversa para ela caber no contexto."""
        msgs = [m for m in b.get("messages", []) if m.get("role") in ("user", "assistant")]
        if not msgs:
            return self.send_json({"error": "nada para resumir"}, 400)
        try:
            with self.engine_for(b.get("model", "")) as eng:
                self.engine_port = eng.port
                budget = int((eng.n_ctx or 8192) * 0.55 * 3)  # ~3 caracteres por token, deixando espaço para o resumo
                per = max(300, budget // len(msgs))
                lines = []
                for m in msgs:
                    t = S.text_of(m.get("content", ""))
                    if len(t) > per:
                        t = t[:per * 2 // 3] + "\n[...]\n" + t[-per // 3:]
                    lines.append(("Usuário" if m["role"] == "user" else "Assistente") + ": " + t)
                prompt = ("Resuma a conversa a seguir para que ela possa continuar sem o histórico completo. Preserve: "
                          "o objetivo do usuário, fatos, decisões, nomes, números, trechos de código importantes e o que "
                          "ainda está pendente. Seja conciso e use tópicos. Escreva no idioma da conversa.\n\n" + "\n\n".join(lines))
                with self.engine_call({"messages": [{"role": "user", "content": prompt}], "stream": False, "temperature": 0.3,
                                       "max_tokens": 1200, "chat_template_kwargs": {"enable_thinking": False}}, False) as r:
                    data = json.loads(r.read().decode("utf-8"))
            S.add_stats(data.get("timings"))
            text = re.sub(r"<think>[\s\S]*?(</think>|$)", "", data["choices"][0]["message"].get("content") or "").strip()
            return self.send_json({"summary": text})
        except (RuntimeError, KeyError, IndexError, ValueError) as e:
            return self.send_json({"error": str(e)}, 400)

    def kb_augment(self, messages, kid):
        """Busca na base de conhecimento os trechos mais parecidos com a pergunta e os junta à mensagem."""
        try:
            meta = kb.load_meta(kid)
        except (ValueError, OSError):
            return messages
        users = [S.text_of(m["content"]) for m in messages if m.get("role") == "user"]
        if not users:
            return messages
        query = users[-1] if len(users[-1]) > 40 or len(users) < 2 else users[-2] + "\n" + users[-1]
        self.sse({"kb": {"stage": "searching", "name": meta["name"]}})
        try:
            hits = kb.search(kid, query[-2000:], lambda texts: S.embed_texts(texts, self.loading_notifier(
                lambda n: {"kb": {"stage": "loading", "name": meta["name"], "model": n}})))
        except RuntimeError as e:
            self.sse({"kb": {"stage": "error", "name": meta["name"], "text": str(e)}})
            return messages
        sources = [{"n": i, "path": h["path"], "file": Path(h["path"]).name, "lines": h["lines"], "score": h["score"]}
                   for i, h in enumerate(hits, 1)]
        self.sse({"kb": {"stage": "done", "name": meta["name"], "sources": sources}})
        if not hits:
            return messages
        blocks = [f"[D{i}] {Path(h['path']).name} (linhas {h['lines']})\n{h['text']}" for i, h in enumerate(hits, 1)]
        ctx = (f"[Trechos da base RAG \"{meta['name']}\" relacionados à pergunta]\n\n" + "\n\n---\n\n".join(blocks)
               + "\n\n[Fim dos trechos]\n\nUse os trechos acima quando forem relevantes e cite as fontes no texto como "
               "[D1], [D2]. Se eles não bastarem para responder, diga isso.\n\nMensagem do usuario:\n")
        out = [dict(m) for m in messages]
        for m in reversed(out):
            if m["role"] == "user":
                if isinstance(m["content"], list):
                    m["content"] = [dict(x) for x in m["content"]]
                    for part in m["content"]:
                        if part.get("type") == "text":
                            part["text"] = ctx + part["text"]
                            break
                else:
                    m["content"] = ctx + m["content"]
                break
        return out

    def engine_for(self, model_rel, on_wait=None, device="auto"):
        """Motor para esta requisicao: o principal ou um modelo extra (carregado sob demanda, na GPU ou CPU)."""
        return S.POOL.use(model_rel or "", S.load_settings(), on_wait, device)

    def loading_notifier(self, make_event):
        """on_wait do POOL: avisa a interface uma vez e depois so manda pings (detecta desconexao)."""
        return lambda name, first: self.sse_safe(make_event(name) if first else None)

    # ---- ferramentas dos modulos do Piphin Hub no chat (Piphin Macro, InfoHardware...)
    def hub_tools(self, payload):
        """Liga as ferramentas da Central neste chat: (ferramentas, mapa) ou None se nao houver."""
        if not mcp.hub_server():
            return None
        tools, mapping = mcp.MANAGER.toolset([mcp.HUB_ID], None,
                                             lambda name, err: self.sse({"tool": {"stage": "error", "text": err}}))
        if not tools:
            return None
        mods = sorted({(t.get("title") or "").split(":")[0] for _, t in mapping.values() if t.get("title")})
        note = ("Você está dentro da Arasaka Central e pode usar ferramentas de outros módulos instalados (" + ", ".join(mods) +
                "). Use-as só quando o pedido envolver esses programas (ex.: criar ou ajustar macros, analisar o hardware do PC). "
                "Para analisar o PC, leia os dados reais com a ferramenta antes de responder. Depois de usar uma ferramenta, "
                "explique em linguagem simples o que foi feito ou encontrado. Não invente resultados.")
        msgs = payload["messages"]
        if msgs and msgs[0].get("role") == "system" and isinstance(msgs[0].get("content"), str):
            msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\n\n" + note}
        else:
            msgs.insert(0, {"role": "system", "content": note})
        return tools, mapping

    def chat_with_tools(self, payload, tools, mapping, max_steps=8, n_ctx=0):
        """Chat em que o modelo pode chamar as ferramentas da Central; o texto vai para a tela como no chat normal."""
        payload = {**payload, "tools": tools, "tool_choice": "auto", "parallel_tool_calls": True}
        messages, last = payload["messages"], None
        fit = ctxfit.Fit(n_ctx, tools)
        for _ in range(max_steps):
            content, calls = "", {}
            with fit.call(lambda p: self.engine_call(p, True), payload) as up:
                for raw in iter(up.readline, b""):
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    try:
                        j = json.loads(line[5:])
                    except ValueError:
                        continue
                    if j.get("timings"):
                        last = raw
                    d = (j.get("choices") or [{}])[0].get("delta") or {}
                    content += d.get("content") or ""
                    for tc in d.get("tool_calls") or []:
                        c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                        c["id"] = tc.get("id") or c["id"]
                        fn = tc.get("function") or {}
                        c["name"] += fn.get("name") or ""
                        c["arguments"] += fn.get("arguments") or ""
                    if d.get("content") or d.get("reasoning_content") or j.get("timings") or j.get("error"):
                        self.sse(j)
            if not calls:
                break
            tcs = [{"id": c["id"] or f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}} for _, c in sorted(calls.items())]
            messages.append({"role": "assistant", "content": content, "tool_calls": tcs})
            for tc in tcs:
                name, cid = tc["function"]["name"], tc["id"]
                srv, tool = mapping.get(name, (None, None))
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except ValueError:
                    args = None
                title = (tool or {}).get("title") or name
                self.sse({"tool": {"stage": "call", "id": cid, "title": title}})
                if not srv:
                    text, ok = f"Ferramenta desconhecida: {name}", False
                elif not isinstance(args, dict):
                    text, ok = "Os argumentos não são um JSON válido. Tente de novo.", False
                else:
                    try:
                        res = srv.call(tool["name"], args)
                        text, _ = mcp.result_text(res, limit=12_000)
                        ok = not res.get("isError")
                    except mcp.MCPError as e:
                        text, ok = f"Erro: {e}", False
                self.sse({"tool": {"stage": "done", "id": cid, "title": title, "ok": ok, "result": text[:600]}})
                messages.append({"role": "tool", "tool_call_id": cid, "content": fit.cap(text)})
        return last

    def proxy_chat(self, b):
        ag = agents.get_agent(b["agent_id"]) if b.get("agent_id") else None
        model = (ag or {}).get("model") or b.get("model") or ""
        device = (ag or {}).get("device") if (ag or {}).get("model") else b.get("device", "auto")
        if not model and S.ENGINE_PROC.state != "ready":
            return self.send_json({"error": "Nenhum modelo carregado."}, 503)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        upstream = lease = None
        try:
            lease = self.engine_for(model, self.loading_notifier(lambda n: {"engine": {"stage": "loading", "model": n}}), device or "auto")
            eng = lease.__enter__()
            self.engine_port = eng.port
            if model:
                self.sse({"engine": {"stage": "ready", "model": S.model_display(eng.model), "device": eng.device}})
            msgs = b.get("messages", [])
            if ag:
                msgs = [{"role": "system", "content": (ag["instructions"] or "Você é um assistente útil.")
                         + f"\n\nVocê é o agente \"{ag['name']}\". Responda no idioma do usuário."}] + \
                       [m for m in msgs if m.get("role") != "system"]
                if ag["temperature"] is not None:
                    b["temperature"] = ag["temperature"]
                if ag["web"]:
                    b["web"] = True
            messages = S.with_reply_rule(S.build_history(msgs, eng.modalities))
            kid = (ag or {}).get("kb") or b.get("kb")
            original = messages
            if kid:
                messages = self.kb_augment(messages, kid)
            if b.get("web"):
                messages = self.web_augment(messages)
            messages = promptcache.MEMORY.stabilize(original, messages, eng.n_ctx)
            payload = {"messages": messages, "stream": True, "cache_prompt": True, "timings_per_token": True}
            for k in ("temperature", "top_p", "top_k", "min_p", "repeat_penalty"):
                if k in b:
                    payload[k] = b[k]
            if int(b.get("max_tokens") or 0) > 0:
                payload["max_tokens"] = int(b["max_tokens"])
            S.apply_model_sampling(payload, eng.model, (ag or {}).get("temperature"))
            last_timings = None
            hub = self.hub_tools(payload) if b.get("hub") and not ag else None
            if hub:
                last_timings = self.chat_with_tools(payload, *hub, n_ctx=eng.n_ctx)
            else:
                upstream = self.engine_call(payload, True)
                for line in iter(upstream.readline, b""):
                    self.wfile.write(line)
                    self.wfile.flush()
                    if b'"timings"' in line:
                        last_timings = line
        except RuntimeError as e:
            try:
                self.sse({"error": {"message": str(e)}})
            except OSError:
                pass
        except (ConnectionError, OSError):
            pass  # usuario clicou em Parar: fechar a conexao interrompe a geracao
        finally:
            if upstream:
                upstream.close()
                try:
                    S.add_stats(json.loads(last_timings.decode("utf-8")[5:])["timings"]) if last_timings else None
                except (ValueError, KeyError):
                    pass
            if lease and lease.eng:
                lease.__exit__(None, None, None)
