"""Contexto: tarefa longa que nao cabe, contexto automatico com a placa ocupada, grupos de agentes e lista de conversas."""
import contextlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import agent  # noqa: E402
import agents  # noqa: E402
import ctxfit  # noqa: E402

GIB = 2 ** 30


def tool_step(n, size=6000, args="{}"):
    return [{"role": "assistant", "content": "", "tool_calls": [{"id": f"c{n}", "type": "function", "function": {"name": "read_file", "arguments": args}}]},
            {"role": "tool", "tool_call_id": f"c{n}", "content": f"resultado {n} " + "x" * size}]


class FitTest(unittest.TestCase):
    def conversation(self, steps=8):
        return [{"role": "system", "content": "s"}, {"role": "user", "content": "pedido"}] + [m for n in range(steps) for m in tool_step(n)]

    def test_nothing_changes_while_it_fits(self):
        msgs = self.conversation()
        before = json.dumps(msgs)
        self.assertFalse(ctxfit.Fit(32768).fit(msgs))
        self.assertEqual(json.dumps(msgs), before)

    def test_shrinks_old_results_and_leaves_room(self):
        msgs = self.conversation()
        originals = list(msgs)
        f = ctxfit.Fit(8192)
        self.assertTrue(f.fit(msgs))
        self.assertLessEqual(f.tokens(msgs), 8192 * 0.55, "sobra folga: nao encurta de novo no passo seguinte")
        self.assertFalse(f.fit(msgs))
        self.assertIn("encurtado", msgs[3]["content"], "o mais velho primeiro")
        self.assertEqual(msgs[-1], originals[-1], "o resultado do passo atual fica inteiro")
        self.assertEqual(msgs[1]["content"], "pedido")
        self.assertGreater(len(originals[3]["content"]), 6000, "a conversa salva (trace) nao muda")
        self.assertEqual([m.get("tool_call_id") for m in msgs], [m.get("tool_call_id") for m in originals])

    def test_shortens_written_files_keeping_valid_json(self):
        big = json.dumps({"path": "a.py", "content": "y" * 20000})
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "p"}] + tool_step(0, 10, big) + tool_step(1, 10)
        self.assertTrue(ctxfit.Fit(4096).fit(msgs))
        args = json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(args["path"], "a.py")
        self.assertLess(len(args["content"]), 400)

    def test_single_huge_result_is_cut(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "p"}] + tool_step(0, 90000)
        f = ctxfit.Fit(8192)
        self.assertTrue(f.fit(msgs))
        self.assertLessEqual(f.tokens(msgs), 8192 - f.reserve())

    def test_result_limit_follows_context(self):
        self.assertEqual(ctxfit.Fit(0).result_limit(), 60000)
        self.assertEqual(ctxfit.Fit(16384).result_limit(), 12288)
        self.assertEqual(ctxfit.Fit(262144).result_limit(), 60000)
        out = ctxfit.Fit(16384).cap("a" * 50000)
        self.assertLess(len(out), 12500)
        self.assertIn("omitidos", out)

    def test_learns_real_token_count(self):
        f = ctxfit.Fit(8192)
        f.learn(20000, {"prompt_n": 9000, "cache_n": 1000})
        self.assertAlmostEqual(f.per_token, 2.0, 1)
        f.learn(1000, {"prompt_n": 5})
        self.assertAlmostEqual(f.per_token, 2.0, 1, "pedido pequeno nao serve de medida")

    def test_retries_once_when_engine_refuses(self):
        calls = []

        def engine(payload):
            calls.append(len(json.dumps(payload["messages"])))
            if len(calls) == 1:
                raise RuntimeError("the request exceeds the available context size (8192 tokens), try increasing it")
            return "ok"
        msgs = self.conversation()
        self.assertEqual(ctxfit.Fit(0).call(engine, {"messages": msgs}), "ok")
        self.assertLess(calls[1], calls[0])

        def always(payload):
            raise RuntimeError("request exceeds the available context size")
        with self.assertRaises(RuntimeError) as cm:
            ctxfit.Fit(0).call(always, {"messages": self.conversation()})
        self.assertIn("não cabe no contexto", str(cm.exception))

        def other(payload):
            raise RuntimeError("Motor nao respondeu")
        with self.assertRaises(RuntimeError) as cm:
            ctxfit.Fit(0).call(other, {"messages": self.conversation()})
        self.assertIn("Motor nao respondeu", str(cm.exception))

    def test_read_file_respects_limit(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "g.txt").write_text("\n".join("linha grande " * 5 for _ in range(2000)), "utf-8")
            out = agent.tool_read_file(agent.Workspace(d), "g.txt", limit=5000)
            self.assertLess(len(out), 5400)
            self.assertIn("Leia a partir da linha", out)


class GroupPrompt(unittest.TestCase):
    """O comeco do pedido e igual para todos os agentes do grupo; o papel vem no fim."""

    def run_steps(self, n):
        sent = []

        class H:
            def reply_rule(self):
                return ""

            def split_content(self, m, mod):
                return m["content"], []

            def apply_model_sampling(self, p, rel, t=None):
                return p

            @contextlib.contextmanager
            def _cm(self):
                class Up:
                    lines = [b'data: {"choices":[{"delta":{"content":"feito"}}]}\n', b""]

                    def readline(self):
                        return self.lines.pop(0) if self.lines else b""
                u = Up()
                u.lines = list(Up.lines)
                yield u

            def engine_call(self, payload, stream):
                sent.append(payload["messages"])
                return self._cm()

        class Eng:
            model, device, modalities, n_ctx = "m.gguf", "gpu", {}, 8192
        ags = {str(i): {"id": str(i), "name": f"A{i}", "icon": "x", "color": "#fff", "instructions": f"Papel {i}.", "web": False,
                        "temperature": None, "kb": "", "mcp": []} for i in range(n)}
        steps = [{"agent": str(i), "role": "agent"} for i in range(n)]
        group = {"id": "g", "name": "Time", "description": "Fazer jogos", "max_retries": 0}
        hist = [{"role": "user", "content": "antes"}, {"role": "assistant", "content": "ok"}]
        outs = {}
        for i in range(n):
            outs[i] = agents._run_step(H(), lambda o: True, group, ags, steps, i, ags[str(i)], i == n - 1 and n > 1, Eng, {"content": "crie"}, hist, outs, {}, {}, {})[0]
        return sent

    def test_shared_prefix(self):
        a, b, c = self.run_steps(3)
        self.assertEqual(a[:-1], b[:-1], "sistema e conversa iguais")
        self.assertEqual(b[:-1], c[:-1])
        cut = b[-1]["content"].index("\n\n---\nSUA FUNÇÃO")
        self.assertTrue(c[-1]["content"].startswith(b[-1]["content"][:cut]), "o trabalho anterior so cresce no fim")
        self.assertIn("Papel 1.", b[-1]["content"][cut:])
        self.assertNotIn("Papel 1.", b[0]["content"])
        self.assertIn("Fazer jogos", a[0]["content"])
        self.assertIn(agents.VALIDATOR_SUFFIX.strip()[:40], c[-1]["content"])

    def test_single_agent_keeps_role_in_system(self):
        (a,) = self.run_steps(1)
        self.assertIn("Papel 0.", a[0]["content"])
        self.assertNotIn("SUA FUNÇÃO", a[-1]["content"])


class ServerParts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "server" not in sys.modules:
            cls.tmp = tempfile.TemporaryDirectory()
            os.environ["LLM_HOME"] = cls.tmp.name
        import server
        cls.s = server

    def patched(self, **kw):
        old = {k: getattr(self.s, k) for k in kw}
        self.addCleanup(lambda: [setattr(self.s, k, v) for k, v in old.items()])
        for k, v in kw.items():
            setattr(self.s, k, v)

    def test_auto_context_uses_free_vram(self):
        seen = []
        self.patched(GPU={"vendor": "nvidia", "vram_mb": 16000}, gpu_sample=lambda: (10000, 16000, 0), gguf_context_length=lambda p: 131072,
                     fit_context=lambda rel, kv="", free=None: seen.append(free) or 40000)
        main, extra = self.s.ENGINE_PROC, self.s.EngineProcess(1)
        self.assertEqual(self.s.auto_context("m", "f16", main), 40000)
        self.assertEqual(seen[-1], 6000 * 2 ** 20, "conta so o que esta livre")
        self.assertEqual(self.s.auto_context("m", "f16", extra), self.s.EXTRA_CTX, "um extra nao toma a placa do principal")
        self.assertEqual(self.s.auto_context("m", "f16", main, on_cpu=True), 16384)

    def test_auto_context_when_it_does_not_fit(self):
        self.patched(GPU={"vendor": "nvidia", "vram_mb": 16000}, gpu_sample=lambda: None, gguf_context_length=lambda p: 131072,
                     fit_context=lambda *a, **k: 0)
        self.assertEqual(self.s.auto_context("m", "f16", self.s.ENGINE_PROC), self.s.EXTRA_CTX)
        self.s.fit_context = lambda *a, **k: 8192
        self.assertEqual(self.s.auto_context("m", "f16", self.s.ENGINE_PROC), self.s.EXTRA_CTX, "16 mil e o minimo do principal")
        self.assertEqual(self.s.auto_context("m", "f16", self.s.EngineProcess(1)), 8192, "o extra fica no que cabe")
        self.s.gguf_context_length = lambda p: 4096
        self.assertEqual(self.s.auto_context("m", "f16", self.s.ENGINE_PROC), 4096, "nunca passa do treino do modelo")

    def test_vision_and_accelerator_count_once(self):
        with tempfile.TemporaryDirectory() as d:
            for n, size in (("m.gguf", 1000), ("v.gguf", 100), ("a.gguf", 10)):
                (Path(d) / n).write_bytes(b"x" * size)
            seen = []
            self.patched(MODELS=Path(d), GPU={"vendor": "nvidia", "vram_mb": 16000}, mmproj_for=lambda r: Path(d) / "v.gguf",
                         mtp_for=lambda r: Path(d) / "a.gguf", accel_for=lambda r: ("mtp", Path(d) / "a.gguf"))
            real = self.s.gguf.best_context
            self.s.gguf.best_context = lambda meta, size, extra, *a, **k: seen.append(size + extra)
            try:
                self.s.fit_context("m.gguf", "f16")
            finally:
                self.s.gguf.best_context = real
            self.assertEqual(seen, [1110])

    def test_free_vram_without_measure_discounts_other_models(self):
        other = self.s.EngineProcess(1)
        other.state, other.device, other.model = "ready", "gpu", "outro.gguf"
        self.patched(GPU={"vendor": "amd", "vram_mb": 16384}, gpu_sample=lambda: None, model_bytes=lambda rel: 4 * GIB)
        self.s.POOL.extra[("outro.gguf", "gpu")] = other
        self.addCleanup(lambda: self.s.POOL.extra.pop(("outro.gguf", "gpu"), None))
        self.assertEqual(self.s.free_vram(self.s.ENGINE_PROC), 16 * GIB - self.s.gguf.DESKTOP - 4 * GIB - self.s.POOL.OVERHEAD)

    def test_chat_list_is_cached_and_notices_changes(self):
        with tempfile.TemporaryDirectory() as d:
            self.patched(CHATS=Path(d), _chats_cache={})
            f = Path(d) / "a.json"
            f.write_text(json.dumps({"id": "a", "title": "um", "updated": 1}), "utf-8")
            (Path(d) / "ruim.json").write_text("{", "utf-8")
            self.assertEqual([c["title"] for c in self.s.list_chats()], ["um"])
            real, reads = Path.read_text, []
            Path.read_text = lambda p, *a, **k: reads.append(p.name) or real(p, *a, **k)
            try:
                self.s.list_chats()
            finally:
                Path.read_text = real
            self.assertNotIn("a.json", reads, "nao le de novo o que nao mudou")
            time.sleep(0.02)
            f.write_text(json.dumps({"id": "a", "title": "dois novo", "updated": 2}), "utf-8")
            (Path(d) / "b.json").write_text(json.dumps({"id": "b", "title": "b", "updated": 9}), "utf-8")
            self.assertEqual([c["title"] for c in self.s.list_chats()], ["b", "dois novo"])
            f.unlink()
            self.assertEqual([c["id"] for c in self.s.list_chats()], ["b"])


if __name__ == "__main__":
    unittest.main()
