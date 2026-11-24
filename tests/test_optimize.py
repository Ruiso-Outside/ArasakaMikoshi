"""Otimizacao do motor: memoria do contexto por arquitetura, Visual C++ antigo, cache do prompt e escolha do Otimizar."""
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import gguf  # noqa: E402
import modelsettings  # noqa: E402
import promptcache  # noqa: E402
import tune  # noqa: E402
import winrt  # noqa: E402

GIB = 1024 ** 3
LLAMA = {"general.architecture": "llama", "llama.block_count": 32, "llama.attention.head_count": 32,
         "llama.attention.head_count_kv": 8, "llama.embedding_length": 4096, "llama.context_length": 131072}
QWEN = {"general.architecture": "qwen35", "qwen35.block_count": 65, "qwen35.nextn_predict_layers": 1, "qwen35.attention.head_count": 24,
        "qwen35.attention.head_count_kv": 4, "qwen35.attention.key_length": 256, "qwen35.attention.value_length": 256,
        "qwen35.full_attention_interval": 4, "qwen35.embedding_length": 5120, "qwen35.context_length": 262144}
GEMMA = {"general.architecture": "gemma4", "gemma4.block_count": 30, "gemma4.attention.head_count": 16,
         "gemma4.attention.head_count_kv": ([8] * 5 + [2]) * 5, "gemma4.attention.sliding_window": 1024,
         "gemma4.attention.sliding_window_pattern": ([True] * 5 + [False]) * 5, "gemma4.attention.key_length": 512,
         "gemma4.attention.value_length": 512, "gemma4.attention.key_length_swa": 256, "gemma4.attention.value_length_swa": 256,
         "gemma4.embedding_length": 2816, "gemma4.context_length": 262144}


class ContextMemorySize(unittest.TestCase):
    def test_llama_full_attention(self):
        # 32 camadas x 8 cabecas x (128 + 128) x 64000 tokens x 2 bytes
        self.assertEqual(gguf.kv_bytes(LLAMA, 64000), 32 * 8 * 256 * 64000 * 2)
        self.assertAlmostEqual(gguf.kv_bytes(LLAMA, 64000) / GIB, 7.81, 2)
        self.assertAlmostEqual(gguf.kv_bytes(LLAMA, 64000, "q8_0") / gguf.kv_bytes(LLAMA, 64000), 0.53125, 4)

    def test_hybrid_counts_only_attention_layers(self):
        self.assertEqual(gguf.kv_bytes(QWEN, 64000), 16 * 4 * 512 * 64000 * 2)  # 1 em cada 4 das 64 (a 65a e do MTP)

    def test_sliding_window_does_not_grow(self):
        per_full = 2 * 1024 * 2
        cells = -(-(1024 * 4 + 512) // 256) * 256
        self.assertEqual(gguf.kv_bytes(GEMMA, 64000), 5 * per_full * 64000 + 25 * 8 * 512 * 2 * cells)
        self.assertEqual(gguf.kv_bytes(GEMMA, 64000) - gguf.kv_bytes(GEMMA, 32000), 5 * per_full * 32000)

    def test_best_context(self):
        vram = 16303 * 2 ** 20
        self.assertEqual(gguf.best_context(LLAMA, int(6.14 * GIB), 0, vram), 49152)
        self.assertGreater(gguf.best_context(LLAMA, int(6.14 * GIB), 0, vram, "q8_0"), 49152)
        self.assertEqual(gguf.best_context(LLAMA, int(15 * GIB), 0, vram), 0, "nao cabe nem com o menor contexto")
        self.assertEqual(gguf.best_context({**LLAMA, "llama.context_length": 8192}, 4 * GIB, 0, vram), 8192, "nunca passa do treino")
        self.assertLess(gguf.best_context(LLAMA, int(6.14 * GIB), 0, vram, free=9 * GIB), 16384, "com um jogo aberto cabe menos")
        self.assertIsNone(gguf.best_context({}, GIB, 0, vram), "arquivo sem cabecalho: nao da para estimar")

    def test_unknown_architecture_is_not_guessed(self):
        self.assertEqual(gguf.kv_bytes({**LLAMA, "llama.attention.key_length_mla": 64}, 8192), 0)

    def test_reads_header(self):
        def text(s):
            return struct.pack("<Q", len(s)) + s.encode()
        kvs = [text("general.architecture") + struct.pack("<I", 8) + text("llama"),
               text("llama.block_count") + struct.pack("<II", 4, 32),
               text("llama.attention.head_count_kv") + struct.pack("<IIQ3I", 9, 4, 3, 8, 2, 8),
               text("llama.context_length") + struct.pack("<II", 4, 4096),
               text("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 999999)]  # cortado aqui de proposito
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.gguf"
            p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + b"".join(kvs))
            m = gguf.read_meta(p)
            self.assertEqual((m["general.architecture"], m["llama.block_count"], m["llama.attention.head_count_kv"]), ("llama", 32, [8, 2, 8]))
            self.assertEqual(gguf.context_length(p), 4096)


class OldVisualCpp(unittest.TestCase):
    def test_explains_only_the_known_crash(self):
        self.assertIn("14.28", winrt.crash_hint(-1073741819, (14, 28)))
        self.assertIn("winget", winrt.crash_hint(0xC0000005, (14, 28)))
        self.assertIn("não está instalado", winrt.crash_hint(0xC0000135, None))
        self.assertEqual(winrt.crash_hint(0xC0000005, (14, 51)), "", "runtime novo: a causa e outra")
        self.assertEqual(winrt.crash_hint(1, (14, 28)), "", "erro normal do motor")

    @unittest.skipUnless(sys.platform == "win32", "so no Windows")
    def test_reads_installed_version(self):
        v = winrt.vc_runtime_version()
        self.assertTrue(v is None or v[0] == 14, v)


class PromptCache(unittest.TestCase):
    def turn(self, mem, msgs, ctx="", n_ctx=32768):
        return mem.stabilize(msgs, promptcache.prepend(msgs, ctx) if ctx else msgs, n_ctx)

    def test_previous_turn_stays_identical(self):
        mem = promptcache.ContextMemory()
        sysm = {"role": "system", "content": "s"}
        a = [sysm, {"role": "user", "content": "pergunta 1"}]
        sent1 = self.turn(mem, a, "[trechos A]\n")
        self.assertEqual(sent1[1]["content"], "[trechos A]\npergunta 1")
        b = a + [{"role": "assistant", "content": "r1"}, {"role": "user", "content": "pergunta 2"}]
        sent2 = self.turn(mem, b, "[trechos B]\n")
        self.assertEqual(sent2[:2], sent1, "o comeco e igual ao do pedido anterior: o motor reaproveita o cache")
        self.assertEqual(sent2[3]["content"], "[trechos B]\npergunta 2")
        self.assertEqual(a[1]["content"], "pergunta 1", "a conversa salva nao muda")

    def test_other_conversation_and_edited_message_do_not_match(self):
        mem = promptcache.ContextMemory()
        self.turn(mem, [{"role": "user", "content": "oi"}], "[A]\n")
        other = [{"role": "user", "content": "outra"}, {"role": "assistant", "content": "x"}, {"role": "user", "content": "oi"}]
        self.assertEqual(self.turn(mem, other), other)

    def test_drops_when_it_does_not_fit(self):
        mem = promptcache.ContextMemory()
        a = [{"role": "user", "content": "p"}]
        self.turn(mem, a, "x" * 50000)
        b = a + [{"role": "assistant", "content": "r"}, {"role": "user", "content": "q"}]
        self.assertEqual(self.turn(mem, b, n_ctx=8192), b)
        self.assertNotEqual(self.turn(mem, b, n_ctx=65536), b)

    def test_memory_is_bounded(self):
        mem = promptcache.ContextMemory()
        for i in range(promptcache.MAX_ITEMS + 50):
            self.turn(mem, [{"role": "user", "content": f"p{i}"}], "y" * 100)
        self.assertEqual(len(mem.items), promptcache.MAX_ITEMS)
        self.assertEqual(mem.chars, sum(len(v) for v in mem.items.values()))

    def test_images_keep_their_parts(self):
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:x"}}, {"type": "text", "text": "o que e?"}]}]
        out = promptcache.prepend(msgs, "[doc] ")
        self.assertEqual(out[0]["content"][1]["text"], "[doc] o que e?")
        self.assertEqual(msgs[0]["content"][1]["text"], "o que e?")


class Tune(unittest.TestCase):
    def row(self, ub, pp, tg, kv="f16"):
        return {"ubatch": ub, "kv": kv, "pp": pp, "tg": tg}

    def test_parse(self):
        out = 'lixo\n{"n_prompt": 1536, "n_gen": 0, "avg_ts": 1234.56}\n{"n_prompt": 0, "n_gen": 64, "avg_ts": 40.04}\n{quebrado\n'
        self.assertEqual(tune.parse(out), {"pp": 1234.6, "tg": 40.0})
        self.assertIsNone(tune.parse("nada"))

    def test_choose_needs_clear_gain(self):
        self.assertEqual(tune.choose([self.row(512, 1000, 40), self.row(1024, 1030, 40)])[1], {}, "3% e ruido")
        self.assertEqual(tune.choose([self.row(512, 1000, 40), self.row(1024, 1500, 40)])[1], {"ubatch": 1024})
        self.assertEqual(tune.choose([self.row(512, 1000, 40), self.row(2048, 2000, 30)])[1], {}, "nao troca se escrever mais devagar")

    def test_choose_8bit_only_when_writing_gets_faster(self):
        self.assertEqual(tune.choose([self.row(512, 1000, 40), self.row(512, 1500, 41, "q8_0")])[1], {})
        self.assertEqual(tune.choose([self.row(512, 1000, 10), self.row(512, 1000, 25, "q8_0")])[1], {"kv_cache": "q8_0"})

    def test_command(self):
        c = tune.command("b", "m.gguf", 16384, 1024, "q8_0", 8)
        self.assertEqual((c[c.index("-fitc") + 1], c[c.index("-ub") + 1], c[c.index("-ctv") + 1], c[c.index("-t") + 1]), ("16384", "1024", "q8_0", "8"))
        self.assertNotIn("-fitt", tune.command("b", "m", 8192, gpu=False))


class Settings(unittest.TestCase):
    def test_kv_cache_and_batch_per_model(self):
        with tempfile.TemporaryDirectory() as d:
            old = modelsettings.FILE
            try:
                modelsettings.configure(d)
                self.assertEqual(modelsettings.save("m", {"kv_cache": "q8_0", "ubatch": "1024", "ctx_size": 0}),
                                 {"kv_cache": "q8_0", "ubatch": 1024, "ctx_size": 0})
                self.assertEqual(modelsettings.save("m", {"kv_cache": "--host 0.0.0.0"}), {}, "so os tipos conhecidos viram opcao do motor")
            finally:
                modelsettings.FILE = old


if __name__ == "__main__":
    unittest.main()
