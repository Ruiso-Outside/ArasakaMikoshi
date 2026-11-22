"""Motor Strata no servidor: inicia um modelo (Strata falso) na porta do motor, conversa e para."""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
from test_server import free_port  # noqa: E402


class StrataServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.home = Path(cls._tmp.name)
        # o comando `strata` instalado: aponta para o falso (o Mikoshi procura em STRATA_DATA_DIR/terminal/current/bin)
        bindir = cls.home / "strata" / "terminal" / "current" / "bin"
        bindir.mkdir(parents=True)
        if os.name == "nt":
            (bindir / "strata.cmd").write_text(f'@"{sys.executable}" "{TESTS / "fake_strata.py"}" %*\r\n', "utf-8")
        else:
            exe = bindir / "strata"
            exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{TESTS / "fake_strata.py"}" "$@"\n', "utf-8")
            exe.chmod(0o755)
        cls.port = free_port()
        cls.base = f"http://127.0.0.1:{cls.port}"
        env = dict(os.environ, LLM_HOME=str(cls.home), LLM_PORT=str(cls.port), PYTHONIOENCODING="utf-8",
                   STRATA_DATA_DIR=str(cls.home / "strata"), ARASAKA_CONSTRUCT_EDITOR=str(cls.home / "sem-construct"))
        env.pop("LLM_PARENT_PID", None)
        cls.proc = subprocess.Popen([sys.executable, str(TESTS / "run_server.py")], env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for _ in range(100):
            try:
                cls.call("/api/status")
                break
            except (urllib.error.URLError, OSError):
                time.sleep(0.2)
        else:
            cls.proc.kill()
            raise RuntimeError("servidor não subiu: " + cls.proc.stdout.read().decode("utf-8", "replace"))

    @classmethod
    def tearDownClass(cls):
        try:
            cls.call("/api/unload", {"all": True})
        except (urllib.error.URLError, OSError):
            pass
        if os.name == "nt":
            subprocess.run(["taskkill", "/pid", str(cls.proc.pid), "/T", "/F"], capture_output=True)
        else:
            cls.proc.kill()
        cls.proc.wait(10)
        cls.proc.stdout.close()
        time.sleep(0.5)
        cls._tmp.cleanup()

    @classmethod
    def call(cls, path, body=None, method=None):
        req = urllib.request.Request(cls.base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method=method)
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read().decode("utf-8"))

    def wait_engine(self, secs=40):
        for _ in range(secs * 5):
            e = self.call("/api/status")["engine"]
            if e["state"] in ("ready", "error"):
                return e
            time.sleep(0.2)
        return e

    def test_launch_chat_and_stop(self):
        st = self.call("/api/strata")
        self.assertTrue(st["installed"])
        self.assertEqual(st["version"], "0.2.19")
        self.assertEqual(st["catalog"][0]["family"], "qwen")

        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.call("/api/strata/launch", {"family": "qwen; rm", "model": "q2_0"})
        self.assertEqual(cm.exception.code, 400)

        self.assertEqual(self.call("/api/strata/launch", {"family": "qwen", "model": "Q2_0", "context": "64k"})["model"],
                         "strata:qwen/q2_0")
        e = self.wait_engine()
        self.assertEqual((e["state"], e["backend"], e["model"]), ("ready", "strata", "strata:qwen/q2_0"), e.get("error"))
        self.assertIn("Strata", e["label"])
        log = "\n".join(self.call("/api/log")["log"])
        self.assertIn("--context 64k", log)
        self.assertIn("--access local", log)
        self.assertIn("teste: modelo falso", log)

        # o motor principal (porta do app + 1) agora e o servidor do Strata
        req = urllib.request.Request(f"http://127.0.0.1:{self.port + 1}/v1/chat/completions",
                                     data=json.dumps({"messages": [{"role": "user", "content": "oi"}]}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            self.assertEqual(r.status, 200)

        self.call("/api/unload", {})
        self.assertEqual(self.call("/api/status")["engine"]["state"], "stopped")
        time.sleep(0.5)
        with self.assertRaises((urllib.error.URLError, OSError)):
            urllib.request.urlopen(f"http://127.0.0.1:{self.port + 1}/health", timeout=2)

    def test_unknown_family_shows_strata_error(self):
        self.call("/api/strata/launch", {"family": "inexistente", "model": "x1"})
        e = self.wait_engine()
        self.assertEqual(e["state"], "error")
        self.assertIn("not a model family", e["error"])


if __name__ == "__main__":
    unittest.main()
