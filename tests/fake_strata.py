"""Comando `strata` falso para os testes: `launch` sobe o llama-server falso na porta pedida, `launch --stop` o encerra."""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

TESTS = Path(__file__).resolve().parent
args = sys.argv[1:]
PID = Path(os.environ.get("LLM_HOME", TESTS)) / "fake_strata.pid"


def out(obj):
    print(json.dumps({"schema_version": 1, **obj}), flush=True)


if args[:1] == ["--version"]:
    print("strata 0.2.19 (teste)")
    sys.exit(0)
if args[:1] != ["launch"]:
    sys.exit(2)
if "--stop" in args:
    if PID.exists():
        pid = int(PID.read_text())
        if os.name == "nt":
            subprocess.run(["taskkill", "/pid", str(pid), "/T", "/F"], capture_output=True)
        else:
            try:
                os.kill(pid, 9)
            except OSError:
                pass
        PID.unlink()
    print("Stopped.")
    sys.exit(0)
family, model, port = args[args.index("--family") + 1], args[args.index("--model") + 1], int(args[args.index("--port") + 1])
if family not in ("qwen", "coder", "unsloth", "gyro", "qwen36"):
    out({"ok": False, "error": {"code": 2, "message": f"'{family}' is not a model family Strata knows."}})
    sys.exit(2)
print("Engine: downloading strata-engine-cuda (teste)", flush=True)
flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
p = subprocess.Popen([sys.executable, str(TESTS / "fake_llama.py"), "--port", str(port), "-m", f"strata-{family}-{model}.gguf", "-c", "32768"],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags,
                     start_new_session=os.name != "nt")
PID.write_text(str(p.pid))
for _ in range(100):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            if r.status == 200:
                break
    except OSError:
        time.sleep(0.2)
print("Strata is serving.", flush=True)
out({"ok": True, "name": f"{family} {model}", "profile": "launch-teste", "urls": [f"http://127.0.0.1:{port}"], "reused": False,
     "api_key": None, "tuned": "not tuned", "notices": ["teste: modelo falso"]})
