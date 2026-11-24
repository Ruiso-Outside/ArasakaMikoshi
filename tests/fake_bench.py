"""llama-bench falso: o lote maior le mais rapido; com FAKE_BENCH_FAIL, fecha como o motor com o Visual C++ antigo."""
import json
import os
import sys

args = sys.argv[1:]
if os.environ.get("FAKE_BENCH_FAIL"):
    sys.exit("erro: nao rodou")
ub = int(args[args.index("-ub") + 1])
kv = args[args.index("-ctk") + 1]
if ub == 2048:
    sys.exit("faltou memoria")  # uma configuracao que nao cabe nao derruba o teste
print("load_backend: texto que nao e json")
print(json.dumps({"n_prompt": 1536, "n_gen": 0, "n_ubatch": ub, "type_k": kv, "avg_ts": 1000.0 + ub}))
print(json.dumps({"n_prompt": 0, "n_gen": 64, "n_ubatch": ub, "type_k": kv, "avg_ts": 40.0}))
