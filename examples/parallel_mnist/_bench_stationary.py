import json
import sys
import time
import warnings

sys.path.insert(0, "examples")
from parallel_mnist import run

WIDTHS = [6, 12, 24]
N_STEPS = 8000
SEED = 0

images, labels = run.data.load_mnist("train", pool=4)
out = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for w in WIDTHS:
        cfg = run.Config(width=w, n_steps=N_STEPS, permute_period=0, seed=SEED)
        for name, fn in [
            ("dense", run.run_dense),
            ("block-oracle", run.run_block),
            ("dynamic", run.run_dynamic),
        ]:
            t0 = time.time()
            m, params = fn(cfg, images, labels)
            s = m.summary()
            rec = {
                "width": w,
                "model": name,
                "params": params,
                "avg_loss": s["average_loss"],
                "asy_loss": s["asymptotic_loss"],
                "avg_acc": s["average_accuracy"],
                "asy_acc": s["asymptotic_accuracy"],
                "sec": round(time.time() - t0, 1),
            }
            out.append(rec)
            print(json.dumps(rec), flush=True)

with open("examples/parallel_mnist/_stationary_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("DONE")
