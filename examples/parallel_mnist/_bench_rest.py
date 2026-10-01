import json
import sys
import time
import warnings

sys.path.insert(0, "examples")
from parallel_mnist import run

images, labels = run.data.load_mnist("train", pool=4)

# --- dynamic stationary at scale-adjusted prune threshold ---
dyn_stat = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for w in [6, 12, 24]:
        cfg = run.Config(
            width=w, n_steps=8000, permute_period=0, seed=0, prune_threshold=1e-4
        )
        t0 = time.time()
        m, params = run.run_dynamic(cfg, images, labels)
        s = m.summary()
        rec = {
            "width": w,
            "model": "dynamic(pt=1e-4)",
            "params": params,
            "avg_loss": s["average_loss"],
            "asy_loss": s["asymptotic_loss"],
            "avg_acc": s["average_accuracy"],
            "asy_acc": s["asymptotic_accuracy"],
            "cross_final": m.cross_task_frac[-1],
            "sec": round(time.time() - t0, 1),
        }
        dyn_stat.append(rec)
        print("STAT", json.dumps(rec), flush=True)

# --- non-stationary sweep ---
nonstat = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for pp in [2000, 5000, 20000]:
        for name, fn, extra in [
            ("dense", run.run_dense, {}),
            ("block-oracle", run.run_block, {}),
            ("dynamic(pt=1e-4)", run.run_dynamic, {"prune_threshold": 1e-4}),
        ]:
            cfg = run.Config(
                width=12, n_steps=12000, permute_period=pp, seed=0, **extra
            )
            t0 = time.time()
            m, params = fn(cfg, images, labels)
            s = m.summary()
            rec = {
                "period": pp,
                "model": name,
                "params": params,
                "avg_loss": s["average_loss"],
                "asy_loss": s["asymptotic_loss"],
                "avg_acc": s["average_accuracy"],
                "asy_acc": s["asymptotic_accuracy"],
                "cross_final": (m.cross_task_frac[-1] if m.cross_task_frac else None),
                "sec": round(time.time() - t0, 1),
            }
            nonstat.append(rec)
            print("NONSTAT", json.dumps(rec), flush=True)

with open("examples/parallel_mnist/_dyn_stationary_results.json", "w") as f:
    json.dump(dyn_stat, f, indent=2)
with open("examples/parallel_mnist/_nonstationary_results.json", "w") as f:
    json.dump(nonstat, f, indent=2)
print("DONE")
