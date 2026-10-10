"""Per-call breakdown of an Nsight Systems trace, for the GPU gap analysis.

Reads an nsys SQLite export (``nsys export --type sqlite``) and, over the last
``--last`` NVTX ranges named ``--range`` (``grow_call`` from
``gpu_gap_probe.py`` or the patched cx ``bench_growth``), prints the median
per call of: the range's wall time, device busy time (the union of kernel,
memcpy and memset intervals), the host lead before the first device op and
tail after the last, idle gaps between device ops, the kernel / memcpy /
memset counts, device time per kernel class, the top kernels, and the CUDA
API calls of the last range. Prints JSON.

    python examples/benchmarks/gpu_gap_nsys.py trace.sqlite --last 9
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import statistics
from typing import Any


def kernel_class(name: str) -> str:
    """Bucket a kernel name into a coarse class.

    Args:
        name: the demangled or short kernel name.

    Returns:
        The class label.
    """
    low = name.lower()
    if "radixsort" in low or "onesweep" in low:
        return "sort (cub radix)"
    if "sort" in low:
        return "sort (XLA)"
    if any(s in low for s in ("scan", "cumsum", "reduce_window", "exclusivesum")):
        return "scan / cumsum"
    if "select" in low and "cub" in low:
        return "scan / cumsum"
    if "scatter" in low or "dynamic_update" in low:
        return "scatter"
    if "gather" in low:
        return "gather"
    if "reduce" in low:
        return "reduce"
    if "claim" in low:
        return "claim"
    return "elementwise / fused"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("db")
    ap.add_argument("--range", default="grow_call")
    ap.add_argument("--last", type=int, default=7)
    args = ap.parse_args()
    con = sqlite3.connect(args.db)
    tables = {r[0] for r in con.execute("select name from sqlite_master")}
    strs = dict(con.execute("select id, value from StringIds"))
    ranges = sorted(
        r
        for r in con.execute(
            "select start, end, coalesce(text, (select value from StringIds"
            " where id = textId)) from NVTX_EVENTS"
        )
        if r[2] == args.range
    )[-args.last :]
    kernels = con.execute(
        "select start, end, shortName, demangledName from CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchall()

    def ops(table: str) -> list[tuple[int, int]]:
        if table not in tables:
            return []
        return con.execute(f"select start, end from {table}").fetchall()

    memcpy, memset = (
        ops("CUPTI_ACTIVITY_KIND_MEMCPY"),
        ops("CUPTI_ACTIVITY_KIND_MEMSET"),
    )
    api = con.execute(
        "select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"
    ).fetchall()
    calls: list[dict[str, Any]] = []
    for s, e, _ in ranges:

        def inside(x: tuple[Any, ...], s: int = s, e: int = e) -> bool:
            return bool(x[0] >= s and x[1] <= e + 1000)

        ks = [k for k in kernels if inside(k)]
        mc = [m for m in memcpy if inside(m)]
        ms = [m for m in memset if inside(m)]
        iv = sorted([(k[0], k[1]) for k in ks] + mc + ms)
        busy = gaps = 0
        cur: list[int] | None = None
        for a, b in iv:
            if cur is None:
                cur = [a, b]
            elif a > cur[1]:
                busy += cur[1] - cur[0]
                gaps += a - cur[1]
                cur = [a, b]
            else:
                cur[1] = max(cur[1], b)
        if cur is not None:
            busy += cur[1] - cur[0]
        by_class: collections.Counter[str] = collections.Counter()
        by_kernel: collections.Counter[str] = collections.Counter()
        for k in ks:
            by_class[kernel_class(strs.get(k[3], strs.get(k[2], "?")))] += k[1] - k[0]
            by_kernel[strs.get(k[2], "?")] += k[1] - k[0]
        apis = collections.Counter(
            strs.get(a[2], "?").split("_v")[0] for a in api if a[0] >= s and a[1] <= e
        )
        calls.append(
            {
                "wall": e - s,
                "busy": busy,
                "gaps": gaps,
                "lead": iv[0][0] - s if iv else 0,
                "tail": e - iv[-1][1] if iv else 0,
                "nk": len(ks),
                "nm": len(mc),
                "nmset": len(ms),
                "by_class": by_class,
                "by_kernel": by_kernel,
                "apis": apis,
            }
        )

    def med(key: str, scale: float = 1e3) -> float:
        return statistics.median(c[key] for c in calls) / scale

    def med_counter(key: str) -> dict[str, float]:
        names = set().union(*(c[key] for c in calls))
        vals = {
            n: statistics.median(c[key].get(n, 0) for c in calls) / 1e3 for n in names
        }
        return dict(sorted(vals.items(), key=lambda kv: -kv[1]))

    out: dict[str, Any] = {k: med(k) for k in ("wall", "busy", "gaps", "lead", "tail")}
    out.update({k: med(k, 1.0) for k in ("nk", "nm", "nmset")})
    out["by_class_us"] = med_counter("by_class")
    out["top_kernels_us"] = dict(list(med_counter("by_kernel").items())[:25])
    out["api_counts_last_call"] = dict(calls[-1]["apis"].most_common(12))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
