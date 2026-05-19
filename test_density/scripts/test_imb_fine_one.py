#!/usr/bin/env python3
"""
冒烟测试：flow_dyn_int_imb_fine_one 能否成功跑通（全手动欧拉版本）。

用法：
    python test_density/scripts/test_imb_fine_one.py
    python test_density/scripts/test_imb_fine_one.py --L 4 --dim 1 --qmax 3000
    python test_density/scripts/test_imb_fine_one.py --store-flow
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import importlib
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

import numpy as np
import models.models as models

_mod = importlib.import_module("core.diag_routines.spinless_fermion copy")
flow_dyn_int_imb_fine_one = _mod.flow_dyn_int_imb_fine_one


def make_dl_list(lmax: float, qmax: int) -> np.ndarray:
    return np.logspace(np.log10(0.001), np.log10(lmax), qmax,
                       endpoint=True, base=10)


def _check_array(name: str, arr, expected_shape=None):
    if arr is None:
        print(f"    {name:22s}  MISSING")
        return False
    a = np.asarray(arr)
    shape_ok = (expected_shape is None) or (a.shape == expected_shape)
    finite = bool(np.all(np.isfinite(a))) if a.size else True
    status = "OK" if (shape_ok and finite) else "FAIL"
    extra = []
    if expected_shape is not None and not shape_ok:
        extra.append(f"want {expected_shape}")
    if not finite:
        extra.append("non-finite")
    extra_str = f"  ({', '.join(extra)})" if extra else ""
    amax = float(np.max(np.abs(a))) if a.size and finite else float("nan")
    print(f"    {name:22s}  shape={str(a.shape):20s}  |max|={amax:.3e}  [{status}]{extra_str}")
    return shape_ok and finite


def run_smoke(
    L: int = 2,
    dim: int = 2,
    qmax: int = 3000,
    lmax: float = 100.0,
    cutoff: float = 1e-3,
    dis: float = 5.0,
    dis_type: str = "linear",
    seed: int = 42,
    store_flow: bool = False,
) -> dict:
    n = L ** dim
    tlist = [0, 10.0]
    nt = len(tlist)
    J = 1.0
    delta = 0.1
    x = 0.0

    os.environ["PYFLOW_SEED"] = str(seed)
    ham = models.hamiltonian("spinless fermion", dis_type, intr=True)
    ham.build(n, dim, dis, J, x, delta=delta)

    dl_list = make_dl_list(lmax, qmax)

    num = np.zeros((n, n), dtype=np.float64)
    num[n // 2, n // 2] = 1.0
    num_int = np.zeros((n, n, n, n), dtype=np.float64)

    print("-" * 70)
    print("  flow_dyn_int_imb_fine_one — smoke test（全手动欧拉）")
    print("-" * 70)
    t0 = time.perf_counter()
    res = flow_dyn_int_imb_fine_one(
        n, ham, num, num_int, dl_list, qmax, cutoff,
        tlist=tlist, store_flow=store_flow,
    )
    elapsed = time.perf_counter() - t0
    print(f"        Time: {elapsed:.3f}s")
    print()

    print("=" * 70)
    print("  RETURN VALUES")
    print("=" * 70)

    checks = []
    checks.append(_check_array("H0_diag", res.get("H0_diag"), (n, n)))
    checks.append(_check_array("Hint", res.get("Hint"), (n, n, n, n)))
    checks.append(_check_array("LIOM Interactions", res.get("LIOM Interactions"), (n - 1,)))
    checks.append(_check_array("Imbalance", res.get("Imbalance"), (nt,)))

    if store_flow:
        checks.append(_check_array("flow2", res.get("flow2")))
        checks.append(_check_array("flow4", res.get("flow4")))
        checks.append(_check_array("dl_list", res.get("dl_list")))

    all_pass = all(checks)

    imb = np.asarray(res.get("Imbalance"))
    if imb is not None and imb.size:
        print()
        print("  Imbalance(t) preview:")
        for i in [0, nt - 1]:
            print(f"      t[{i:2d}]={tlist[i]:.3f}   Imbalance = {float(imb[i]):+.6e}")

    print()
    print("=" * 70)
    print(f"  >>> {'SMOKE TEST PASSED' if all_pass else 'SMOKE TEST FAILED'} <<<")
    print("=" * 70)

    return {"result": res, "passed": all_pass, "elapsed": elapsed}


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for flow_dyn_int_imb_fine_one (全手动欧拉)"
    )
    parser.add_argument("--L", type=int, default=2)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--qmax", type=int, default=3000)
    parser.add_argument("--lmax", type=float, default=100.0)
    parser.add_argument("--cutoff", type=float, default=1e-3)
    parser.add_argument("--dis", type=float, default=5.0)
    parser.add_argument("--dis-type", type=str, default="linear", dest="dis_type")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--store-flow", action="store_true", dest="store_flow")
    args = parser.parse_args()

    print("=" * 70)
    print("  SMOKE TEST: flow_dyn_int_imb_fine_one（全手动欧拉）")
    print("=" * 70)
    print(f"    L={args.L}  dim={args.dim}  n={args.L**args.dim}")
    print(f"    qmax={args.qmax}  lmax={args.lmax}  cutoff={args.cutoff:.1e}")
    print(f"    dis={args.dis}  dis_type={args.dis_type}")
    print(f"    seed={args.seed}  store_flow={args.store_flow}")
    print()

    out = run_smoke(
        L=args.L,
        dim=args.dim,
        qmax=args.qmax,
        lmax=args.lmax,
        cutoff=args.cutoff,
        dis=args.dis,
        dis_type=args.dis_type,
        seed=args.seed,
        store_flow=args.store_flow,
    )

    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
