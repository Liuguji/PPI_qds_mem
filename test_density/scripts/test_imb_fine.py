#!/usr/bin/env python3
"""
冒烟测试：flow_dyn_int_imb_fine 能否成功跑通。

仅验证函数可执行、返回结构合理（关键字段存在、形状正确、无 NaN/Inf）。
不做与其他实现的数值对比。

用法：
    # 基础用法：L=4, dim=1, qmax=300
    python test_density/scripts/test_imb_fine.py

    # 较大系统
    python test_density/scripts/test_imb_fine.py --L 4 --dim 2 --qmax 500

    # 指定无序与缩并方法
    python test_density/scripts/test_imb_fine.py --L 4 --dim 1 --qmax 500 --dis 3.0 --method tensordot

    # 保留 flow 轨迹
    python test_density/scripts/test_imb_fine.py --L 4 --dim 1 --qmax 300 --store-flow
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import json
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
flow_dyn_int_imb_fine = _mod.flow_dyn_int_imb_fine


def make_dl_list(lmax: float, qmax: int) -> np.ndarray:
    return np.logspace(np.log10(0.001), np.log10(lmax), qmax, endpoint=True, base=10)


def _check_array(name: str, arr, expected_shape=None):
    """打印一行字段诊断；返回 True 表示通过。"""
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
    L: int = 4,
    dim: int = 1,
    qmax: int = 300,
    lmax: float = 100.0,
    cutoff: float = 1e-3,
    method: str = "tensordot",
    dis: float = 5.0,
    dis_type: str = "linear",
    seed: int = 42,
    store_flow: bool = False,
) -> dict:
    n = L ** dim
    # tlist = [0.1 * i for i in range(31)]
    tlist = [0, 10.0]
    nt = len(tlist)
    J = 1.0
    delta = 0.1
    x = 0.0

    # ── 可复现的哈密顿量 ──
    os.environ["PYFLOW_SEED"] = str(seed)
    ham = models.hamiltonian("spinless fermion", dis_type, intr=True)
    ham.build(n, dim, dis, J, x, delta=delta)

    dl_list = make_dl_list(lmax, qmax)

    # imb_fine 接口与 imb 一致：num/num_int 为占位（内部不使用为初值）
    num = np.zeros((n, n), dtype=np.float64)
    num[n // 2, n // 2] = 1.0
    num_int = np.zeros((n, n, n, n), dtype=np.float64)

    print("-" * 70)
    print("  flow_dyn_int_imb_fine — smoke test")
    print("-" * 70)
    t0 = time.perf_counter()
    res = flow_dyn_int_imb_fine(
        n, ham, num, num_int, dl_list, qmax, cutoff,
        tlist=tlist, method=method, store_flow=store_flow,
    )
    elapsed = time.perf_counter() - t0
    print(f"        Time: {elapsed:.3f}s")
    print()

    # ==================================================================
    # 字段检查
    # ==================================================================
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

    # ==================================================================
    # imbalance 数值预览，按照0，1/4，1/2，3/4，末尾 的五个时间点打印
    # ==================================================================
    imb = np.asarray(res.get("Imbalance"))
    if imb is not None and imb.size:
        print()
        print("  Imbalance(t) preview:")
        preview_idx = [0, nt - 1]
        for i in preview_idx:
            print(f"      t[{i:2d}]={tlist[i]:.3f}   Imbalance = {float(imb[i]):+.6e}")

    # ==================================================================
    # 总判定
    # ==================================================================
    print()
    print("=" * 70)
    print(f"  >>> {'SMOKE TEST PASSED' if all_pass else 'SMOKE TEST FAILED'} <<<")
    print("=" * 70)

    # ==================================================================
    # 摘要存盘（不存原数组，只存指标）
    # ==================================================================
    # out_dir = REPO_ROOT / "test_results"
    # out_dir.mkdir(parents=True, exist_ok=True)
    # safe_dis = "".join(c if c.isalnum() else "_" for c in dis_type)
    # out_file = (
    #     out_dir
    #     / f"imbfine_smoke_L{L}_dim{dim}_qmax{qmax}_cutoff{cutoff:.0e}_{safe_dis}.json"
    # )
    # summary = {
    #     "config": {
    #         "L": L, "dim": dim, "n": n,
    #         "qmax": qmax, "lmax": lmax, "cutoff": cutoff,
    #         "method": method, "dis": dis, "dis_type": dis_type,
    #         "seed": seed, "store_flow": store_flow,
    #     },
    #     "timing_s": elapsed,
    #     "passed": all_pass,
    #     "imbalance_first": float(imb[0]) if imb is not None and imb.size else None,
    #     "imbalance_last":  float(imb[-1]) if imb is not None and imb.size else None,
    #     "imbalance_max_abs": float(np.max(np.abs(imb))) if imb is not None and imb.size else None,
    # }
    # with open(out_file, "w") as f:
    #     json.dump(summary, f, indent=2)
    # print(f"    Summary saved to: {out_file}")

    return {"result": res, "passed": all_pass, "elapsed": elapsed}


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for flow_dyn_int_imb_fine"
    )
    parser.add_argument("--L", type=int, default=2, help="线性尺寸")
    parser.add_argument("--dim", type=int, default=2, help="空间维度")
    parser.add_argument("--qmax", type=int, default=1000, help="最大流步数")
    parser.add_argument("--lmax", type=float, default=100.0, help="最大流时间")
    parser.add_argument("--cutoff", type=float, default=1e-3, help="非对角截断")
    parser.add_argument("--method", type=str, default="tensordot", help="缩并方法")
    parser.add_argument("--dis", type=float, default=5.0, help="无序强度")
    parser.add_argument("--dis-type", type=str, default="linear", dest="dis_type",
                        help="无序类型")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--store-flow", action="store_true", dest="store_flow",
                        help="保留 flow 轨迹（flow2/flow4/dl_list）")
    args = parser.parse_args()

    print("=" * 70)
    print("  SMOKE TEST: flow_dyn_int_imb_fine")
    print("=" * 70)
    print(f"    L={args.L}  dim={args.dim}  n={args.L**args.dim}")
    print(f"    qmax={args.qmax}  lmax={args.lmax}  cutoff={args.cutoff:.1e}")
    print(f"    method={args.method}  dis={args.dis}  dis_type={args.dis_type}")
    print(f"    seed={args.seed}  store_flow={args.store_flow}")
    print()

    out = run_smoke(
        L=args.L,
        dim=args.dim,
        qmax=args.qmax,
        lmax=args.lmax,
        cutoff=args.cutoff,
        method=args.method,
        dis=args.dis,
        dis_type=args.dis_type,
        seed=args.seed,
        store_flow=args.store_flow,
    )

    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
