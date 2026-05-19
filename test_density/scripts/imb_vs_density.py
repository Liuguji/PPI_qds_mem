#!/usr/bin/env python3
"""
对比测试：flow_dyn_density 分支 0（默认版） vs 其他优化分支。

用 switch_num=0 的结果作为 baseline，与指定优化分支的结果对比：
  1. 对角化结果：H0_diag、Hint
  2. l-bit 相互作用系数 LIOM Interactions
  3. 动力学结果：density（per-site 数密度）
  4. 执行时间

用法：
    python test_density/scripts/imb_vs_density.py [--L 4] [--dim 1] [--qmax 500] [--switch-num 1]
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
flow_dyn_density = _mod.flow_dyn_density


def make_dl_list(lmax: float, qmax: int) -> np.ndarray:
    return np.logspace(np.log10(0.001), np.log10(lmax), qmax, endpoint=True, base=10)


def density_to_imbalance(density: np.ndarray) -> np.ndarray:
    """从 per-site density (n, nt) 重建 imbalance (nt,)。
    公式：I(t) = 2 * Σ_i (-1)^i density[i,t] / n
    """
    n = density.shape[0]
    sign = np.array([(-1) ** i for i in range(n)], dtype=density.dtype)
    return 2.0 * np.sum(sign[:, None] * density, axis=0) / n


def set_switch_env(switch_num: int) -> None:
    """设置环境变量控制 flow_dyn_density 的分支选择。"""
    os.environ["pipeline_switch"]   = str((switch_num // 1000) % 10)
    os.environ["parallel_switch"]   = str((switch_num // 100) % 10)
    os.environ["compress_switch"]   = str((switch_num // 10) % 10)
    os.environ["checkpoint_switch"] = str(switch_num % 10)
    os.environ["compress_mode"]     = "0"


def run_comparison(
    L: int = 4,
    dim: int = 1,
    qmax: int = 500,
    lmax: float = 100.0,
    cutoff: float = 1e-3,
    method: str = "tensordot",
    dis: float = 5.0,
    dis_type: str = "random",
    seed: int = 42,
    test_switch: int = 1,
) -> dict:
    n = L ** dim
    tlist = [0.01 * i for i in range(31)]
    J = 1.0
    delta = 0.1
    x = 0.0

    # ── 可复现的哈密顿量 ──
    os.environ["PYFLOW_SEED"] = str(seed)
    ham = models.hamiltonian("spinless fermion", dis_type, intr=True)
    ham.build(n, dim, dis, J, x, delta=delta)

    dl_list = make_dl_list(lmax, qmax)

    # ── 两个函数都接收 num / num_int，但内部都自行覆盖 ──
    num = np.zeros((n, n), dtype=np.float64)
    num[n // 2, n // 2] = 1.0
    num_int = np.zeros((n, n, n, n), dtype=np.float64)

    # ==================================================================
    # 1. baseline：flow_dyn_density 分支 0（默认版本）
    # ==================================================================
    set_switch_env(0)
    print("-" * 70)
    print("  [1/2] flow_dyn_density (switch_num=0, baseline)")
    print("-" * 70)
    t0 = time.perf_counter()
    res_base = flow_dyn_density(
        n, ham, num, num_int, dl_list, qmax, cutoff,
        tlist=tlist, state=None, method=method,
    )
    t_base = time.perf_counter() - t0
    print(f"        Time: {t_base:.3f}s")

    # ==================================================================
    # 2. 测试分支：flow_dyn_density switch_num = test_switch
    # ==================================================================
    set_switch_env(test_switch)
    print("-" * 70)
    print(f"  [2/2] flow_dyn_density (switch_num={test_switch})")
    print("-" * 70)
    t0 = time.perf_counter()
    res_test = flow_dyn_density(
        n, ham, num, num_int, dl_list, qmax, cutoff,
        tlist=tlist, state=None, method=method,
    )
    t_test = time.perf_counter() - t0
    print(f"        Time: {t_test:.3f}s")

    # ==================================================================
    # 3. 提取公共物理量
    # ==================================================================
    # H0_diag
    H0_base = np.array(res_base["H0_diag"])
    H0_test = np.array(res_test["H0_diag"])

    # Hint（分支 0 返回 Hint_diag，其他分支返回 Hint）
    Hint_base = np.array(res_base.get("Hint_diag", res_base.get("Hint")))
    Hint_test = np.array(res_test.get("Hint", res_test.get("Hint_diag")))

    # LIOM Interactions
    if "LIOM Interactions" in res_base:
        lbits_base = np.array(res_base["LIOM Interactions"])
    else:
        lbits_base = None
    if "LIOM Interactions" in res_test:
        lbits_test = np.array(res_test["LIOM Interactions"])
    else:
        lbits_test = None

    # LIOM2 / LIOM4（代替原来的 density）
    liom2_base = np.array(res_base["LIOM2"])  # (n, n, n)
    liom2_test = np.array(res_test.get("LIOM2", res_test.get("LIOM2")))
    liom4_base = np.array(res_base["LIOM4"])  # (n, n, n, n, n)
    liom4_test = np.array(res_test.get("LIOM4", res_test.get("LIOM4")))

    # 步数
    steps_base = int(res_base.get("steps_evolved", -1))
    steps_test = int(res_test.get("steps_evolved", -1))

    # ==================================================================
    # 4. 对比指标
    # ==================================================================
    def _rel_diff(a, b):
        denom = max(np.max(np.abs(a)), 1e-300)
        return float(np.max(np.abs(a - b)) / denom)

    def _max_abs(a, b):
        return float(np.max(np.abs(a - b)))

    comp = {
        "H0_diag": {
            "max_abs_diff": _max_abs(H0_base, H0_test),
            "rel_diff": _rel_diff(H0_base, H0_test),
        },
        "Hint": {
            "max_abs_diff": _max_abs(Hint_base, Hint_test),
            "rel_diff": _rel_diff(Hint_base, Hint_test),
        },
        "LIOM2": {
            "max_abs_diff": _max_abs(liom2_base, liom2_test),
            "rel_diff": _rel_diff(liom2_base, liom2_test),
        },
        "LIOM4": {
            "max_abs_diff": _max_abs(liom4_base, liom4_test),
            "rel_diff": _rel_diff(liom4_base, liom4_test),
        },
        "timing": {
            "baseline_s": t_base,
            "test_s": t_test,
            "test_over_baseline": t_test / t_base if t_base > 0 else float("inf"),
        },
        "steps": {
            "baseline": steps_base,
            "test": steps_test,
        },
    }

    if lbits_base is not None and lbits_test is not None:
        comp["lbits"] = {
            "max_abs_diff": _max_abs(lbits_base, lbits_test),
            "rel_diff": _rel_diff(lbits_base, lbits_test),
        }

    # ==================================================================
    # 5. 输出汇总
    # ==================================================================
    print()
    print("=" * 70)
    print(f"  COMPARISON: switch_num=0  vs  switch_num={test_switch}")
    print("=" * 70)

    def _print_metric(label: str, max_abs: float, rel: float):
        if rel < 1e-15:
            rel_str = "<1e-15"
        else:
            rel_str = f"{rel:.3e}"
        status = "✓" if max_abs < 1e-10 else "Δ"
        print(f"    {label:22s}  |Δ|max={max_abs:>10.3e}  rel={rel_str:>10}  [{status}]")

    _print_metric("H0_diag", comp["H0_diag"]["max_abs_diff"], comp["H0_diag"]["rel_diff"])
    _print_metric("Hint", comp["Hint"]["max_abs_diff"], comp["Hint"]["rel_diff"])
    if "lbits" in comp:
        _print_metric("lbits", comp["lbits"]["max_abs_diff"], comp["lbits"]["rel_diff"])
    _print_metric("LIOM2", comp["LIOM2"]["max_abs_diff"], comp["LIOM2"]["rel_diff"])
    _print_metric("LIOM4", comp["LIOM4"]["max_abs_diff"], comp["LIOM4"]["rel_diff"])

    print(f"    {'Timing':22s}  base={t_base:.3f}s  test={t_test:.3f}s  "
          f"ratio={comp['timing']['test_over_baseline']:.2f}x")
    print(f"    {'Steps':22s}  base={steps_base}  test={steps_test}")

    # 最终判定
    tol = 1e-10
    check_keys = ["H0_diag", "Hint", "LIOM2", "LIOM4"]
    all_pass = all(comp[k]["max_abs_diff"] < tol for k in check_keys)
    print()
    print(f"    >>> {'ALL MATCH' if all_pass else 'SOME DEVIATIONS'} "
          f"(tolerance={tol:.0e}) <<<")

    return {
        "baseline_result": res_base,
        "test_result": res_test,
        "comparison": comp,
    }


def main():
    parser = argparse.ArgumentParser(
        description="对比测试：flow_dyn_density 分支 0 vs 其他分支"
    )
    parser.add_argument("--L", type=int, default=4, help="线性尺寸")
    parser.add_argument("--dim", type=int, default=2, help="空间维度")
    parser.add_argument("--qmax", type=int, default=500, help="最大流步数")
    parser.add_argument("--lmax", type=float, default=100.0, help="最大流时间")
    parser.add_argument("--cutoff", type=float, default=1e-3, help="非对角截断")
    parser.add_argument("--method", type=str, default="tensordot", help="缩并方法")
    parser.add_argument("--dis", type=float, default=5.0, help="无序强度")
    parser.add_argument("--dis-type", type=str, default="linear", dest="dis_type",
                        help="无序类型")
    parser.add_argument("--switch-num", type=int, default=1, dest="test_switch",
                        help="待测试的 density 分支号")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    print("=" * 70)
    print("  COMPARISON: flow_dyn_density switch_num=0  vs  others")
    print("=" * 70)
    print(f"    L={args.L}  dim={args.dim}  n={args.L**args.dim}")
    print(f"    qmax={args.qmax}  lmax={args.lmax}  cutoff={args.cutoff:.1e}")
    print(f"    method={args.method}  dis={args.dis}  dis_type={args.dis_type}")
    print(f"    test_switch={args.test_switch}  seed={args.seed}")
    print()

    _ = run_comparison(
        L=args.L,
        dim=args.dim,
        qmax=args.qmax,
        lmax=args.lmax,
        cutoff=args.cutoff,
        method=args.method,
        dis=args.dis,
        dis_type=args.dis_type,
        seed=args.seed,
        test_switch=args.test_switch,
    )


if __name__ == "__main__":
    main()
