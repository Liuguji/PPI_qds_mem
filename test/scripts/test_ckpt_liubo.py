#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import argparse
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 环境变量必须在 JAX 首次 import 之前设置
# ─────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR  = REPO_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

import numpy as np
import jax.numpy as jnp
import models.models as models
from core.diag_routines.spinless_fermion import flow_static_int_ckpt_liubo


def make_dl_list(lmax: float, qmax: int) -> np.ndarray:
    return np.logspace(np.log10(0.001), np.log10(lmax), qmax, endpoint=True, base=10)


def run_one(L: int, dim: int, qmax: int, lmax: float, cutoff: float,
            method: str, dis: float, dis_type: str) -> dict:
    n = L ** dim
    J     = 1.0
    delta = 0.1
    x     = 0.0

    print(f"\n{'═'*62}")
    print(f"  L={L}  n={n}  dim={dim}  dis={dis}  U={delta}  J={J}")
    print(f"  qmax={qmax}  lmax={lmax}  cutoff={cutoff:.1e}  method={method}")
    print(f"{'═'*62}")

    ham = models.hamiltonian("spinless fermion", dis_type, intr=True)
    ham.build(n, dim, dis, J, x, delta=delta)

    dl_list = make_dl_list(lmax, qmax)

    t0 = time.perf_counter()
    result = flow_static_int_ckpt_liubo(
        n, ham, dl_list, qmax, cutoff,
        method=method, norm=False, Hflow=False, store_flow=False,
    )
    elapsed = time.perf_counter() - t0

    H0    = result["H0_diag"]
    liom2 = result["LIOM2"]
    lbits = result["LIOM Interactions"]

    offdiag_max = float(np.max(np.abs(H0 - np.diag(np.diag(H0)))))
    liom_trace  = float(np.trace(liom2))
    liom_norm   = float(np.linalg.norm(liom2))
    steps_used  = len(result["dl_list"])

    print(f"\n  [结果摘要]")
    print(f"    耗时              : {elapsed:.2f} s")
    print(f"    实际积分步数      : {steps_used}")
    print(f"    H0 对角元(排序)   : {np.sort(np.diag(H0))}")
    print(f"    off-diag 最大值   : {offdiag_max:.3e}  (cutoff={cutoff:.1e})")
    print(f"    LIOM2 迹          : {liom_trace:.6f}  (期望≈1.0)")
    print(f"    LIOM2 Frobenius   : {liom_norm:.6f}")
    print(f"    l-bit 衰减(q=1..{n-1}): {lbits}")

    passed = offdiag_max < cutoff
    print(f"    状态: {'✓ PASS' if passed else '✗ WARN'}")

    return {
        "L": L, "n": n, "elapsed": elapsed, "steps": steps_used,
        "offdiag_max": offdiag_max, "liom_trace": liom_trace, "pass": passed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="测试 flow_static_int_ckpt_liubo 在多个 L 下的运行情况",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python test/scripts/test_ckpt_liubo.py
  python test/scripts/test_ckpt_liubo.py --L 4 6
  python test/scripts/test_ckpt_liubo.py --L 4 --qmax 500 --lmax 75
        """,
    )
    parser.add_argument("--L",       type=int,   nargs="+", default=[4],    help="线性系统大小列表，n=L^dim（默认 4）")
    parser.add_argument("--dim",     type=int,   default=2,                  help="空间维度（默认 2）")
    parser.add_argument("--qmax",    type=int,   default=750,                help="最大流方程步数（默认 750）")
    parser.add_argument("--lmax",    type=float, default=75.0,               help="最大流时间（默认 75）")
    parser.add_argument("--cutoff",  type=float, default=1e-3,               help="非对角截断值（默认 1e-3）")
    parser.add_argument("--method",  type=str,   default="einsum",           help="张量收缩方法：einsum/tensordot/jit（默认 einsum）")
    parser.add_argument("--dis",     type=float, default=5.0,                help="无序强度（默认 5.0）")
    parser.add_argument("--dis-type",type=str,   default="random", dest="dis_type", help="无序类型（默认 random）")
    args = parser.parse_args()

    ckpt_step_env = os.environ.get("PYFLOW_CKPT_STEP", "(auto)")
    print("测试目标: flow_static_int_ckpt_liubo")
    print(f"L_LIST={args.L}  dim={args.dim}  qmax={args.qmax}  lmax={args.lmax}")
    print(f"cutoff={args.cutoff:.1e}  method={args.method}  dis={args.dis}  dis_type={args.dis_type}")
    print(f"PYFLOW_CKPT_STEP={ckpt_step_env}")

    all_results = []
    for L in args.L:
        try:
            r = run_one(
                L=L, dim=args.dim, qmax=args.qmax, lmax=args.lmax,
                cutoff=args.cutoff, method=args.method,
                dis=args.dis, dis_type=args.dis_type,
            )
            all_results.append(r)
        except Exception as e:
            import traceback
            print(f"\n[ERROR] L={L} 运行失败: {e}")
            traceback.print_exc()
            all_results.append({"L": L, "pass": False, "error": str(e)})

    print(f"\n{'═'*62}")
    print("  汇总")
    print(f"  {'L':>4}  {'n':>4}  {'耗时(s)':>9}  {'步数':>6}  {'off-diag':>10}  {'LIOM迹':>8}  状态")
    print(f"  {'-'*58}")
    for r in all_results:
        if "error" in r:
            print(f"  {r['L']:>4}  {'':>4}  {'ERROR':>9}  {r.get('error','')[:35]}")
        else:
            status = "PASS" if r["pass"] else "WARN"
            print(f"  {r['L']:>4}  {r['n']:>4}  {r['elapsed']:>9.2f}  {r['steps']:>6}  {r['offdiag_max']:>10.2e}  {r['liom_trace']:>8.4f}  {status}")
    print(f"{'═'*62}")


if __name__ == "__main__":
    main()
