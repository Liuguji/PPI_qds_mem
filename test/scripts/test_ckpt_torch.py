#!/usr/bin/env python3
"""
对比测试 flow_static_int_ckpt_torch 与 flow_static_int_ckpt_liubo 的数值一致性。

用法:
    python test/scripts/test_ckpt_torch.py
    python test/scripts/test_ckpt_torch.py --L 4 --qmax 300 --lmax 50
    python test/scripts/test_ckpt_torch.py --torch-only   # 只跑 torch 版（跳过对比）
"""

from __future__ import annotations

import os
import sys
import argparse
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR  = REPO_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

import unicodedata
import numpy as np
import models.models as models
from core.diag_routines.spinless_fermion import (
    flow_static_int_ckpt_liubo,
    flow_static_int_ckpt_torch,
)

W = 80  # 分隔线宽度（终端列数）


def _dw(s: str) -> int:
    """字符串的终端显示宽度：中文等宽字符占 2 列，其余占 1 列。"""
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in s)


def _ljust_dw(s: str, width: int) -> str:
    """按终端显示宽度左对齐填充空格。"""
    return s + ' ' * max(width - _dw(s), 0)


def row(label: str, value: str, width: int = 24):
    print("  " + _ljust_dw(label, width) + " " + value)


def box_row(label: str, value: str, width: int = 18, border: str = "│"):
    content = "  " + _ljust_dw(label, width) + " " + value
    print(border + _ljust_dw(content, W - 2) + border)


def _print_result_block(label: str, elapsed: float, steps: int,
                        offdiag: float, cutoff: float, trace: float,
                        passed=None, lbits=None):
    print("┌" + "─" * (W - 2) + "┐")
    print("│" + _ljust_dw(f"  {label}", W - 2) + "│")
    print("├" + "─" * (W - 2) + "┤")
    box_row("耗时",            f"{elapsed:.2f} s")
    box_row("积分步数",        f"{steps}")
    box_row("off-diag 最大值", f"{offdiag:.3e}  (cutoff={cutoff:.1e})")
    box_row("LIOM2 迹",        f"{trace:.6f}  (期望 ≈ 1.0)")
    if lbits is not None:
        box_row("lbits", f"[{'  '.join(f'{v:.2f}' for v in lbits)}]")
    if passed is not None:
        print("├" + "─" * (W - 2) + "┤")
        box_row("判断", "✓ PASS" if passed else "✗ WARN")
    print("└" + "─" * (W - 2) + "┘")


def make_dl_list(lmax: float, qmax: int) -> np.ndarray:
    return np.logspace(np.log10(0.001), np.log10(lmax), qmax, endpoint=True, base=10)


def run_one(L: int, dim: int, qmax: int, lmax: float, cutoff: float,
            dis: float, dis_type: str, torch_only: bool) -> dict:
    n     = L ** dim
    J     = 1.0
    delta = 0.1
    x     = 0.0

    print("═" * (W + 10))
    print(f"  L={L}  n={n}  dim={dim}  dis={dis}  U={delta}  J={J}"
          f"  qmax={qmax}  lmax={lmax}  cutoff={cutoff:.1e}")
    print("═" * (W + 10))

    ham = models.hamiltonian("spinless fermion", dis_type, intr=True)
    ham.build(n, dim, dis, J, x, delta=delta)
    dl_list = make_dl_list(lmax, qmax)

    
    # ── torch 版 ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    res_torch = flow_static_int_ckpt_torch(
        n, ham, dl_list.copy(), qmax, cutoff, norm=False, Hflow=False,
    )
    t_torch = time.perf_counter() - t0

    H0_t      = res_torch["H0_diag"]
    liom2_t   = res_torch["LIOM2"]
    lbits_t   = res_torch["LIOM Interactions"]
    offdiag_t = float(np.max(np.abs(H0_t - np.diag(np.diag(H0_t)))))
    trace_t   = float(np.trace(liom2_t))

    _print_result_block("torch 版结果", t_torch, len(res_torch["dl_list"]),
                        offdiag_t, cutoff, trace_t, lbits=lbits_t)

    passed_torch = offdiag_t < cutoff * 10 and abs(trace_t - 1.0) < 0.5
    row("判断", f"{'✓ PASS' if passed_torch else '✗ WARN'}")

    if torch_only:
        return {"L": L, "n": n, "pass": passed_torch,
                "offdiag_torch": offdiag_t, "trace_torch": trace_t,
                "t_torch": t_torch}

    
    # ── JAX/liubo 版（对比基准）────────────────────────────────────────────────
    t0 = time.perf_counter()
    res_jax = flow_static_int_ckpt_liubo(
        n, ham, dl_list.copy(), qmax, cutoff, norm=False, Hflow=False,
    )
    t_jax = time.perf_counter() - t0

    H0_j      = res_jax["H0_diag"]
    liom2_j   = res_jax["LIOM2"]
    offdiag_j = float(np.max(np.abs(H0_j - np.diag(np.diag(H0_j)))))
    trace_j   = float(np.trace(liom2_j))

    _print_result_block("JAX (liubo) 版结果", t_jax, len(res_jax["dl_list"]),
                        offdiag_j, cutoff, trace_j)

    
    
    # ── 数值对比 ──────────────────────────────────────────────────────────────
    eig_err  = float(np.max(np.abs(np.sort(np.diag(H0_t)) - np.sort(np.diag(H0_j)))))
    liom_err = float(np.linalg.norm(liom2_t - liom2_j))

    numerically_close = eig_err < 5e-3 and liom_err < 5e-2
    row("数值一致性", f"{'✓ OK' if numerically_close else '✗ DIFF'}")

    return {
        "L": L, "n": n,
        "offdiag_torch": offdiag_t, "trace_torch": trace_t,
        "offdiag_jax":   offdiag_j, "trace_jax":   trace_j,
        "eig_err": eig_err, "liom_err": liom_err,
        "t_torch": t_torch, "t_jax": t_jax,
        "pass": passed_torch and numerically_close,
    }


def main():
    parser = argparse.ArgumentParser(
        description="对比 flow_static_int_ckpt_torch 与 flow_static_int_ckpt_liubo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python test/scripts/test_ckpt_torch.py
  python test/scripts/test_ckpt_torch.py --L 4 6
  python test/scripts/test_ckpt_torch.py --torch-only --L 4
        """,
    )
    parser.add_argument("--L",          type=int,   nargs="+", default=[4],   help="线性系统大小（默认 4）")
    parser.add_argument("--dim",        type=int,   default=2,                help="空间维度（默认 2）")
    parser.add_argument("--qmax",       type=int,   default=1000,             help="最大流方程步数（默认 1000）")
    parser.add_argument("--lmax",       type=float, default=100.0,            help="最大流时间（默认 100）")
    parser.add_argument("--cutoff",     type=float, default=1e-3,             help="非对角截断值（默认 1e-3）")
    parser.add_argument("--dis",        type=float, default=5.0,              help="无序强度（默认 5.0）")
    parser.add_argument("--dis-type",   type=str,   default="random",         dest="dis_type")
    parser.add_argument("--torch-only", action="store_true",                  help="只运行 torch 版，跳过 JAX 对比")
    args = parser.parse_args()

    print("╔" + "═" * (W - 2) + "╗")
    print("║" + _ljust_dw("  测试目标: flow_static_int_ckpt_torch 与 flow_static_int_ckpt_liubo", W - 2) + "║")
    print("╠" + "═" * (W - 2) + "╣")
    box_row("L列表",    str(args.L),                                                          border="║")
    box_row("dim",       str(args.dim),                                                        border="║")
    box_row("qmax",      str(args.qmax),                                                       border="║")
    box_row("lmax",      str(args.lmax),                                                       border="║")
    box_row("cutoff",    f"{args.cutoff:.1e}",                                                 border="║")
    box_row("dis",       str(args.dis),                                                        border="║")
    box_row("模式",      "torch-only" if args.torch_only else "torch(GPU)vsJAX(CPU) 对比",   border="║")
    box_row("CKPT_STEP", os.environ.get("PYFLOW_CKPT_STEP", "auto, min(20, sqrt(qmax))"),     border="║")
    print("╚" + "═" * (W - 2) + "╝")

    all_results = []
    for L in args.L:
        try:
            r = run_one(L=L, dim=args.dim, qmax=args.qmax, lmax=args.lmax,
                        cutoff=args.cutoff, dis=args.dis, dis_type=args.dis_type,
                        torch_only=args.torch_only)
            all_results.append(r)
        except Exception as e:
            import traceback
            print(f"\n  [ERROR] L={L} 运行失败: {e}")
            traceback.print_exc()
            all_results.append({"L": L, "pass": False, "error": str(e)})

    # ── 汇总表 ────────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print("  汇总")
    print(f"  {'─' * (W - 2)}")

    if args.torch_only:
        print(f"  {'L':>4}  {'n':>4}  {'off-diag':>10}  {'LIOM 迹':>9}  {'耗时(s)':>8}  状态")
        print(f"  {'─' * 52}")
        for r in all_results:
            if "error" in r:
                print(f"  {r['L']:>4}  {'':>4}  ERROR  {r.get('error','')[:38]}")
            else:
                status = "PASS" if r["pass"] else "WARN"
                print(f"  {r['L']:>4}  {r['n']:>4}  {r['offdiag_torch']:>10.2e}"
                      f"  {r['trace_torch']:>9.5f}  {r['t_torch']:>8.2f}  {status}")
    else:
        print(f"  {'L':>4}  {'n':>4}  {'eig_err':>9}  {'liom_err':>9}"
              f"  {'t_torch':>9}  {'t_jax':>9}  状态")
        print(f"  {'─' * 62}")
        for r in all_results:
            if "error" in r:
                print(f"  {r['L']:>4}  {'':>4}  ERROR  {r.get('error','')[:38]}")
            else:
                status = "PASS" if r["pass"] else "WARN/DIFF"
                print(f"  {r['L']:>4}  {r['n']:>4}  {r['eig_err']:>9.2e}"
                      f"  {r['liom_err']:>9.2e}"
                      f"  {r['t_torch']:>8.2f}s  {r['t_jax']:>8.2f}s  {status}")

    print(f"{'═' * W}")


if __name__ == "__main__":
    main()
