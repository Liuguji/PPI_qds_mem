#!/usr/bin/env python3
"""
Simple runtime test for flow_test_cpu.

Usage:
    python test/scripts/test_ckpt_cpu.py
    python test/scripts/test_ckpt_cpu.py --L 3 4 --qmax 1000 2000
"""

from __future__ import annotations

import os
import sys
import argparse
import time
import json
from pathlib import Path

# Must be set before first JAX import
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

import numpy as np
import models.models as models
from core.diag_routines.spinless_fermion import flow_test_cpu


def make_dl_list(lmax: float, qmax: int) -> np.ndarray:
    return np.logspace(np.log10(0.001), np.log10(lmax), qmax, endpoint=True, base=10)


def format_array_for_print(arr: list | np.ndarray) -> str:
    """格式化数组为 [val1, val2, ...] 或 [val] 的形式"""
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()
    if isinstance(arr, list):
        if len(arr) == 1:
            return f"[{arr[0]}]"
        else:
            return f"{arr}"
    return f"[{arr}]"


def format_dl_list(dl_list: np.ndarray, num_show: int = 3) -> str:
    """格式化 dl_list 显示，开头和结尾各显示 num_show 个，中间用省略号"""
    n = len(dl_list)
    if n <= 2 * num_show:
        return f"{dl_list.tolist()}"
    
    head = dl_list[:num_show].tolist()
    tail = dl_list[-num_show:].tolist()
    head_str = ", ".join(f"{v:.6f}" for v in head)
    tail_str = ", ".join(f"{v:.6f}" for v in tail)
    return f"[{head_str}, ..., {tail_str}]"


def run_one(L: int, dim: int, qmax: int, lmax: float, cutoff: float,
            method: str, dis: float, dis_type: str, vary_params: dict) -> dict:
    n = L ** dim
    J = 1.0
    delta = 0.1
    x = 0.0

    # 打印非固定参数
    vary_str = ", ".join(f"{k}={v}" for k, v in vary_params.items())
    print(f"    [{vary_str}]", flush=True)

    ham = models.hamiltonian("spinless fermion", dis_type, intr=True)
    ham.build(n, dim, dis, J, x, delta=delta)

    dl_list = make_dl_list(lmax, qmax)

    t0 = time.perf_counter()
    result = flow_test_cpu(
        n, ham, dl_list, qmax, cutoff,
        method=method, norm=False, Hflow=False, store_flow=False,
    )
    elapsed = time.perf_counter() - t0

    # 打印 dl_list 的头尾部分
    dl_list_formatted = format_dl_list(result["dl_list"])
    print(f"    dl_list: {dl_list_formatted}", flush=True)

    # 提取关键结果
    steps_used = int(result["steps_evolved"])
    l_intercepted = float(result["l_intercepted"])
    offdiag_max = float(result["H2_offdiag_max"])
    ckpt_step = int(result["ckpt_step"])

    # 构建返回字典
    run_result = {
        "L": L,
        "n": n,
        "qmax": qmax,
        "lmax": lmax,
        "cutoff": cutoff,
        "method": method,
        "dis": dis,
        "dis_type": dis_type,
        "elapsed": elapsed,
        "steps_evolved": steps_used,
        "l_intercepted": l_intercepted,
        "H2_offdiag_max": offdiag_max,
        "ckpt_step": ckpt_step,
    }

    return run_result, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple runtime test for flow_test_cpu")
    parser.add_argument("--L", type=int, nargs="+", default=[4], help="linear system sizes")
    parser.add_argument("--dim", type=int, default=2, help="spatial dimension")
    parser.add_argument("--qmax", type=int, nargs="+", help="max flow steps")
    parser.add_argument("--lmax", type=float, nargs="+", help="max flow time")
    parser.add_argument("--cutoff", type=float, default=1e-3, help="off-diagonal cutoff")
    parser.add_argument("--method", type=str, default="einsum", help="contraction method")
    parser.add_argument("--dis", type=float, default=5.0, help="disorder strength")
    parser.add_argument("--dis-type", type=str, default="random", dest="dis_type", help="disorder type")
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory for JSON files")
    args = parser.parse_args()

    # 处理 qmax 和 lmax，允许单个值或多个值
    qmax_list = args.qmax if args.qmax else [2000]
    lmax_list = args.lmax if args.lmax else [150.0]

    # 打印所有设定参数
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"L           : {format_array_for_print(args.L)}")
    print(f"dim         : {args.dim}")
    print(f"qmax        : {format_array_for_print(qmax_list)}")
    print(f"lmax        : {format_array_for_print(lmax_list)}")
    print(f"cutoff      : {args.cutoff:.1e}")
    print(f"method      : {args.method}")
    print(f"dis         : {args.dis}")
    print(f"dis_type    : {args.dis_type}")
    if args.out_dir:
        print(f"output_dir  : {args.out_dir}")
    print("=" * 80 + "\n")

    # 确定哪些参数是变化的
    varying_keys = []
    if len(args.L) > 1:
        varying_keys.append("L")
    if len(qmax_list) > 1:
        varying_keys.append("qmax")
    if len(lmax_list) > 1:
        varying_keys.append("lmax")

    all_results = []
    all_raw_results = []

    # 创建输出目录
    if args.out_dir is None:
        args.out_dir = REPO_ROOT / "test_results"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for L in args.L:
        for qmax in qmax_list:
            for lmax in lmax_list:
                # 构建变化参数字典
                vary_params = {}
                if "L" in varying_keys:
                    vary_params["L"] = L
                if "qmax" in varying_keys:
                    vary_params["qmax"] = qmax
                if "lmax" in varying_keys:
                    vary_params["lmax"] = lmax

                try:
                    run_result, raw_result = run_one(
                        L=L,
                        dim=args.dim,
                        qmax=qmax,
                        lmax=lmax,
                        cutoff=args.cutoff,
                        method=args.method,
                        dis=args.dis,
                        dis_type=args.dis_type,
                        vary_params=vary_params,
                    )
                    all_results.append(run_result)

                    # 保存原始结果到 JSON 文件
                    json_filename = f"flow_result_L{L}_qmax{qmax}_lmax{lmax:g}.json"
                    json_path = args.out_dir / json_filename

                    # 转换 numpy 数组为列表以便 JSON 序列化
                    json_result = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                   for k, v in raw_result.items()}

                    with open(json_path, "w") as f:
                        json.dump(json_result, f, indent=2)
                    print(f"    Saved to {json_path}\n", flush=True)

                except Exception as exc:
                    import traceback
                    print(f"[ERROR] L={L}, qmax={qmax}, lmax={lmax} failed: {exc}")
                    traceback.print_exc()
                    all_results.append({
                        "L": L,
                        "qmax": qmax,
                        "lmax": lmax,
                        "error": str(exc),
                    })

    # 打印汇总表格
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(
        f"{'L':>4} {'qmax':>6} {'elapsed(s)':>12} {'steps':>7} {'l_int':>10} {'J0':>11} {'ckpt_step':>10}"
    )
    print("-" * 100)

    for r in all_results:
        if "error" in r:
            print(f"{r.get('L', ''):>4} {r.get('qmax', ''):>6} {'ERROR':>12} "
                  f"{'':>7} {'':>10} {'':>11} {'':>10}")
            continue
        
        print(
            f"{r['L']:>4} {r['qmax']:>6} {r['elapsed']:>12.3f} {r['steps_evolved']:>7} "
            f"{r['l_intercepted']:>10.6f} {r['H2_offdiag_max']:>11.3e} {r['ckpt_step']:>10}"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
