#!/usr/bin/env python3
"""
Simple runtime test for flow_test_cpu.

Usage:
    python test/scripts/test_ckpt_cpu.py
    python test/scripts/test_ckpt_cpu.py --L 3 4 --qmax 1000 2000
"""

from __future__ import annotations
from psutil import cpu_count
import os
import sys
import argparse
import time
import json
from pathlib import Path

# Must be set before first JAX import
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

def _resolve_cpu_threads() -> int:
    """Resolve target CPU thread count: physical cores first, then logical cores."""
    env_v = os.environ.get("PYFLOW_CPU_THREADS", "auto").strip().lower()
    if env_v not in ("", "auto"):
        try:
            return max(1, int(env_v))
        except ValueError:
            pass

    try:
        n_physical = cpu_count(logical=False)
        if n_physical and int(n_physical) > 0:
            return int(n_physical)
    except Exception:
        pass

    try:
        n_logical = cpu_count(logical=True)
        if n_logical and int(n_logical) > 0:
            return int(n_logical)
    except Exception:
        pass

    return int(os.cpu_count() or 1)


def _configure_cpu_parallelism() -> int:
    threads = _resolve_cpu_threads()

    # Force this test harness to use many-core CPU threading by default.
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)

    # Ensure XLA CPU backend uses multi-thread Eigen and requested intra-op threads.
    xla_flags = os.environ.get("XLA_FLAGS", "")
    wanted = [
        "--xla_cpu_multi_thread_eigen=true",
        f"intra_op_parallelism_threads={threads}",
    ]
    for flag in wanted:
        if flag not in xla_flags:
            xla_flags = f"{xla_flags} {flag}".strip()
    os.environ["XLA_FLAGS"] = xla_flags

    return threads


CPU_THREADS = _configure_cpu_parallelism()


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


def safe_token(v) -> str:
    """Convert value to a filename-safe token."""
    s = str(v)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


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
    parser.add_argument("--cutoff", type=float, nargs="+", default=[1e-3], help="off-diagonal cutoff (one or more values)")
    parser.add_argument("--method", type=str, default="einsum", help="contraction method")
    parser.add_argument("--dis", type=float, default=5.0, help="disorder strength")
    parser.add_argument("--dis-type", type=str, nargs="+", default=["random"], dest="dis_type", help="disorder type (one or more values)")
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory for JSON files")
    args = parser.parse_args()

    # 处理 qmax 和 lmax，允许单个值或多个值
    qmax_list = args.qmax if args.qmax else [1000]
    lmax_list = args.lmax if args.lmax else [100.0]
    cutoff_list = args.cutoff if args.cutoff else [1e-3]
    dis_type_list = args.dis_type if args.dis_type else ["random"]

    # 打印所有设定参数
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"L           : {format_array_for_print(args.L)}")
    print(f"dim         : {args.dim}")
    print(f"qmax        : {format_array_for_print(qmax_list)}")
    print(f"lmax        : {format_array_for_print(lmax_list)}")
    print(f"cutoff      : {format_array_for_print(cutoff_list)}")
    print(f"method      : {args.method}")
    print(f"dis         : {args.dis}")
    print(f"dis_type    : {format_array_for_print(dis_type_list)}")
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
    if len(dis_type_list) > 1:
        varying_keys.append("dis_type")
    if len(cutoff_list) > 1:
        varying_keys.append("cutoff")

    all_results = []
    all_raw_results = []

    # 创建输出目录
    if args.out_dir is None:
        args.out_dir = REPO_ROOT / "test_results"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for L in args.L:
        for qmax in qmax_list:
            for lmax in lmax_list:
                for dis_type in dis_type_list:
                    for cutoff in cutoff_list:
                        # 构建变化参数字典
                        vary_params = {}
                        if "L" in varying_keys:
                            vary_params["L"] = L
                        if "qmax" in varying_keys:
                            vary_params["qmax"] = qmax
                        if "lmax" in varying_keys:
                            vary_params["lmax"] = lmax
                        if "dis_type" in varying_keys:
                            vary_params["dis_type"] = dis_type
                        if "cutoff" in varying_keys:
                            vary_params["cutoff"] = cutoff

                        try:
                            run_result, raw_result = run_one(
                                L=L,
                                dim=args.dim,
                                qmax=qmax,
                                lmax=lmax,
                                cutoff=cutoff,
                                method=args.method,
                                dis=args.dis,
                                dis_type=dis_type,
                                vary_params=vary_params,
                            )
                            all_results.append(run_result)

                            # 保存原始结果到 JSON 文件，参数顺序：L, lmax, qmax, dis-type, cutoff
                            json_filename = (
                                f"flow_result_"
                                f"L{safe_token(L)}_"
                                f"lmax{safe_token(f'{lmax:g}')}_"
                                f"qmax{safe_token(qmax)}_"
                                f"dis-type{safe_token(dis_type)}_"
                                f"cutoff{safe_token(f'{cutoff:g}')}.json"
                            )
                            json_path = args.out_dir / json_filename

                            # 转换 numpy 数组为列表以便 JSON 序列化
                            json_result = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                           for k, v in raw_result.items()}

                            with open(json_path, "w") as f:
                                json.dump(json_result, f, indent=2)
                            print(f"    Saved to {json_path}\n", flush=True)

                        except Exception as exc:
                            import traceback
                            print(
                                f"[ERROR] L={L}, lmax={lmax}, qmax={qmax}, "
                                f"dis_type={dis_type}, cutoff={cutoff} failed: {exc}"
                            )
                            traceback.print_exc()
                            all_results.append({
                                "L": L,
                                "qmax": qmax,
                                "lmax": lmax,
                                "dis_type": dis_type,
                                "cutoff": cutoff,
                                "error": str(exc),
                            })

    # 打印汇总表格
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(
        f"{'L':>4} {'lmax':>8} {'qmax':>6} {'dis_type':>10} {'cutoff':>10} {'total(s)':>10} {'l_end':>12} {'steps_end':>10} {'J0':>11} {'group_len':>10}"
    )
    print("-" * 100)

    for r in all_results:
        if "error" in r:
            print(
                f"{r.get('L', ''):>4} {r.get('lmax', ''):>8} {r.get('qmax', ''):>6} {str(r.get('dis_type', '')):>10} "
                f"{str(r.get('cutoff', '')):>10} {'ERROR':>10} {'':>12} {'':>10} {'':>11} {'':>10}"
            )
            continue
        
        print(
            f"{r['L']:>4} {r['lmax']:>8.2f} {r['qmax']:>6} {r['dis_type']:>10} {r['cutoff']:>10.1e} {r['elapsed']:>10.3f} "
            f"{r['l_intercepted']:>12.6f} {r['steps_evolved']:>10} {r['H2_offdiag_max']:>11.3e} {r['ckpt_step']:>10}"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
