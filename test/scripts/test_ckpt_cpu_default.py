#!/usr/bin/env python3
"""
Simple runtime test for flow_test_cpu using default CPU threading settings.

Usage:
    python test/scripts/test_ckpt_cpu_default.py
    python test/scripts/test_ckpt_cpu_default.py --L 3 4 --qmax 1000 2000
"""

from __future__ import annotations

import os
import sys
import argparse
import time
import json
from itertools import product
from pathlib import Path

# Must be set before first JAX import
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

import numpy as np
import models.models as models
from core.diag_routines.spinless_fermion import flow_test_cpu


def format_array_for_print(arr: list | np.ndarray) -> str:
    """Format array as [val1, val2, ...] or [val]."""
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()
    if isinstance(arr, list):
        if len(arr) == 1:
            return f"[{arr[0]}]"
        else:
            return f"{arr}"
    return f"[{arr}]"


def format_dl_list(dl_list: np.ndarray, num_show: int = 3) -> str:
    """Format dl_list with head and tail values."""
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


def run_one(
    L: int,
    dim: int,
    qmax: int,
    lmax: float,
    cutoff: float,
    method: str,
    dis: float,
    dis_type: str,
    vary_params: dict,
) -> tuple[dict, dict]:
    n = L ** dim
    J = 1.0
    delta = 0.1
    x = 0.0

    # Print varying parameters for this run.
    vary_str = ", ".join(f"{k}={v}" for k, v in vary_params.items())
    print(f"    [{vary_str}]", flush=True)

    ham = models.hamiltonian("spinless fermion", dis_type, intr=True)
    ham.build(n, dim, dis, J, x, delta=delta)

    dl_list = np.logspace(np.log10(0.001), np.log10(lmax), qmax, endpoint=True, base=10)

    t0 = time.perf_counter()
    result = flow_test_cpu(
        n,
        ham,
        dl_list,
        qmax,
        cutoff,
        method=method,
        norm=False,
        Hflow=False,
        store_flow=False,
    )
    elapsed = time.perf_counter() - t0

    dl_list_formatted = format_dl_list(result["dl_list"])
    print(f"    dl_list: {dl_list_formatted}", flush=True)

    steps_used = int(result["steps_evolved"])
    l_intercepted = float(result["l_intercepted"])
    offdiag_max = float(result["H2_offdiag_max"])
    ckpt_step = int(result["ckpt_step"])

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
    parser.add_argument(
        "--cutoff",
        type=float,
        nargs="+",
        default=[1e-3],
        help="off-diagonal cutoff (one or more values)",
    )
    parser.add_argument("--method", type=str, default="einsum", help="contraction method")
    parser.add_argument("--dis", type=float, default=5.0, help="disorder strength")
    parser.add_argument(
        "--dis-type",
        type=str,
        nargs="+",
        default=["linear"],
        dest="dis_type",
        help="disorder type (one or more values)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory for JSON files")
    args = parser.parse_args()

    qmax_list = args.qmax or [1000]
    lmax_list = args.lmax or [100.0]
    cutoff_list = args.cutoff or [1e-3]
    dis_type_list = args.dis_type or ["random"]

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

    varying_keys = [
        key
        for key, values in (
            ("L", args.L),
            ("qmax", qmax_list),
            ("lmax", lmax_list),
            ("dis_type", dis_type_list),
            ("cutoff", cutoff_list),
        )
        if len(values) > 1
    ]

    all_results = []

    if args.out_dir is None:
        args.out_dir = REPO_ROOT / "test_results"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for L, qmax, lmax, dis_type, cutoff in product(args.L, qmax_list, lmax_list, dis_type_list, cutoff_list):
        run_params = {
            "L": L,
            "qmax": qmax,
            "lmax": lmax,
            "dis_type": dis_type,
            "cutoff": cutoff,
        }
        vary_params = {k: run_params[k] for k in varying_keys}

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

            json_filename = (
                f"flow_result_"
                f"L{safe_token(L)}_"
                f"lmax{safe_token(f'{lmax:g}')}_"
                f"qmax{safe_token(qmax)}_"
                f"dis-type{safe_token(dis_type)}_"
                f"cutoff{safe_token(f'{cutoff:g}')}.json"
            )
            json_path = args.out_dir / json_filename

            json_result = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in raw_result.items()
            }

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
            all_results.append(
                {
                    "L": L,
                    "qmax": qmax,
                    "lmax": lmax,
                    "dis_type": dis_type,
                    "cutoff": cutoff,
                    "error": str(exc),
                }
            )

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
