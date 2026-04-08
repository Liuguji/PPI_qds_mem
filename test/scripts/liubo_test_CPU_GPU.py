#!/usr/bin/env python3
"""
Compare CPU(JAX) and GPU(Torch) flow runs with identical initial parameters.

Usage:
    python test/scripts/liubo_test_CPU_GPU.py
    python test/scripts/liubo_test_CPU_GPU.py --L 4 --qmax 1000 --lmax 100
    python test/scripts/liubo_test_CPU_GPU.py --L 3 4 --qmax 800 1200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from psutil import cpu_count

# Must be set before first JAX import.
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
# Skip current parameter instance if GPU flow hits max steps without convergence.
os.environ.setdefault("PYFLOW_GPU_SKIP_UNCONVERGED", "1")


def _resolve_cpu_threads() -> int:
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
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)

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
from core.diag_routines.spinless_fermion import (
    flow_static_int_ckpt_liubo,
    flow_static_int_ckpt_torch,
)


def make_dl_list(lmax: float, qmax: int) -> np.ndarray:
    return np.logspace(np.log10(0.001), np.log10(lmax), qmax, endpoint=True, base=10)


def safe_token(v) -> str:
    s = str(v)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def estimate_full_flow_mem_gb(n: int, num_points: int, bytes_per_elem: int) -> float:
    """Estimate memory for storing full H2+Hint trajectory on CPU."""
    total_bytes = num_points * (n**2 + n**4) * bytes_per_elem
    return float(total_bytes / 1e9)


def run_one(L: int, dim: int, qmax: int, lmax: float, cutoff: float,
            method: str, dis: float, dis_type: str) -> dict | None:
    n = L ** dim
    J = 1.0
    delta = 0.1
    x = 0.0

    ham = models.hamiltonian("spinless fermion", dis_type, intr=True)
    ham.build(n, dim, dis, J, x, delta=delta)

    dl_list = make_dl_list(lmax, qmax)

    # GPU/Torch run
    t1 = time.perf_counter()
    result_gpu = flow_static_int_ckpt_torch(
        n, ham, dl_list.copy(), qmax, cutoff,
        method=method, norm=False, Hflow=False, store_flow=False,
    )
    elapsed_gpu = time.perf_counter() - t1

    if bool(result_gpu.get("_skip_instance", False)):
        print(
            "      [SKIP] GPU forward diagonalization did not converge within max steps; skip this parameter set.",
            flush=True,
        )
        return None

    # CPU/JAX run
    t0 = time.perf_counter()
    result_cpu = flow_static_int_ckpt_liubo(
        n, ham, dl_list.copy(), qmax, cutoff,
        method=method, norm=False, Hflow=False, store_flow=False,
    )
    elapsed_cpu = time.perf_counter() - t0

    cpu_points = int(len(result_cpu["dl_list"]))
    gpu_points = int(len(result_gpu["dl_list"]))
    cpu_steps = max(cpu_points - 1, 0)
    gpu_steps = max(gpu_points - 1, 0)

    # H2 and Hint in CPU flow share the same dtype in this implementation.
    cpu_bytes_per_elem = int(np.asarray(result_cpu["H0_diag"]).dtype.itemsize)
    cpu_full_flow_mem_gb = estimate_full_flow_mem_gb(n, cpu_points, cpu_bytes_per_elem)

    summary = {
        "L": L,
        "n": n,
        "dim": dim,
        "qmax": qmax,
        "lmax": lmax,
        "cutoff": cutoff,
        "method": method,
        "dis": dis,
        "dis_type": dis_type,
        "cpu_elapsed_s": elapsed_cpu,
        "gpu_elapsed_s": elapsed_gpu,
        "speedup_gpu_vs_cpu": (elapsed_cpu / elapsed_gpu) if elapsed_gpu > 0 else np.nan,
        "cpu_steps_evolved": cpu_steps,
        "gpu_steps_evolved": gpu_steps,
        "cpu_full_flow_mem_gb": cpu_full_flow_mem_gb,
    }

    print(
        "      CPU: {:.3f}s | GPU: {:.3f}s | speedup: {:.2f}x | CPU full-flow mem~{:.3f} GB".format(
            summary["cpu_elapsed_s"],
            summary["gpu_elapsed_s"],
            summary["speedup_gpu_vs_cpu"],
            summary["cpu_full_flow_mem_gb"],
        ),
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run same-parameter CPU/JAX vs GPU/Torch flow tests")
    parser.add_argument("--L", type=int, nargs="+", default=[4], help="linear system sizes")
    parser.add_argument("--dim", type=int, default=2, help="spatial dimension")
    parser.add_argument("--qmax", type=int, nargs="+", default=[1000], help="max flow steps")
    parser.add_argument("--lmax", type=float, nargs="+", default=[100.0], help="max flow time")
    parser.add_argument("--cutoff", type=float, nargs="+", default=[1e-3], help="off-diagonal cutoff")
    parser.add_argument("--method", type=str, default="einsum", help="contraction method")
    parser.add_argument("--dis", type=float, default=5.0, help="disorder strength")
    parser.add_argument("--dis-type", type=str, nargs="+", default=["random"], dest="dis_type", help="disorder type")
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory for json files")
    args = parser.parse_args()

    print("=" * 100)
    print("CPU(JAX) vs GPU(Torch) FLOW TEST")
    print("=" * 100)
    print(f"L           : {args.L}")
    print(f"dim         : {args.dim}")
    print(f"qmax        : {args.qmax}")
    print(f"lmax        : {args.lmax}")
    print(f"cutoff      : {args.cutoff}")
    print(f"method      : {args.method}")
    print(f"dis         : {args.dis}")
    print(f"dis_type    : {args.dis_type}")
    print(f"cpu_threads : {CPU_THREADS}")
    print(f"gpu_skip_unconverged : {os.environ.get('PYFLOW_GPU_SKIP_UNCONVERGED', '0')} (1=skip, 0=keep)")
    print("=" * 100)

    out_dir = args.out_dir if args.out_dir is not None else (REPO_ROOT / "liubo_test_CPU_GPU_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    ts = time.strftime("%Y%m%d_%H%M%S")

    for L in args.L:
        for qmax in args.qmax:
            for lmax in args.lmax:
                for dis_type in args.dis_type:
                    for cutoff in args.cutoff:
                        try:
                            summary = run_one(
                                L=L,
                                dim=args.dim,
                                qmax=qmax,
                                lmax=lmax,
                                cutoff=cutoff,
                                method=args.method,
                                dis=args.dis,
                                dis_type=dis_type,
                            )
                            if summary is None:
                                continue
                            all_results.append(summary)

                            tag = (
                                f"L{safe_token(L)}_"
                                f"lmax{safe_token(f'{lmax:g}')}_"
                                f"qmax{safe_token(qmax)}_"
                                f"dis_type{safe_token(dis_type)}_"
                                f"cutoff{safe_token(f'{cutoff:g}')}_"
                                f"{ts}"
                            )
                            summary_path = out_dir / f"liubo_test_CPU_GPU_summary_{tag}.json"
                            with open(summary_path, "w", encoding="utf-8") as f:
                                json.dump(summary, f, indent=2)

                        except Exception as exc:
                            import traceback

                            print(
                                f"[ERROR] L={L}, lmax={lmax}, qmax={qmax}, dis_type={dis_type}, cutoff={cutoff} failed: {exc}",
                                flush=True,
                            )
                            traceback.print_exc()
                            all_results.append(
                                {
                                    "L": L,
                                    "dim": args.dim,
                                    "qmax": qmax,
                                    "lmax": lmax,
                                    "dis_type": dis_type,
                                    "cutoff": cutoff,
                                    "error": str(exc),
                                }
                            )

    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(
        f"{'L':>4} {'n':>6} {'qmax':>6} {'lmax':>8} {'type':>10} {'cutoff':>10} {'CPU(s)':>10} {'GPU(s)':>10} {'spd(x)':>8} {'cpu_steps':>10} {'gpu_steps':>10} {'cpu_mem(GB)':>12}"
    )
    print("-" * 120)

    for r in all_results:
        if "error" in r:
            print(
                f"{r.get('L', ''):>4} {'':>6} {r.get('qmax', ''):>6} {str(r.get('lmax', '')):>8} {str(r.get('dis_type', '')):>10} {str(r.get('cutoff', '')):>10} {'ERROR':>10} {'':>10} {'':>8} {'':>10} {'':>10} {'':>12}"
            )
            continue

        print(
            f"{r['L']:>4} {r['n']:>6} {r['qmax']:>6} {r['lmax']:>8.2f} {r['dis_type']:>10} {r['cutoff']:>10.1e} "
            f"{r['cpu_elapsed_s']:>10.3f} {r['gpu_elapsed_s']:>10.3f} {r['speedup_gpu_vs_cpu']:>8.2f} "
            f"{r['cpu_steps_evolved']:>10} {r['gpu_steps_evolved']:>10} {r['cpu_full_flow_mem_gb']:>12.3f}"
        )

    print("=" * 120)


if __name__ == "__main__":
    main()
