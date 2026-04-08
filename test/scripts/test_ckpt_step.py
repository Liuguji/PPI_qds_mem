#!/usr/bin/env python3
"""
Runtime sweep for flow_test_torch_real_step.

Goal:
- measure how many diagonalization steps are actually used;
- estimate full-trajectory memory peak if H2+Hint are stored at every step.

Usage:
    python test/scripts/test_ckpt_step.py
    python test/scripts/test_ckpt_step.py --L 3 4 --qmax 1000 2000 --lmax 100 150
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from psutil import cpu_count

# Must be set before first JAX import.
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")


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
from core.diag_routines.spinless_fermion import flow_test_torch_real_step


# Unit conversion: 1 GiB = 1024^3 bytes, 1 GB = 1000^3 bytes.
_GIB_TO_GB = (1024.0 ** 3) / (1000.0 ** 3)


def make_dl_list(lmax: float, qmax: int) -> np.ndarray:
    # Build qmax steps => qmax+1 time points, so "steps < qmax" means early convergence.
    return np.logspace(np.log10(0.001), np.log10(lmax), qmax + 1, endpoint=True, base=10)


def _fmt_array(x: list | np.ndarray) -> str:
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, list):
        return str(x)
    return f"[{x}]"


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

    # Keep per-case logs minimal; detailed parameter echo is optional.
    if os.environ.get("PYFLOW_TEST_VERBOSE", "0") in ("1", "true", "True", "on", "ON") and vary_params:
        vary_str = ", ".join(f"{k}={v}" for k, v in vary_params.items())
        print(f"    [{vary_str}]", flush=True)

    ham = models.hamiltonian("spinless fermion", dis_type, intr=True)
    ham.build(n, dim, dis, J, x, delta=delta)

    dl_list = make_dl_list(lmax, qmax)

    t0 = time.perf_counter()
    result = flow_test_torch_real_step(
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

    peak_gb = result.get("full_store_peak_gb_est")
    if peak_gb is None:
        peak_gib = result.get("full_store_peak_gib_est")
        if peak_gib is None:
            raise KeyError("flow_test_torch_real_step output missing memory estimate key")
        peak_gb = float(peak_gib) * _GIB_TO_GB

    run_result = {
        "L": L,
        "n": n,
        "qmax": qmax,
        "lmax": lmax,
        "elapsed": float(elapsed),
        "steps_evolved": int(result["steps_evolved"]),
        "full_store_peak_gb_est": float(peak_gb),
        "success": bool(result["steps_evolved"] < qmax),
    }

    return run_result, result


def _init_summary_xlsx(out_path: Path) -> None:
    if out_path.exists():
        # Append mode across reruns: keep existing workbook and append new rows.
        return

    try:
        wb_mod = __import__("openpyxl")
        wb = wb_mod.Workbook()
    except Exception as exc:
        raise RuntimeError("openpyxl is required for XLSX output. Install it with: pip install openpyxl") from exc

    ws = wb.active
    ws.title = "summary"

    # Required output schema:
    # 1) N, 2) lmax, 3) qmax, 4) actual steps, 5) required memory
    ws.append(["N", "lmax", "qmax", "real_steps", "required_memory_GB"])

    wb.save(out_path)


def _append_summary_xlsx(row: dict, out_path: Path) -> None:
    try:
        wb_mod = __import__("openpyxl")
        wb = wb_mod.load_workbook(out_path)
    except Exception as exc:
        raise RuntimeError("openpyxl is required for XLSX output. Install it with: pip install openpyxl") from exc

    ws = wb.active

    ws.append([
        int(row["n"]),
        float(row["lmax"]),
        int(row["qmax"]),
        int(row["steps_evolved"]),
        float(row["full_store_peak_gb_est"]),
    ])

    wb.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Runtime sweep for flow_test_torch_real_step")
    parser.add_argument("--L", type=int, nargs="+", default=[4], help="linear system sizes")
    parser.add_argument("--dim", type=int, default=2, help="spatial dimension")
    parser.add_argument("--qmax", type=int, nargs="+", help="max flow steps")
    parser.add_argument("--lmax", type=float, nargs="+", help="max flow time")
    parser.add_argument("--cutoff", type=float, default=1e-3, help="off-diagonal cutoff")
    parser.add_argument("--method", type=str, default="einsum", help="contraction method")
    parser.add_argument("--dis", type=float, default=5.0, help="disorder strength")
    parser.add_argument("--dis-type", type=str, default="random", dest="dis_type", help="disorder type")
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory for XLSX file")
    args = parser.parse_args()

    qmax_list = args.qmax if args.qmax else [2000]
    lmax_list = args.lmax if args.lmax else [150.0]

    print(
        "Run config: "
        f"L={_fmt_array(args.L)} dim={args.dim} qmax={_fmt_array(qmax_list)} "
        f"lmax={_fmt_array(lmax_list)} cutoff={args.cutoff:.1e} method={args.method}",
        flush=True,
    )

    varying_keys = []
    if len(args.L) > 1:
        varying_keys.append("L")
    if len(qmax_list) > 1:
        varying_keys.append("qmax")
    if len(lmax_list) > 1:
        varying_keys.append("lmax")

    all_results = []
    skipped_unsuccessful = 0

    if args.out_dir is None:
        args.out_dir = REPO_ROOT / "aaa_real_k"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = args.out_dir / "step_summary.xlsx"
    _init_summary_xlsx(xlsx_path)

    for L in args.L:
        for qmax in qmax_list:
            for lmax in lmax_list:
                vary_params = {}
                if "L" in varying_keys:
                    vary_params["L"] = L
                if "qmax" in varying_keys:
                    vary_params["qmax"] = qmax
                if "lmax" in varying_keys:
                    vary_params["lmax"] = lmax

                try:
                    run_result, _raw_result = run_one(
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
                    if not run_result["success"]:
                        skipped_unsuccessful += 1
                        if os.environ.get("PYFLOW_TEST_VERBOSE", "0") in ("1", "true", "True", "on", "ON"):
                            print(
                                f"    Skip: L={L}, qmax={qmax}, lmax={lmax}, "
                                f"steps={run_result['steps_evolved']} >= qmax={qmax}",
                                flush=True,
                            )
                        continue

                    all_results.append(run_result)
                    _append_summary_xlsx(run_result, xlsx_path)
                    if os.environ.get("PYFLOW_TEST_VERBOSE", "0") in ("1", "true", "True", "on", "ON"):
                        print("    Kept (successful).", flush=True)

                except Exception as exc:
                    import traceback

                    print(f"[ERROR] L={L}, qmax={qmax}, lmax={lmax} failed: {exc}")
                    traceback.print_exc()
                    skipped_unsuccessful += 1

    print("\n" + "=" * 80)
    print("SUCCESSFUL PARAMETER CASES")
    print("=" * 80)
    if all_results:
        for r in all_results:
            print(f"[SUCCESS] L={r['L']}, qmax={r['qmax']}, lmax={r['lmax']}")
    else:
        print("None")
    print("=" * 80)
    print(f"Successful cases kept: {len(all_results)}")
    print(f"Unsuccessful/failed cases skipped: {skipped_unsuccessful}")

    print(f"Saved XLSX summary to: {xlsx_path}")


if __name__ == "__main__":
    main()
