#!/usr/bin/env python3
"""
flow_imb 分支 0 测试（全轨迹，串行反向流，JAX CPU）。

    python test_imb/scripts/test_imb_0.py
"""

from __future__ import annotations

import os, sys, time, importlib
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

import numpy as np
import models.models as models

_mod = importlib.import_module("core.diag_routines.spinless_fermion copy")
flow_imb = _mod.flow_imb

from datetime import datetime
import json
from params import PARAM_SETS

DATA_ROOT = REPO_ROOT / "test_imb" / "datas" / "cpu_0"


def make_dl_list(lmax: float, qmax: int) -> np.ndarray:
    return np.logspace(np.log10(0.001), np.log10(lmax), qmax,
                       endpoint=True, base=10)


def make_t_val(default: float = 0.0) -> float:
    return float(default)


def set_switch_env(switch_num: int) -> None:
    os.environ["pipeline_switch"]   = str((switch_num // 1000) % 10)
    os.environ["parallel_switch"]   = str((switch_num // 100) % 10)
    os.environ["compress_switch"]   = str((switch_num // 10) % 10)
    os.environ["checkpoint_switch"] = str(switch_num % 10)
    os.environ["compress_mode"]     = "0"


def fmt_bytes(b: int) -> str:
    if b < 1024:
        return f"{b} B"
    elif b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    elif b < 1024 * 1024 * 1024:
        return f"{b / (1024 * 1024):.2f} MB"
    else:
        return f"{b / (1024 * 1024 * 1024):.3f} GB"


def run_single(config: dict, switch_num: int) -> dict:
    n = config["n"]
    ham = config["ham"]
    dl_list = config["dl_list"]
    qmax = config["qmax"]
    cutoff = config["cutoff"]
    t_val = config["t_val"]

    set_switch_env(switch_num)
    label = f"branch_{switch_num}"
    print(f"    [{label}] 运行中...", end=" ", flush=True)
    t0 = time.perf_counter()
    res = flow_imb(
        n, ham, dl_list, qmax, cutoff, t_val=t_val,
        method="tensordot",
    )
    elapsed = time.perf_counter() - t0
    steps = int(res.get("steps_evolved", -1))
    imb = float(res.get('Imbalance', 0.0))
    peak_mem = int(res.get("peak_memory_bytes", -1))
    timing = res.get('timing', {})
    print(f"耗时={elapsed:.1f}s  步数={steps}  I(0)={imb:.4f}  I(t_max)={imb:.4f}")

    return {
        "elapsed_s": elapsed,
        "steps_evolved": steps,
        "imbalance": float(imb),
        "peak_memory_bytes": peak_mem,
        "t_val": config["t_val"],
        "n": config["n"],
        "qmax": config["qmax"],
        "lmax": config["lmax"],
        "timing": {
            "h_diag_s": float(timing.get("h_diag_s", 0.0)),
            "op_fwd_s": float(timing.get("op_fwd_s", 0.0)),
            "time_evo_s": float(timing.get("time_evo_s", 0.0)),
            "op_bck_s": float(timing.get("op_bck_s", 0.0)),
            "total_s": float(timing.get("total_s", elapsed)),
        },
    }


def build_hamiltonian(params: dict) -> dict:
    L = params["L"]; dim = params["dim"]
    dis = params["dis"]; dis_type = params["dis_type"]
    J = params["J"]; delta = params["delta"]
    seed = params["seed"]; qmax = params["qmax"]
    lmax = params["lmax"]; cutoff = params["cutoff"]
    n = L ** dim

    os.environ["PYFLOW_SEED"] = str(seed)
    ham = models.hamiltonian("spinless fermion", dis_type, intr=True)
    ham.build(n, dim, dis, J, x=0.0, delta=delta)

    return {
        "n": n, "ham": ham,
        "dl_list": make_dl_list(lmax, qmax),
        "qmax": qmax, "lmax": lmax, "cutoff": cutoff,
        "t_val": make_t_val(),
        "L": L, "dim": dim, "dis": dis, "dis_type": dis_type,
        "J": J, "delta": delta, "seed": seed,
    }


_RESULT_KEYS = ["dim", "L", "dis_type", "dis", "J", "delta", "seed", "lmax", "qmax", "cutoff"]


def get_result_dir(params: dict) -> Path:
    dir_path = DATA_ROOT
    for key in _RESULT_KEYS:
        val = params[key]
        dir_name = f"{key}_{val:.6g}" if isinstance(val, float) else f"{key}_{val}"
        dir_path = dir_path / dir_name
    return dir_path


def count_existing_results(params: dict) -> int:
    dir_path = get_result_dir(params)
    return len(list(dir_path.glob("*.json"))) if dir_path.is_dir() else 0


def is_already_tested(params: dict, min_count: int = 1) -> bool:
    return count_existing_results(params) >= min_count


def save_result(result: dict, timestamp: str):
    c = result["config"]
    dir_path = get_result_dir(c)
    dir_path.mkdir(parents=True, exist_ok=True)

    serializable = {
        "config": {
            k: (float(v) if isinstance(v, (np.floating, float)) else
                int(v) if isinstance(v, (np.integer, int)) else v)
            for k, v in c.items()
            if k not in ("n", "ham", "dl_list", "t_val")
        },
        "branch_0": {
            "elapsed_s": result["branch_0"]["elapsed_s"],
            "t_val": float(result["branch_0"].get("t_val", 0.0)),
            "n": int(result["branch_0"].get("n", 0)),
            "qmax": int(result["branch_0"].get("qmax", 0)),
            "lmax": float(result["branch_0"].get("lmax", 0.0)),
            "steps_evolved": result["branch_0"]["steps_evolved"],
            "imbalance": result["branch_0"]["imbalance"],
            "peak_memory_bytes": result["branch_0"]["peak_memory_bytes"],
            "timing": {
                "h_diag_s": float(result["branch_0"].get("timing", {}).get("h_diag_s", 0.0)),
                "op_fwd_s": float(result["branch_0"].get("timing", {}).get("op_fwd_s", 0.0)),
                "time_evo_s": float(result["branch_0"].get("timing", {}).get("time_evo_s", 0.0)),
                "op_bck_s": float(result["branch_0"].get("timing", {}).get("op_bck_s", 0.0)),
                "total_s": float(result["branch_0"].get("timing", {}).get("total_s", 0.0)),
            },
        },
        "timestamp": timestamp,
    }

    out_file = dir_path / f"{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"    → 已保存: {out_file}")


def run_config(config: dict) -> dict:
    c = config
    print(f"\n{'─' * 60}")
    print(f"  配置: L={c['L']}  dim={c['dim']}  n={c['n']}  "
          f"dis={c['dis']:.1f}  J={c['J']:.1f}  delta={c['delta']:.1f}  "
          f"{c['dis_type']}  seed={c['seed']}")
    print(f"         qmax={c['qmax']}  lmax={c['lmax']}  cutoff={c['cutoff']:.1e}")
    print(f"{'─' * 60}")

    res0 = run_single(config, 0)

    imb = res0["imbalance"]
    print(f"    分支0: I(0)={imb:.4f} I(t_max)={imb:.4f} 耗时={res0['elapsed_s']:.1f}s")

    return {"config": config, "branch_0": res0}


def main():
    set_switch_env(0)

    print("=" * 70)
    print("  flow_imb 分支 0 测试（全轨迹，串行反向 ODE 流，JAX CPU）")
    print("=" * 70)
    print(f"  共 {len(PARAM_SETS)} 组参数")

    all_results = []
    skipped_params = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"  时间戳: {timestamp}")

    for i, params in enumerate(PARAM_SETS):
        print(f"\n[{i + 1}/{len(PARAM_SETS)}] "
              f"L={params['L']}, dis={params['dis']}, seed={params['seed']}")

        if is_already_tested(params):
            print(f"  → 已测试，跳过")
            skipped_params.append(params)
            continue

        config = build_hamiltonian(params)
        result = run_config(config)
        all_results.append(result)
        save_result(result, timestamp)

    # ── 汇总表格 ──
    print(f"\n{'=' * 90}")
    print("  汇总：各配置下分支 0 结果")
    print(f"{'=' * 90}")
    header = (f"{'L':>3s} {'dim':>3s} {'n':>4s} {'dis':>6s} {'J':>5s} {'Δ':>5s} "
              f"{'类型':>8s}  {'I(0)':>7s} {'I(tmax)':>9s} {'耗时':>8s}")
    print(header)
    print("-" * len(header))
    for r in all_results:
        c = r["config"]
        r0 = r["branch_0"]
        imb = r0["imbalance"]
        print(f"{c['L']:3d} {c['dim']:3d} {c['n']:4d} "
              f"{c['dis']:6.1f} {c['J']:5.1f} {c['delta']:5.2f} "
              f"{c['dis_type']:>8s}  {imb:7.4f} {imb:9.4f} "
              f"{r0['elapsed_s']:>7.1f}s")
    print(f"{'=' * 90}")

    if skipped_params:
        print(f"\n  已跳过 {len(skipped_params)} 组")
    print(f"\n  实际测试: {len(all_results)} 组 | 跳过: {len(skipped_params)} 组 | 总计: {len(PARAM_SETS)} 组")
    print("  完成。")


if __name__ == "__main__":
    main()
