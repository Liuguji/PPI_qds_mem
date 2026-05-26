#!/usr/bin/env python3
"""
内存峰值测试：flow_dyn_density 分支 11（compress + checkpoint）。

所有参数都在顶层 PARAM_SETS 列表中设定，直接运行即可：
    python test_density_11.py
"""

from __future__ import annotations

import os
import sys
import time
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

from datetime import datetime
import json

# 数据输出根目录
DATA_ROOT = REPO_ROOT / "test_density" / "data" / "data_2"


# ══════════════════════════════════════════════════════════════════════
#  顶层参数组：{dim, L, dis_type, dis, J, delta, seed, lmax, qmax, cutoff}
#  按 dim → L → dis_type → dis 从小到大排序
#
#  注意：seed 只影响 dis_type="random" / "QPgolden" 等含随机性的无序类型。
#        对 dis_type="linear"（确定性无序）seed 不起作用，可任意设置。
# ══════════════════════════════════════════════════════════════════════
PARAM_SETS = [
    # ═════════════════════════════════════════════════════
    #  已测试参数组（全部注释，留档备查）
    #  按 L 分组，每组内按 dis 从小到大排列
    # ═════════════════════════════════════════════════════

    # ──────── L=2 (n=4) ────────
    {"dim": 2, "L": 2, "dis_type": "linear", "dis": 1.0,  "J": 1.0, "delta": 0.1, "seed": 42,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},
    {"dim": 2, "L": 2, "dis_type": "linear", "dis": 5.0,  "J": 1.0, "delta": 0.1, "seed": 43,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},
    {"dim": 2, "L": 2, "dis_type": "linear", "dis": 10.0, "J": 1.0, "delta": 0.1, "seed": 44,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},

    # ──────── L=3 (n=9) ────────
    {"dim": 2, "L": 3, "dis_type": "linear", "dis": 1.0,  "J": 1.0, "delta": 0.1, "seed": 47,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},
    {"dim": 2, "L": 3, "dis_type": "linear", "dis": 1.0,  "J": 1.0, "delta": 0.1, "seed": 50,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},
    {"dim": 2, "L": 3, "dis_type": "linear", "dis": 2.0,  "J": 1.0, "delta": 0.1, "seed": 51,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},
    {"dim": 2, "L": 3, "dis_type": "linear", "dis": 3.0,  "J": 1.0, "delta": 0.1, "seed": 52,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},
    {"dim": 2, "L": 3, "dis_type": "linear", "dis": 4.0,  "J": 1.0, "delta": 0.1, "seed": 53,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},
    {"dim": 2, "L": 3, "dis_type": "linear", "dis": 5.0,  "J": 1.0, "delta": 0.1, "seed": 54,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},
    {"dim": 2, "L": 3, "dis_type": "linear", "dis": 10.0, "J": 1.0, "delta": 0.1, "seed": 55,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},

    # ──────── L=4 (n=16) ────────
    {"dim": 2, "L": 4, "dis_type": "linear", "dis": 1.0,  "J": 1.0, "delta": 0.1, "seed": 48,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},
    {"dim": 2, "L": 4, "dis_type": "linear", "dis": 1.0,  "J": 1.0, "delta": 0.1, "seed": 56,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},
    {"dim": 2, "L": 4, "dis_type": "linear", "dis": 2.0,  "J": 1.0, "delta": 0.1, "seed": 57,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},
    {"dim": 2, "L": 4, "dis_type": "linear", "dis": 3.0,  "J": 1.0, "delta": 0.1, "seed": 58,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},
    {"dim": 2, "L": 4, "dis_type": "linear", "dis": 4.0,  "J": 1.0, "delta": 0.1, "seed": 59,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},
    {"dim": 2, "L": 4, "dis_type": "linear", "dis": 5.0,  "J": 1.0, "delta": 0.1, "seed": 60,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},
    {"dim": 2, "L": 4, "dis_type": "linear", "dis": 10.0, "J": 1.0, "delta": 0.1, "seed": 61,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},

    # ──────── L=5 (n=25) ────────
    # {"dim": 2, "L": 5, "dis_type": "linear", "dis": 1.0,  "J": 1.0, "delta": 0.1, "seed": 49,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},
    # {"dim": 2, "L": 5, "dis_type": "linear", "dis": 1.0,  "J": 1.0, "delta": 0.1, "seed": 62,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},
    # {"dim": 2, "L": 5, "dis_type": "linear", "dis": 2.0,  "J": 1.0, "delta": 0.1, "seed": 63,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},
    # {"dim": 2, "L": 5, "dis_type": "linear", "dis": 3.0,  "J": 1.0, "delta": 0.1, "seed": 64,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},
    # {"dim": 2, "L": 5, "dis_type": "linear", "dis": 4.0,  "J": 1.0, "delta": 0.1, "seed": 65,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},
    # {"dim": 2, "L": 5, "dis_type": "linear", "dis": 5.0,  "J": 1.0, "delta": 0.1, "seed": 66,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},
    # {"dim": 2, "L": 5, "dis_type": "linear", "dis": 10.0, "J": 1.0, "delta": 0.1, "seed": 67,  "lmax": 100.0, "qmax": 2000, "cutoff": 1e-3},

    # # ──────── L=6 (n=36) ────────
    # {"dim": 2, "L": 6, "dis_type": "linear", "dis": 1.0,  "J": 1.0, "delta": 0.1, "seed": 49,  "lmax": 1000.0, "qmax": 10000, "cutoff": 1e-3},
    # {"dim": 2, "L": 6, "dis_type": "linear", "dis": 1.0,  "J": 1.0, "delta": 0.1, "seed": 62,  "lmax": 1000.0, "qmax": 10000, "cutoff": 1e-3},
    # {"dim": 2, "L": 6, "dis_type": "linear", "dis": 2.0,  "J": 1.0, "delta": 0.1, "seed": 63,  "lmax": 1000.0, "qmax": 10000, "cutoff": 1e-3},
    # {"dim": 2, "L": 6, "dis_type": "linear", "dis": 3.0,  "J": 1.0, "delta": 0.1, "seed": 64,  "lmax": 1000.0, "qmax": 10000, "cutoff": 1e-3},
    # {"dim": 2, "L": 6, "dis_type": "linear", "dis": 4.0,  "J": 1.0, "delta": 0.1, "seed": 65,  "lmax": 1000.0, "qmax": 10000, "cutoff": 1e-3},
    # {"dim": 2, "L": 6, "dis_type": "linear", "dis": 5.0,  "J": 1.0, "delta": 0.1, "seed": 66,  "lmax": 1000.0, "qmax": 10000, "cutoff": 1e-3},
    # {"dim": 2, "L": 6, "dis_type": "linear", "dis": 10.0, "J": 1.0, "delta": 0.1, "seed": 67,  "lmax": 1000.0, "qmax": 10000, "cutoff": 1e-3},

]


def make_dl_list(lmax: float, qmax: int) -> np.ndarray:
    return np.logspace(np.log10(0.001), np.log10(lmax), qmax,
                       endpoint=True, base=10)


def set_switch_env(switch_num: int) -> None:
    """设置环境变量控制 flow_dyn_density 的分支选择。"""
    os.environ["pipeline_switch"]   = str((switch_num // 1000) % 10)
    os.environ["parallel_switch"]   = str((switch_num // 100) % 10)
    os.environ["compress_switch"]   = str((switch_num // 10) % 10)
    os.environ["checkpoint_switch"] = str(switch_num % 10)
    os.environ["compress_mode"]     = "0"
    # Torch GPU 后端控制：设为 "1" 启用 PyTorch GPU 路径，"0" 使用 JAX CPU
    os.environ["PYFLOW_USE_TORCH"]  = "1"
    os.environ["PYFLOW_GPU_ID"]     = "0"


def fmt_bytes(b: int) -> str:
    """格式化字节数为可读字符串。"""
    if b < 1024:
        return f"{b} B"
    elif b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    elif b < 1024 * 1024 * 1024:
        return f"{b / (1024 * 1024):.2f} MB"
    else:
        return f"{b / (1024 * 1024 * 1024):.3f} GB"


def run_single(config: dict, switch_num: int) -> dict:
    """运行单个分支，返回结果字典。"""
    n = config["n"]
    ham = config["ham"]
    dl_list = config["dl_list"]
    qmax = config["qmax"]
    cutoff = config["cutoff"]
    tlist = config["tlist"]

    num = np.zeros((n, n), dtype=np.float64)
    num_int = np.zeros((n, n, n, n), dtype=np.float64)

    set_switch_env(switch_num)
    label = f"branch_{switch_num}"
    print(f"    [{label}] 运行中...", end=" ", flush=True)
    t0 = time.perf_counter()
    res = flow_dyn_density(
        n, ham, num, num_int, dl_list, qmax, cutoff,
        tlist=tlist, state=None, method="tensordot",
    )
    elapsed = time.perf_counter() - t0
    mem = int(res.get("peak_memory_bytes", -1))
    steps = int(res.get("steps_evolved", -1))
    print(f"耗时={elapsed:.1f}s  内存峰值={fmt_bytes(mem)}  步数={steps}")

    return {
        "elapsed_s": elapsed,
        "peak_memory_bytes": mem,
        "steps_evolved": steps,
    }


def build_hamiltonian(params: dict) -> dict:
    """根据参数组构建哈密顿量和流列表，返回完整 config。"""
    L = params["L"]
    dim = params["dim"]
    dis = params["dis"]
    dis_type = params["dis_type"]
    J = params["J"]
    delta = params["delta"]
    seed = params["seed"]
    qmax = params["qmax"]
    lmax = params["lmax"]
    cutoff = params["cutoff"]
    n = L ** dim

    os.environ["PYFLOW_SEED"] = str(seed)
    ham = models.hamiltonian("spinless fermion", dis_type, intr=True)
    ham.build(n, dim, dis, J, x=0.0, delta=delta)

    dl_list = make_dl_list(lmax, qmax)
    tlist = [0.0]

    return {
        "n": n, "ham": ham,
        "dl_list": dl_list, "qmax": qmax, "lmax": lmax, "cutoff": cutoff,
        "tlist": tlist,
        "L": L, "dim": dim, "dis": dis, "dis_type": dis_type,
        "J": J, "delta": delta, "seed": seed,
    }


# 用于判断"已测试"的参数键列表（与 save_result 目录层级一致）
_RESULT_KEYS = ["dim", "L", "dis_type", "dis", "J", "delta", "seed", "lmax", "qmax", "cutoff"]


def get_result_dir(params: dict) -> Path:
    """根据参数 dict 计算对应的结果存储目录路径。"""
    use_torch = os.environ.get("PYFLOW_USE_TORCH", "0")
    if use_torch == "1":
        base = REPO_ROOT / "test_density" / "data" / "gpu_data_2"
    else:
        base = DATA_ROOT
    compress_mode = os.environ.get("compress_mode", "1")
    dir_path = base / f"compress_mode_{compress_mode}"
    for key in _RESULT_KEYS:
        val = params[key]
        if isinstance(val, float):
            dir_name = f"{key}_{val:.6g}"
        else:
            dir_name = f"{key}_{val}"
        dir_path = dir_path / dir_name
    return dir_path


def count_existing_results(params: dict) -> int:
    """统计某个参数组已保存的结果数量（JSON 文件数）。"""
    dir_path = get_result_dir(params)
    if not dir_path.is_dir():
        return 0
    return len(list(dir_path.glob("*.json")))


def is_already_tested(params: dict, min_count: int = 2) -> bool:
    """检测参数组是否已测试过 >= min_count 次。"""
    return count_existing_results(params) >= min_count


def save_result(result: dict, timestamp: str):
    """将单组参数的结果写入带参数层级目录的 JSON 文件。"""
    c = result["config"]
    dir_path = get_result_dir(c)

    dir_path.mkdir(parents=True, exist_ok=True)

    # 序列化（numpy 类型 → Python 原生类型）
    serializable = {
        "config": {
            k: (float(v) if isinstance(v, (np.floating, float)) else
                int(v) if isinstance(v, (np.integer, int)) else v)
            for k, v in c.items()
            if k not in ("n", "ham", "dl_list", "tlist")
        },
        "branch_11": {
            "elapsed_s": result["branch_11"]["elapsed_s"],
            "peak_memory_bytes": result["branch_11"]["peak_memory_bytes"],
            "steps_evolved": result["branch_11"]["steps_evolved"],
        },
        "timestamp": timestamp,
    }

    out_file = dir_path / f"{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"    → 已保存: {out_file}")


def run_config(config: dict) -> dict:
    """对单个配置运行分支 11，返回结果。"""
    c = config
    print(f"\n{'─' * 60}")
    print(f"  配置: L={c['L']}  dim={c['dim']}  n={c['n']}  "
          f"dis={c['dis']:.1f}  J={c['J']:.1f}  delta={c['delta']:.1f}  "
          f"{c['dis_type']}  seed={c['seed']}")
    print(f"         qmax={c['qmax']}  lmax={c['lmax']}  cutoff={c['cutoff']:.1e}")
    print(f"{'─' * 60}")

    res11 = run_single(config, 11)

    mem11 = res11["peak_memory_bytes"]
    print(f"    分支11 内存: {fmt_bytes(mem11)}")

    return {
        "config": config,
        "branch_11": res11,
    }


def main():
    # 提前调用 set_switch_env，确保 compress_mode 等环境变量
    # 在 is_already_tested 和 save_result 中使用同一份值。
    set_switch_env(11)

    print("=" * 70)
    print("  内存峰值测试：flow_dyn_density 分支 11（compress + checkpoint）")
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
            existing_count = count_existing_results(params)
            print(f"  → 已测试 {existing_count} 次，跳过")
            skipped_params.append(params)
            continue

        config = build_hamiltonian(params)
        result = run_config(config)
        all_results.append(result)
        save_result(result, timestamp)

    # ===== 汇总表格 =====
    print(f"\n{'=' * 90}")
    print("  汇总：各配置下分支 11 内存峰值")
    print(f"{'=' * 90}")
    header = (f"{'L':>3s} {'dim':>3s} {'n':>3s} {'dis':>6s} {'J':>5s} {'Δ':>5s} "
              f"{'类型':>8s}  "
              f"{'分支11内存':>14s} {'耗时':>8s}")
    print(header)
    print("-" * len(header))
    for r in all_results:
        c = r["config"]
        r11 = r["branch_11"]
        print(f"{c['L']:3d} {c['dim']:3d} {c['n']:3d} "
              f"{c['dis']:6.1f} {c['J']:5.1f} {c['delta']:5.2f} "
              f"{c['dis_type']:>8s}  "
              f"{fmt_bytes(r11['peak_memory_bytes']):>14s} "
              f"{r11['elapsed_s']:>7.1f}s")

    print(f"{'=' * 90}")

    if skipped_params:
        print(f"\n  已跳过 {len(skipped_params)} 组参数（已测试 ≥2 次）：")
        for p in skipped_params:
            print(f"    L={p['L']}, dis={p['dis']}, seed={p['seed']}")

    print(f"\n  实际测试: {len(all_results)} 组 | 跳过: {len(skipped_params)} 组 | 总计: {len(PARAM_SETS)} 组")
    print("  完成。")


if __name__ == "__main__":
    main()
