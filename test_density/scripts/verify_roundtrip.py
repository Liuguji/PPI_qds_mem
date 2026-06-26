#!/usr/bin/env python3
"""验证 Wegner 流往返一致性：正向流 → （可选时间演化）→ 反向流
PyTorch GPU 版本
"""
import os
import sys
import time
import importlib
from pathlib import Path

# ===== GPU 选择（PyTorch 也会遵循此变量）=====
# 方式1: 改下面的数字选择 GPU (0, 1, 2, 3)
# 方式2: 命令行 CUDA_VISIBLE_DEVICES=0 python verify_roundtrip.py
GPU_ID =2
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
# =============================================

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

import numpy as np

# ===== 时间演化配置 =====
T_VAL = 1.0  # 单个时间点
# ========================

import models.models as models

_mod = importlib.import_module("core.diag_routines.spinless_fermion copy")
verify_roundtrip = _mod.verify_roundtrip_torch
assert verify_roundtrip.__name__ == "verify_roundtrip_torch", \
    f"错误: 导入了 {verify_roundtrip.__name__} 而非 verify_roundtrip_torch"


def make_dl_list(lmax: float, qmax: int) -> np.ndarray:
    return np.logspace(np.log10(0.001), np.log10(lmax), qmax,
                       endpoint=True, base=10)


if __name__ == "__main__":
    L = 7
    dim = 2
    n = L ** dim
    qmax = 10000
    lmax = 1000.0
    cutoff = 1e-3
    method = "tensordot"
    dis = 1.0
    dis_type = "linear"
    seed = 42
    site = 0

    print("=" * 70)
    print("  往返一致性验证")
    print("=" * 70)
    print(f"  L={L}  dim={dim}  n={n}")
    print(f"  qmax={qmax}  lmax={lmax}  cutoff={cutoff:.1e}")
    print(f"  method={method}  dis={dis}  dis_type={dis_type}")
    print(f"  seed={seed}  site={site}")
    print(f"  时间演化: ON, t_val = {T_VAL}")
    print()

    os.environ["PYFLOW_SEED"] = str(seed)
    ham = models.hamiltonian("spinless fermion", dis_type, intr=True)
    ham.build(n, dim, dis, J=1.0, x=0.0, delta=0.1)

    dl_list = make_dl_list(lmax, qmax)

    num = np.zeros((n, n), dtype=np.float64)
    num_int = np.zeros((n, n, n, n), dtype=np.float64)

    t0 = time.perf_counter()
    n2_roundtrip, n2_init, diff = verify_roundtrip(
        n=n,
        hamiltonian=ham,
        num=num,
        num_int=num_int,
        dl_list=dl_list,
        qmax=qmax,
        cutoff=cutoff,
        site=site,
        t_val=T_VAL,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n耗时: {elapsed:.3f}s")

    # 仅 t=0 时往返结果应与初始算符一致，才做判定
    if abs(T_VAL) < 1e-12:
        max_err = float(np.max(np.abs(diff)))
        has_nan = bool(np.any(np.isnan(diff)))
        passed = (max_err < 1e-2) and (not has_nan)
        print("=" * 70)
        if passed:
            print("  ✅ 往返一致，验证通过")
        else:
            print("  ❌ 往返不一致")
        print("=" * 70)
