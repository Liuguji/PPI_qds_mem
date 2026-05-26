#!/usr/bin/env python3
"""验证 Wegner 流往返一致性：正向流 → 反向流（跳过时间演化）"""
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
verify_roundtrip = _mod.verify_roundtrip


def make_dl_list(lmax: float, qmax: int) -> np.ndarray:
    return np.logspace(np.log10(0.001), np.log10(lmax), qmax,
                       endpoint=True, base=10)


if __name__ == "__main__":
    L = 2
    dim = 2
    n = L ** dim
    qmax = 10000
    lmax = 1000.0
    cutoff = 1e-3
    method = "tensordot"
    dis = 5.0
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
    )
    elapsed = time.perf_counter() - t0

    print(f"\n耗时: {elapsed:.3f}s")

    # 最终判定
    max_err = float(np.max(np.abs(np.array(diff.real))))
    has_nan = bool(np.any(np.isnan(np.array(n2_roundtrip))))
    passed = (max_err < 1e-2) and (not has_nan)

    print("=" * 70)
    if passed:
        print("  ✅ 往返一致，验证通过")
    else:
        print("  ❌ 往返不一致")
    print("=" * 70)
