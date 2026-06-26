"""
共享参数组：flow_imb 各分支测试公用。

用法：from params import PARAM_SETS
"""

PARAM_SETS = [
    # ──────── L=2 (n=4) ────────
    {"dim": 2, "L": 2, "dis_type": "linear", "dis": 1.0, "J": 1.0, "delta": 0.1, "seed": 42, "lmax": 1000.0, "qmax": 10000, "cutoff": 1e-3},
    {"dim": 2, "L": 2, "dis_type": "random", "dis": 1.0, "J": 1.0, "delta": 0.1, "seed": 42, "lmax": 1000.0, "qmax": 10000, "cutoff": 1e-3},

    # ──────── L=3 (n=9) ────────
    {"dim": 2, "L": 3, "dis_type": "linear", "dis": 1.0, "J": 1.0, "delta": 0.1, "seed": 42, "lmax": 1000.0, "qmax": 10000, "cutoff": 1e-3},
    {"dim": 2, "L": 3, "dis_type": "random", "dis": 1.0, "J": 1.0, "delta": 0.1, "seed": 42, "lmax": 1000.0, "qmax": 10000, "cutoff": 1e-3},

    # ──────── L=4 (n=16) ────────
    {"dim": 2, "L": 4, "dis_type": "linear", "dis": 1.0, "J": 1.0, "delta": 0.1, "seed": 42, "lmax": 1000.0, "qmax": 10000, "cutoff": 1e-3},
    {"dim": 2, "L": 4, "dis_type": "random", "dis": 1.0, "J": 1.0, "delta": 0.1, "seed": 42, "lmax": 1000.0, "qmax": 10000, "cutoff": 1e-3},

    # ──────── L=5 (n=25) ────────
    {"dim": 2, "L": 5, "dis_type": "linear", "dis": 1.0, "J": 1.0, "delta": 0.1, "seed": 42, "lmax": 1000.0, "qmax": 10000, "cutoff": 1e-3},
    {"dim": 2, "L": 5, "dis_type": "random", "dis": 1.0, "J": 1.0, "delta": 0.1, "seed": 42, "lmax": 1000.0, "qmax": 10000, "cutoff": 1e-3},

    # ──────── L=6 (n=36) ────────
    {"dim": 2, "L": 6, "dis_type": "linear", "dis": 1.0, "J": 1.0, "delta": 0.1, "seed": 42, "lmax": 1000.0, "qmax": 10000, "cutoff": 1e-3},
    {"dim": 2, "L": 6, "dis_type": "random", "dis": 1.0, "J": 1.0, "delta": 0.1, "seed": 42, "lmax": 1000.0, "qmax": 10000, "cutoff": 1e-3},

    # ──────── L=7 (n=49) ────────
    {"dim": 2, "L": 7, "dis_type": "linear", "dis": 1.0, "J": 1.0, "delta": 0.1, "seed": 42, "lmax": 1000.0, "qmax": 10000, "cutoff": 1e-3},
    {"dim": 2, "L": 7, "dis_type": "random", "dis": 1.0, "J": 1.0, "delta": 0.1, "seed": 42, "lmax": 1000.0, "qmax": 10000, "cutoff": 1e-3},
]
