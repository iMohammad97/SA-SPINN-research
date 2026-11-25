"""Dataset switch: set env DS=tju (default xjtu). Paths and sampling constants per dataset."""
import os
from pathlib import Path

DS = os.environ.get("DS", "xjtu")
PROC = Path("data/proc")
TJU = DS == "tju"
CELLS = PROC / DS                                             # per-cell npz
HI_CSV = PROC / f"{DS}_hi.csv"
CACHE = PROC / ("soc_cache_tju.npz" if TJU else "soc_cache.npz")
DT_MAX = 600.0 if TJU else 2.0                                # s, cap on logging gaps (TJU logs every ~30 s / 5 mV)
STRIDE, CYC_EVERY = (1, 10) if TJU else (10, 5)               # SOC cache subsampling (rows, cycles)
MIN_DT = 60.0 if TJU else 0.0       # s; TJU logs irregularly (rows 0.1 s apart) -> du/dt residual explodes
BATCHES = [1, 2, 3] if TJU else [1, 2, 3, 4, 5, 6]
Q_NOM = {1: 3.5, 2: 3.5, 3: 2.5} if TJU else dict.fromkeys(range(1, 7), 2.0)   # Ah
STEP = "s7" if TJU else None                                  # TJU run folders: runs/s7_<what>


def draws_path(b):
    return PROC / (f"tju_hi_draws_b{b}.pt" if TJU else f"hi_draws_b{b}.pt")


def is_test(cells):
    """XJTU: name has 4 or 8 (AutoSPINN). TJU: sorted 1-based index i with i % 10 in {5, 9} (AutoSPINN / Wang 2024)."""
    if TJU:
        return [c for i, c in enumerate(sorted(cells), 1) if i % 10 in (5, 9)]
    return [c for c in cells if any(ch in c.split("-")[-1] for ch in "48")]
