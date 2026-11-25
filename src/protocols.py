"""Splits of other published studies -> split specs for data.load_batch(split=...).
Cross-cell: explicit train / val / test cell lists. Within-cell: cell@tr / cell@va / cell@te,
cut by cycle fraction f (train = first 0.9 f, val = next 0.1 f, test = rest).
Sources: the cited papers and their supplementary files.
"""
import json
from pathlib import Path

XB = {1: "2C", 2: "3C", 3: "R2.5", 4: "R3", 5: "RW", 6: "Sim_satellite"}
MANIFESTS = Path("data/raw/scpinn_manifests")      # SC-PINN supplementary (batteries-12-00414-s001.zip)


def x(b, i):
    return f"{XB[b]}_battery-{i}"


def within(batch, cells, f):
    return dict(batches=[batch], cut={c: f for c in cells},
                train=[c + "@tr" for c in cells], val=[c + "@va" for c in cells], test=[c + "@te" for c in cells])


def cross(batches, train, test, val=None):
    val = val or train[-1:]
    return dict(batches=batches, train=[c for c in train if c not in val], val=val, test=test)


def xjtu():
    P = {
        "X1_hho_b1c1_70": within(1, [x(1, 1)], 0.7),                    # HHO-LSTM, Sustainability 2024
        "X1_bigru_b1c3_60": within(1, [x(1, 3)], 0.6),                  # BiGRU-Att, Energy 2025
        "X1_bigru_b1c3_70": within(1, [x(1, 3)], 0.7),
        "X2_deng_b1c1-4_60": within(1, [x(1, i) for i in range(1, 5)], 0.6),   # Deng PINN, Batteries 2025
        "X3_he_b1c1_20": within(1, [x(1, 1)], 0.2),                     # He, JES 2025
        "X3_he_b1c6_20": within(1, [x(1, 6)], 0.2),
        "X3_he_b2c2_20": within(2, [x(2, 2)], 0.2),
        "X4_evo_b1-b23": cross([1, 2, 3], [x(1, i) for i in range(1, 9)],       # EVO-LSSVM, Measurement 2025
                               [x(b, i) for b in (2, 3) for i in range(1, 16 if b == 2 else 9)]),
        "X6_xlstm_b2": cross([2], [x(2, i) for i in range(1, 16) if i not in (1, 3, 4, 6, 8, 14)],  # xLSTM, IJEPES 2025
                             [x(2, i) for i in (1, 3, 4, 6, 8, 14)]),
    }
    for b in (1, 2, 3):                                                 # MMAU-Net / MSCNN: leave one cell out
        cs = [x(b, i) for i in range(1, 9)]
        for k, c in enumerate(cs):
            rest = [r for r in cs if r != c]
            P[f"X5_loo_b{b}c{k + 1}"] = cross([b], rest, [c], val=[rest[k % 7]])
    P.update(manifests("XJTU", lambda b, c: c))
    return P


def tju():
    P = {"T1_aw_nca": within(1, [f"b1_CY25-05_1-#{i}" for i in range(1, 6)], 0.5),   # AW-PINN, Batteries 2026
         "T1_aw_ncm": within(2, [f"b2_CY35-05_1-#{i}" for i in range(1, 5)], 0.5)}
    P.update(manifests("TJU", lambda b, c: f"b{b}_{c}"))
    return P


def manifests(ds, name):
    """SC-PINN, Batteries 2026: exact train / val / test lists per batch (X7 XJTU, T2 TJU)."""
    out = {}
    for f in sorted(MANIFESTS.glob(f"{ds}_batch_*.json")):
        m = json.loads(f.read_text())
        b = m["batch_index"] + 1
        out[f"{'X7' if ds == 'XJTU' else 'T2'}_scpinn_b{b}"] = dict(batches=[b], **{
            k: [name(b, c) for c in m[f"{s}_batteries"]] for k, s in (("train", "train"), ("val", "valid"), ("test", "test"))})
    return out


def get(name):
    return (tju() if name[0] == "T" else xjtu())[name]
