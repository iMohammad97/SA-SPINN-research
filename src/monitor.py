"""Live terminal monitor.  Usage: python src/monitor.py runs/s8_xjtu [runs/s8_tju ...]   (Ctrl+C to quit)"""
import json, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def load(p):
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):   # file mid-write
        return None


def bar(f, w=40):
    return "█" * int(f * w) + "░" * (w - int(f * w))


def fmt(s):
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def block(root):
    """Lines for one run folder; second value = finished?"""
    S = [s for s in map(load, sorted((root / "status").glob("*.json"))) if s]
    try:
        n = int((root / "log.txt").read_text(encoding="utf-8").split()[0])   # "<N> runs -> ..."
    except (OSError, ValueError, IndexError):
        n = len(S)
    if not S:
        return [f"{root}   no status yet", ""], False
    done = [s for s in S if s["done"]]
    run = [s for s in S if not s["done"] and s["epoch"] > 0]
    tot = sum(s["epochs"] for s in S) + 100 * (n - len(S))                # unstarted runs: 100 epochs
    cur = sum(s["epoch"] for s in S)
    frac = cur / tot
    spe = [s["elapsed"] / s["epoch"] for s in S if s["epoch"]]
    eta = (sum(spe) / len(spe)) * (tot - cur) / max(len(run), 1) if spe else 0
    lines = [f"{root}",
             f"[{bar(frac)}] {100 * frac:5.1f}%   runs {len(done)}/{n} done, {len(run)} running   ETA ~{fmt(eta)}"]
    for s in run:
        lines.append(f"  {s['id']:<60} ep {s['epoch']:>3}/{s['epochs']:<3} [{bar(s['epoch'] / s['epochs'], 15)}] "
                     f"val SOH {100 * s.get('val_soh', 0):5.2f}%  SOC {100 * s.get('val_soc', 0):5.2f}%")
    return lines + [""], len(done) == n


roots = [Path(r) for r in sys.argv[1:]] or [Path("runs/s1_grouping")]   # several folders allowed
while True:
    out, fin = [time.strftime("%H:%M:%S") + "   (ETA is an upper bound: early stopping shortens runs)", ""], []
    for r in roots:
        lines, f = block(r)
        out += lines; fin.append(f)
    print("\033[2J\033[H" + "\n".join(out), flush=True)
    if all(fin):
        break
    time.sleep(3)
