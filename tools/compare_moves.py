# M1 test (b): our exact legal moves must equal the site's, position by position.
# Both sides are reduced to the same canonical form: per piece, the set of destination squares,
# written as merged step intervals on each of the 4 lines through the piece plus loose points.
#   python3 tools/compare_moves.py positions.icn      (one stand-alone ICN per line)
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "py"))
from match import ask, start_referee  # noqa: E402

ICORE = os.path.expanduser("~/build/icore/release/icore")
INF = float("inf")


def canonical_line(dx, dy):
    """Unit direction -> (line key, sign) with the key pointing right or up."""
    if dx < 0 or (dx == 0 and dy < 0):
        return (-dx, -dy), -1
    return (dx, dy), 1


def steps_on_line(fx, fy, x, y):
    """(line, signed steps) if (x, y) lies on a rank, file or diagonal through (fx, fy)."""
    ddx, ddy = x - fx, y - fy
    if ddx == 0 and ddy == 0:
        return None
    if ddx != 0 and ddy != 0 and abs(ddx) != abs(ddy):
        return None
    n = max(abs(ddx), abs(ddy))
    line, sign = canonical_line((ddx > 0) - (ddx < 0), (ddy > 0) - (ddy < 0))
    return line, sign * n


def merge(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def canonical(fx, fy, points, intervals):
    """points: [(x, y)]; intervals: [(line, lo, hi)]. Returns a comparable structure."""
    lines, loose = {}, set()
    for line, lo, hi in intervals:
        lines.setdefault(line, []).append((lo, hi))
    for x, y in points:
        on = steps_on_line(fx, fy, x, y)
        if on is None:
            loose.add((x, y))
        else:
            lines.setdefault(on[0], []).append((on[1], on[1]))
    return {k: merge(v) for k, v in lines.items() if v}, loose


def ours_canonical(moves):
    by_piece = {}
    for fx, fy, tx, ty, _promo in moves["singles"]:
        by_piece.setdefault((fx, fy), ([], []))[0].append((tx, ty))
    for fx, fy, dx, dy, hi in moves["slides"]:
        line, sign = canonical_line(dx, dy)
        hi = INF if hi is None else hi
        iv = (1, hi) if sign > 0 else (-hi, -1)
        by_piece.setdefault((fx, fy), ([], []))[1].append((line, *iv))
    return {k: canonical(*k, pts, ivs) for k, (pts, ivs) in by_piece.items()}


def site_canonical(pieces):
    out, brute = {}, set()
    for p in pieces:
        fx, fy = int(p["from"][0]), int(p["from"][1])
        pts = [(int(x), int(y)) for x, y in p["individual"]]
        ivs = []
        for key, (lo, hi) in p["sliding"].items():
            dx, dy = map(int, key.split(","))
            line, sign = canonical_line(dx, dy)
            lo = -INF if lo is None else int(lo)
            hi = INF if hi is None else int(hi)
            if sign < 0:
                lo, hi = -hi, -lo
            # The site's slide limits span the piece's own square (step 0); it is not a move.
            for a, b in ((lo, min(hi, -1)), (max(lo, 1), hi)):
                if a <= b:
                    ivs.append((line, a, b))
        c = canonical(fx, fy, pts, ivs)
        if c[0] or c[1]:
            out[(fx, fy)] = c
        if p["brute"]:
            brute.add((fx, fy))
    return out, brute


def main(path):
    icns = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
    res = subprocess.run([ICORE, "moves"], input="\n".join(icns) + "\n", capture_output=True, text=True, check=True)
    ours = [json.loads(l) for l in res.stdout.splitlines()]
    referee = start_referee()
    same = differ = brute_pos = checks = end_differ = 0
    for icn, o in zip(icns, ours):
        r = ask(referee, {"cmd": "moves", "icn": icn}, 120)
        if not r or not r["ok"]:
            print("referee error:", r, icn)
            differ += 1
            continue
        checks += o["check"]
        e = ask(referee, {"cmd": "new", "id": "cmp", "icn": icn}, 120)
        site_end = (e["result"], e["condition"]) if e["ok"] else ("error", None)
        our_end = (o["end"]["result"], o["end"]["condition"]) if o["end"] else (None, None)
        if site_end != our_end:
            end_differ += 1
            if end_differ <= 5:
                print("END MISMATCH ours", our_end, "site", site_end, icn)
        a = {k: v for k, v in ours_canonical(o["moves"]).items() if v[0] or v[1]}
        b, brute = site_canonical(r["pieces"])
        if brute:
            brute_pos += 1
        if a == b:
            same += 1
            continue
        # A "brute" slide on the site still has to pass a check test per square, so its raw
        # interval may be larger than the legal one. Settle those pieces square by square.
        bad = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
        if all(k in brute for k in bad) and settle_brute(referee, icn, a, b, bad):
            same += 1
            continue
        differ += 1
        if differ <= 10:
            print("MISMATCH", icn)
            for k in bad:
                print("   piece", k, "\n     ours:", a.get(k), "\n     site:", b.get(k))
    print(f"{same + differ} positions ({checks} in check): {same} equal, {differ} differ ({brute_pos} had brute slides); "
          f"game end: {same + differ - end_differ} agree, {end_differ} differ")


def settle_brute(referee, icn, a, b, pieces):
    """Every square where the two sets disagree (sampled at interval edges) must match the
    site's per-square verdict, with ours as the claim being tested."""
    for k in pieces:
        oa, ob = a.get(k, ({}, set())), b.get(k, ({}, set()))
        for line in set(oa[0]) | set(ob[0]):
            edges = {v for iv in oa[0].get(line, []) + ob[0].get(line, []) for v in iv if v not in (INF, -INF)}
            for s in {e + d for e in edges for d in (-1, 0, 1)} - {0}:
                x, y = k[0] + line[0] * s, k[1] + line[1] * s
                mine = any(lo <= s <= hi for lo, hi in oa[0].get(line, []))
                r = ask(referee, {"cmd": "legal", "icn": icn, "move": f"{k[0]},{k[1]}>{x},{y}"}, 60)
                if r["legal"] != mine:
                    return False
    return True


if __name__ == "__main__":
    main(sys.argv[1])
