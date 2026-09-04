#!/usr/bin/env python3
"""Quick sanity statistics for self-play JSONL files.

  python inspect_data.py data/*.jsonl
"""
import collections
import glob
import json
import sys

import numpy as np


def main():
    paths = []
    for pat in sys.argv[1:]:
        paths.extend(sorted(glob.glob(pat)))
    n = 0
    games = set()
    hist = collections.Counter()
    by_shot_kind = collections.Counter()
    used, budget = [], []
    explore = 0
    results = collections.Counter()
    labels = collections.Counter()
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                games.add((p, r["game"]))
                if r["shot"] == 0:
                    hist[r["end_result_hammer"]] += 1
                kind = r["label"].split("(")[0].split(" ")[0]
                by_shot_kind[(r["shot"] % 2 == 1, kind)] += 1
                labels[kind] += 1
                used.append(r["used"]); budget.append(r["budget"])
                if "[explore]" in r["label"]:
                    explore += 1
                results[r["result"]] += 1
    print(f"files={len(paths)} records={n} games={len(games)}")
    tot = sum(hist.values())
    print("hammer end-result (k: count, frac):")
    for k in range(-4, 5):
        print(f"  {k:+d}: {hist[k]:5d} {hist[k] / max(1, tot):.3f}")
    mean_k = sum(k * v for k, v in hist.items()) / max(1, tot)
    print(f"  mean hammer result = {mean_k:+.3f}, ends = {tot}")
    print("shot kinds (all):", ", ".join(f"{k}={v}" for k, v in labels.most_common()))
    print("hammer-side kinds:", ", ".join(f"{k}={v}" for (h, k), v in sorted(by_shot_kind.items()) if h))
    print("lead-side kinds:  ", ", ".join(f"{k}={v}" for (h, k), v in sorted(by_shot_kind.items()) if not h))
    print(f"explore moves: {explore} ({explore / max(1, n):.3f})")
    print(f"used/budget: mean {np.mean(used):.2f}/{np.mean(budget):.2f}s, max used {np.max(used):.2f}s")
    print("record results:", dict(results))


if __name__ == "__main__":
    main()
