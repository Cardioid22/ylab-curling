#!/usr/bin/env python3
"""
Multi-seed depth3 実験用に 8 局面を抽出する。

実装方針:
  - 既存 test_positions_<TIMESTAMP>/batch_*.csv から (match_id, end, shot_num) を
    キーに 8 局面を選び、新ディレクトリ batch_0001.csv に書き出す。
  - 選択は前回 depth3 結果 CSV があれば「shot_num バケット × score_diff バケット」
    で層別ランダム抽出、なければ shot_num だけで層別する。
  - 出力ディレクトリの全行数 == n_states になるので、experiments/depth_n_mcts_experiment.cpp の
    sampleTestPositions が no-op (return all) になり、--seed を変えても 8 局面が固定される。

Usage:
  python scripts/pick_multiseed_positions.py \
      --src-dir   clustered_ayumu/test_positions_20260417_055725 \
      --prev-csv  depth3_experiment/depth3_run_20260506_192817_parallel/depth3_results_COMBINED.csv \
      --out-dir   test_positions_multiseed8 \
      --n 8 --seed 7
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path


def load_positions(src_dir: Path) -> tuple[list[str], dict[tuple[int, int, int], str]]:
    """batch_*.csv を読んで {(match_id, end, shot_num): raw_csv_row} を返す."""
    files = sorted(src_dir.glob("batch_*.csv"))
    if not files:
        sys.exit(f"No batch_*.csv under {src_dir}")
    header: list[str] | None = None
    rows: dict[tuple[int, int, int], str] = {}
    for f in files:
        with open(f, newline="") as fh:
            reader = csv.reader(fh)
            h = next(reader)
            if header is None:
                header = h
            elif h != header:
                sys.exit(f"Header mismatch between {files[0].name} and {f.name}")
            for row in reader:
                key = (int(row[0]), int(row[1]), int(row[2]))
                rows[key] = ",".join(row)
    assert header is not None
    return header, rows


def pick_keys(prev_csv: Path | None, n: int, seed: int) -> list[tuple[int, int, int]]:
    """前回結果から層別抽出。なければ警告して空リスト (後段で fallback)."""
    if prev_csv is None or not prev_csv.exists():
        return []
    rng = random.Random(seed)
    with open(prev_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        records = list(reader)

    def shot_bucket(s: int) -> str:
        if s <= 3:   return "early"
        if s <= 7:   return "mid1"
        if s <= 11:  return "mid2"
        return "late"

    def diff_bucket(d: float) -> str:
        if d <= 0.1: return "tiny"
        if d <= 0.5: return "moderate"
        return "large"

    groups: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        try:
            sn = int(r["shot_num"])
            sd = float(r["score_diff"])
        except (KeyError, ValueError):
            continue
        groups.setdefault((shot_bucket(sn), diff_bucket(sd)), []).append(r)

    # 4 shot bucket × 3 diff bucket = 12 sub-groups. n=8 なら多様な 8 を取る:
    # まず shot bucket ごとに 2 個ずつ (= 8)、diff bucket がばらつくよう shuffle して取る
    picks: list[dict] = []
    for sb in ["early", "mid1", "mid2", "late"]:
        cands = [r for (s, _), rs in groups.items() if s == sb for r in rs]
        rng.shuffle(cands)
        # diff bucket 多様性を出すため diff 順に並べ替えて等間隔ピック
        cands.sort(key=lambda r: float(r["score_diff"]))
        if len(cands) >= 2:
            picks.append(cands[len(cands) // 4])
            picks.append(cands[3 * len(cands) // 4])
        else:
            picks.extend(cands)

    # n 未満ならランダムで埋める
    chosen_keys = {(int(r["game_id"]), int(r["end"]), int(r["shot_num"])) for r in picks}
    if len(chosen_keys) < n:
        remaining = [r for r in records
                     if (int(r["game_id"]), int(r["end"]), int(r["shot_num"])) not in chosen_keys]
        rng.shuffle(remaining)
        for r in remaining:
            picks.append(r)
            chosen_keys.add((int(r["game_id"]), int(r["end"]), int(r["shot_num"])))
            if len(chosen_keys) >= n:
                break

    out = []
    for r in picks[:n]:
        out.append((int(r["game_id"]), int(r["end"]), int(r["shot_num"])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=Path, required=True,
                    help="既存 batch_*.csv が入ったディレクトリ")
    ap.add_argument("--prev-csv", type=Path, default=None,
                    help="前回 depth3 結果 CSV (層別抽出用、任意)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="抽出した 8 局面を書き出す新ディレクトリ")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7,
                    help="層別内の抽選 seed (再現性用)")
    args = ap.parse_args()

    header, all_rows = load_positions(args.src_dir)
    print(f"Loaded {len(all_rows)} positions from {args.src_dir}", file=sys.stderr)

    keys = pick_keys(args.prev_csv, args.n, args.seed)
    if not keys:
        print(f"No prev CSV; falling back to random {args.n} from src", file=sys.stderr)
        rng = random.Random(args.seed)
        keys = rng.sample(list(all_rows.keys()), args.n)

    # キーがソース側に存在しないものを除く
    valid_keys = [k for k in keys if k in all_rows]
    missing = [k for k in keys if k not in all_rows]
    if missing:
        print(f"WARNING: {len(missing)} keys not found in src: {missing[:3]}...", file=sys.stderr)
    if len(valid_keys) < args.n:
        # fallback: 不足分をランダムに足す
        rng = random.Random(args.seed + 1)
        pool = [k for k in all_rows if k not in valid_keys]
        rng.shuffle(pool)
        valid_keys.extend(pool[: args.n - len(valid_keys)])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "batch_0001.csv"
    with open(out_path, "w", newline="") as fh:
        fh.write(",".join(header) + "\n")
        for k in valid_keys[: args.n]:
            fh.write(all_rows[k] + "\n")

    print(f"Wrote {len(valid_keys[:args.n])} positions to {out_path}", file=sys.stderr)
    print("\nSelected (match_id, end, shot_num):", file=sys.stderr)
    for k in valid_keys[: args.n]:
        print(f"  {k}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
