#!/usr/bin/env python3
"""Convert DigitalCurling3 server logs (log/<game>/shot_eXXXsYY.json) into gpw_agent
self-play JSONL records (pre-shot states with end result and game result), so that games
against external opponents (e.g. Jiritsukun-Jr) can be used for training.
Run `gpw_agent --annotate in out` afterwards to add p_hand.

  python convert_server_log.py <log_dir_or_parent> ... --out records.jsonl [--tag name]
"""
import argparse
import glob
import json
import os
import re

TEE_Y = 38.405


def load_game(d):
    files = sorted(glob.glob(os.path.join(d, "shot_e*s*.json")))
    if not files:
        return None
    shots = []
    for f in files:
        m = re.search(r"shot_e(\d+)s(\d+)\.json$", f)
        e, s = int(m.group(1)), int(m.group(2))
        j = json.load(open(f))["log"]
        shots.append((e, s, j))
    shots.sort()
    return shots


def stones_before(j):
    tr = j["trajectory"]["start"]
    out = []
    for t in ("team0", "team1"):
        lst = tr[t]
        for i in range(8):
            x = lst[i] if i < len(lst) else None
            if x is None:
                out.append(None)
            else:
                p = x.get("position", x)
                out.append([p["x"], p["y"]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="server")
    ap.add_argument("--max-end", type=int, default=10)
    args = ap.parse_args()

    game_dirs = []
    for d in args.dirs:
        if glob.glob(os.path.join(d, "shot_e*s*.json")):
            game_dirs.append(d)
        else:
            game_dirs.extend(sorted(p for p in glob.glob(os.path.join(d, "*")) if os.path.isdir(p)))
    n_games = n_rec = 0
    with open(args.out, "a") as out:
        for gi, d in enumerate(game_dirs):
            shots = load_game(d)
            if not shots:
                continue
            # Reconstruct per-end scores from the state after each end's last shot:
            # the game.dcl2 has the final result; we derive end results from the
            # "finish" stone layout of shot 15 (count stones in the house).
            ends = {}
            for e, s, j in shots:
                ends.setdefault(e, {})[s] = j
            end_result = {}   # e -> hammer result k
            hammer_of = {}
            score = {0: 0, 1: 0}
            score_before = {}
            for e in sorted(ends):
                score_before[e] = (score[0], score[1])
                last = ends[e].get(15)
                # hammer: the team that throws shot 1 (odd shots)
                j1 = ends[e].get(1) or ends[e].get(15)
                # team of shot s: shots alternate starting with the non-hammer; server log
                # does not name the thrower per shot, so infer from the first end (team1 has
                # the hammer in end 0) and the scoring rule (scorer loses the hammer).
                if e == 0:
                    hammer = 1
                else:
                    prev = end_result.get(e - 1, 0)
                    prev_h = hammer_of[e - 1]
                    hammer = prev_h if prev <= 0 else 1 - prev_h
                hammer_of[e] = hammer
                if last is None:
                    end_result[e] = 0
                    continue
                fin = last["trajectory"]["finish"]
                best = {0: None, 1: None}
                dists = []
                for t in (0, 1):
                    for x in fin[f"team{t}"]:
                        if x is None:
                            continue
                        p = x.get("position", x)
                        dd = ((p["x"]) ** 2 + (p["y"] - TEE_Y) ** 2) ** 0.5
                        if dd <= 1.829 + 0.145:
                            dists.append((dd, t))
                dists.sort()
                k = 0
                if dists:
                    t0 = dists[0][1]
                    n = 0
                    for dd, t in dists:
                        if t != t0:
                            break
                        n += 1
                    k = n if t0 == hammer else -n
                    score[t0] += n
                end_result[e] = k
            # game result
            s0, s1 = score[0], score[1]
            winner = 0 if s0 > s1 else (1 if s1 > s0 else -1)
            for e, s, j in shots:
                hammer = hammer_of[e]
                team = hammer if s % 2 == 1 else 1 - hammer
                sb = score_before[e]
                rec = {
                    "game": gi, "max_end": args.max_end, "agent": args.tag, "team": team,
                    "end": e, "shot": s, "hammer": hammer, "score": [sb[0], sb[1]],
                    "stones": stones_before(j),
                    "shot_v": [j["selected_move"]["velocity"]["x"], j["selected_move"]["velocity"]["y"]],
                    "cw": 1 if j["selected_move"].get("rotation") == "cw" else 0,
                    "label": "server", "value": 0, "det": 0, "sims": 0, "budget": 0, "used": 0,
                    "end_result_hammer": end_result[e],
                    "result": 0 if winner < 0 else (1 if winner == team else -1),
                }
                out.write(json.dumps(rec) + "\n")
                n_rec += 1
            n_games += 1
            print(f"{d}: ends={len(ends)} final {s0}-{s1} hammer_results={[end_result[e] for e in sorted(ends)]}")
    print(f"wrote {n_rec} records from {n_games} games to {args.out}")


if __name__ == "__main__":
    main()
