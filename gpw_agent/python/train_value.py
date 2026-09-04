#!/usr/bin/env python3
"""Train the end-result model from self-play JSONL records and export it for gpw_agent.

Records come from `gpw_agent --selfplay --out FILE` (pre-shot states). The target is the
hammer team's result of the end the record belongs to (k = -4..4, 9 classes).
The network is a residual on the hand-crafted distribution p_hand stored in each record:
logits = log(p_hand + 1e-4) + net(x). Old records can be annotated with `gpw_agent --annotate in out`.

Feature layout (FEATURE_VERSION 1) must match src/nn.cpp::EncodeFeatures:
  per stone (6): x, y - TEE_Y, +1 hammer / -1 non-hammer, in_house, dist_tee, in_fgz
  global (10): r/16, r_h/8, r_n/8, count_now/4, n_house_h/8, n_house_n/8, n/16, shot<5,
               clip(diff_hammer,-6,6)/6, ends_left/10

Usage:
  python train_value.py data/*.jsonl --out model.txt [--epochs 20] [--max-end 10]
"""
import argparse
import glob
import json
import math
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_VERSION = 2
LOG_EPS = 1e-4
TEE_Y = 38.405
HOUSE_R = 1.829
STONE_R = 0.145
HOG_Y = 32.004
HALF_W = 2.375
NS, NG, K = 9, 12, 9   # v3 superset; --feat-version 2 uses prefixes 6/10
NS_V2, NG_V2 = 6, 10


def stone_feats(stones, hammer):
    """stones: list of 16 entries ([x, y] or None), index 0-7 team0, 8-15 team1."""
    rows = []
    for i, st in enumerate(stones):
        if st is None:
            continue
        x, y = float(st[0]), float(st[1])
        team = 0 if i < 8 else 1
        d = math.hypot(x, y - TEE_Y)
        in_house = 1.0 if d <= HOUSE_R + STONE_R else 0.0
        in_fgz = 1.0 if (y > HOG_Y and y < TEE_Y and in_house == 0.0 and abs(x) < HALF_W) else 0.0
        rows.append((d, [x, y - TEE_Y, 1.0 if team == hammer else -1.0, in_house, d, in_fgz, 0.0, 3.0, 0.0], team, in_house, in_fgz, x, y))
    rows.sort(key=lambda r: r[0])
    # v3 additions: covered proxy, min distance to another stone, rank
    pts = [(r[5], r[6]) for r in rows]
    for i, r in enumerate(rows):
        sx, sy = pts[i]
        cov = 0.0
        md = 3.0
        for j, (gx, gy) in enumerate(pts):
            if j == i:
                continue
            md = min(md, math.hypot(gx - sx, gy - sy))
            if gy >= sy - 0.3 or gy < HOG_Y - 1.0:
                continue
            if abs(gx - sx * (gy / sy)) < 2 * STONE_R + 0.08:
                cov = 1.0
        r[1][6] = cov
        r[1][7] = md
        r[1][8] = i / 16.0
    return rows


def encode(rec, max_end):
    hammer = rec["hammer"]
    shot = rec["shot"]
    rows = stone_feats(rec["stones"], hammer)
    n = len(rows)
    feats = np.zeros((16, NS), dtype=np.float32)
    for i, r in enumerate(rows[:16]):
        feats[i] = r[1]
    # count now
    c = 0
    if n and rows[0][3] > 0:
        t = rows[0][2]
        for r in rows:
            if r[3] == 0 or r[2] != t:
                break
            c += 1
        c = c if t == hammer else -c
    nh = sum(1 for r in rows if r[3] > 0 and r[2] == hammer)
    nn_ = sum(1 for r in rows if r[3] > 0 and r[2] != hammer)
    nh_fgz = sum(1 for r in rows if r[4] > 0 and r[2] == hammer)
    nn_fgz = sum(1 for r in rows if r[4] > 0 and r[2] != hammer)
    r_ = 16 - shot
    r_h = (r_ + 1) // 2
    r_n = r_ - r_h
    s0, s1 = rec["score"]
    diff_h = (s0 - s1) if hammer == 0 else (s1 - s0)
    ends_left = (max_end - rec["end"]) if rec["end"] < max_end else 0
    g = np.array([
        r_ / 16.0, r_h / 8.0, r_n / 8.0, c / 4.0, nh / 8.0, nn_ / 8.0, n / 16.0,
        1.0 if shot < 5 else 0.0, max(-6, min(6, diff_h)) / 6.0, ends_left / 10.0,
        nh_fgz / 8.0, nn_fgz / 8.0,
    ], dtype=np.float32)
    return feats, n, g


class DeepSetsNet(nn.Module):
    def __init__(self, h1=64, h2=64, hh1=128, hh2=64, ns=NS, ng=NG):
        super().__init__()
        self.ns, self.ng = ns, ng
        self.phi1 = nn.Linear(ns, h1)
        self.phi2 = nn.Linear(h1, h2)
        self.head1 = nn.Linear(2 * h2 + ng, hh1)
        self.head2 = nn.Linear(hh1, hh2)
        self.out = nn.Linear(hh2, K)
        nn.init.zeros_(self.out.weight)   # start exactly at the hand-crafted distribution
        nn.init.zeros_(self.out.bias)

    def forward(self, stones, mask, g, log_hand):
        # stones: B x 16 x NS, mask: B x 16 (1 = present), g: B x NG, log_hand: B x K
        stones = stones[:, :, :self.ns]
        g = g[:, :self.ng]
        h = F.relu(self.phi1(stones))
        h = F.relu(self.phi2(h))
        m = mask.unsqueeze(-1)
        s = (h * m).sum(1)
        mx = (h * m + (m - 1.0) * 1e9).max(1).values
        mx = torch.where(mask.sum(1, keepdim=True) > 0, mx, torch.zeros_like(mx))
        x = torch.cat([s, mx, g], dim=1)
        x = F.relu(self.head1(x))
        x = F.relu(self.head2(x))
        return self.out(x) + log_hand


def export(model, path):
    with open(path, "w") as f:
        f.write("gpw_value_v2\n")
        f.write(f"F {model.ns} G {model.ng} K {K}\n")
        for name in ["phi1", "phi2", "head1", "head2", "out"]:
            lin = getattr(model, name)
            w = lin.weight.detach().cpu().numpy()
            b = lin.bias.detach().cpu().numpy()
            f.write(f"{name} {w.shape[0]} {w.shape[1]}\n")
            f.write(" ".join(f"{v:.6g}" for v in w.reshape(-1)) + "\n")
            f.write(" ".join(f"{v:.6g}" for v in b.reshape(-1)) + "\n")


def load_records(paths, max_end, min_shot=0):
    X, M, G, Y, GID, LH = [], [], [], [], [], []
    hist = np.zeros(K, dtype=np.int64)
    n_files = 0
    for p in paths:
        n_files += 1
        with open(p) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec["shot"] < min_shot:
                    continue
                if "p_hand" not in rec:
                    raise SystemExit(f"{p}: record without p_hand; run `gpw_agent --annotate in out` first")
                k = max(-4, min(4, int(rec.get("end_result_hammer", 0))))
                feats, n, g = encode(rec, max_end)
                mask = np.zeros(16, dtype=np.float32)
                mask[:n] = 1.0
                X.append(feats); M.append(mask); G.append(g); Y.append(k + 4)
                LH.append(np.log(np.array(rec["p_hand"], dtype=np.float64) + LOG_EPS).astype(np.float32))
                GID.append(hash((p, rec["game"])) & 0xFFFFFFFF)
                hist[k + 4] += 1
    return (np.stack(X), np.stack(M), np.stack(G), np.array(Y, dtype=np.int64), np.array(GID), np.stack(LH)), hist, n_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="model.txt")
    ap.add_argument("--eval-out", default="", help="write an EvalParams file with the empirical end_dist")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-end", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--feat-version", type=int, default=3, choices=[2, 3])
    ap.add_argument("--h1", type=int, default=64)
    ap.add_argument("--h2", type=int, default=64)
    ap.add_argument("--hh1", type=int, default=128)
    ap.add_argument("--hh2", type=int, default=64)
    args = ap.parse_args()

    paths = []
    for pat in args.inputs:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        print("no input files", file=sys.stderr)
        sys.exit(1)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    (X, M, G, Y, GID, LH), hist, n_files = load_records(paths, args.max_end)
    print(f"loaded {len(Y)} records from {n_files} files in {time.time() - t0:.1f}s")
    dist = hist / max(1, hist.sum())
    print("hammer end-result distribution k=-4..4:", " ".join(f"{v:.3f}" for v in dist))

    # split by game id
    games = np.unique(GID)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(games)
    n_val = max(1, int(len(games) * args.val_frac))
    val_games = set(games[:n_val].tolist())
    is_val = np.array([g in val_games for g in GID])
    tr = ~is_val
    print(f"train {tr.sum()} / val {is_val.sum()} records ({len(games)} games)")

    def to_t(a):
        return torch.from_numpy(a)

    Xtr, Mtr, Gtr, Ytr, Ltr = to_t(X[tr]), to_t(M[tr]), to_t(G[tr]), to_t(Y[tr]), to_t(LH[tr])
    Xva, Mva, Gva, Yva, Lva = to_t(X[is_val]), to_t(M[is_val]), to_t(G[is_val]), to_t(Y[is_val]), to_t(LH[is_val])
    hand_ce = F.cross_entropy(Lva, Yva).item()
    print(f"hand-crafted val_ce={hand_ce:.4f} (the model must beat this)")

    ns, ng = (NS_V2, NG_V2) if args.feat_version == 2 else (NS, NG)
    model = DeepSetsNet(h1=args.h1, h2=args.h2, hh1=args.hh1, hh2=args.hh2, ns=ns, ng=ng)
    print(f"model sizes phi {args.h1}/{args.h2} head {args.hh1}/{args.hh2}, params {sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n = len(Ytr)
    best_val = 1e9
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xs, ms, gs, ys, ls = Xtr[idx].clone(), Mtr[idx], Gtr[idx], Ytr[idx], Ltr[idx]
            # mirror augmentation: flip x for a random half of the batch
            flip = (torch.rand(len(idx)) < 0.5).float().unsqueeze(-1)
            xs[:, :, 0] = xs[:, :, 0] * (1.0 - 2.0 * flip)
            logits = model(xs, ms, gs, ls)
            loss = F.cross_entropy(logits, ys)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            cnt += len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            vlog = model(Xva, Mva, Gva, Lva)
            vl = F.cross_entropy(vlog, Yva).item()
            acc = (vlog.argmax(1) == Yva).float().mean().item()
            # baseline: predict the marginal distribution
            base = -np.sum(dist * np.log(np.maximum(dist, 1e-9)))
        print(f"epoch {ep + 1}/{args.epochs} train_ce={tot / cnt:.4f} val_ce={vl:.4f} val_acc={acc:.3f} (hand ce={hand_ce:.4f}, marginal ce={base:.4f})")
        if vl < best_val:
            best_val = vl
            export(model, args.out)
    print(f"exported best model (val_ce={best_val:.4f}) to {args.out}")

    if args.eval_out:
        with open(args.eval_out, "w") as f:
            f.write("# empirical hammer end-result distribution from self-play\n")
            f.write("end_dist " + " ".join(f"{v:.4f}" for v in dist) + "\n")
            f.write(f"model {args.out}\n")
        print(f"wrote eval params to {args.eval_out}")


if __name__ == "__main__":
    main()
