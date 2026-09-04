#!/usr/bin/env python3
"""Print the exported model's end-result distribution for the first N records of a JSONL file.
Used to verify that the C++ inference (gpw_agent --check-model) matches PyTorch exactly.

  python check_model.py model.txt records.jsonl [--n 5] [--max-end 10]
"""
import argparse
import json

import numpy as np
import torch

from train_value import DeepSetsNet, encode, K, LOG_EPS


def load_model(path):
    with open(path) as f:
        tokens = f.read().split()
    assert tokens[0] == "gpw_value_v2"
    pos = 1
    assert tokens[pos] == "F"; F_ = int(tokens[pos + 1]); assert tokens[pos + 2] == "G"; G_ = int(tokens[pos + 3])
    assert tokens[pos + 4] == "K"; K_ = int(tokens[pos + 5]); pos += 6
    layers = {}
    for name in ["phi1", "phi2", "head1", "head2", "out"]:
        assert tokens[pos] == name, (tokens[pos], name)
        o, i = int(tokens[pos + 1]), int(tokens[pos + 2]); pos += 3
        w = np.array([float(t) for t in tokens[pos:pos + o * i]], dtype=np.float32).reshape(o, i); pos += o * i
        b = np.array([float(t) for t in tokens[pos:pos + o]], dtype=np.float32); pos += o
        layers[name] = (w, b)
    model = DeepSetsNet(h1=layers["phi1"][0].shape[0], h2=layers["phi2"][0].shape[0],
                        hh1=layers["head1"][0].shape[0], hh2=layers["head2"][0].shape[0], ns=F_, ng=G_)
    with torch.no_grad():
        for name, (w, b) in layers.items():
            getattr(model, name).weight.copy_(torch.from_numpy(w))
            getattr(model, name).bias.copy_(torch.from_numpy(b))
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("records")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-end", type=int, default=10)
    args = ap.parse_args()
    model = load_model(args.model)
    with open(args.records) as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            rec = json.loads(line)
            feats, n, g = encode(rec, rec.get("max_end", args.max_end))
            mask = np.zeros(16, dtype=np.float32); mask[:n] = 1.0
            lh = np.log(np.array(rec["p_hand"], dtype=np.float64) + LOG_EPS).astype(np.float32)
            with torch.no_grad():
                logits = model(torch.from_numpy(feats)[None], torch.from_numpy(mask)[None], torch.from_numpy(g)[None], torch.from_numpy(lh)[None])
                p = torch.softmax(logits, dim=1)[0].numpy()
            print(f"rec {i} end={rec['end']} shot={rec['shot']} target={rec.get('end_result_hammer')}: "
                  + " ".join(f"{v:.4f}" for v in p))


if __name__ == "__main__":
    main()
