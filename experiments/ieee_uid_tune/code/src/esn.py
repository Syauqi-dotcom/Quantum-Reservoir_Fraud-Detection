"""
Classical Echo State Network - the control reservoir.

Same sequence input as the QRC, same fixed-reservoir / trained-linear-readout
discipline.  This answers the single most important question in the review:
"is the *quantum* reservoir doing anything a cheap classical reservoir can't?"
"""
from __future__ import annotations

import numpy as np


class EchoStateNetwork:
    def __init__(self, cfg_esn: dict, input_dim: int, seed: int):
        rng = np.random.default_rng(seed)
        n = int(cfg_esn["reservoir_size"])
        self.n = n
        self.a = float(cfg_esn["leak_rate"])
        self.Win = (rng.uniform(-1, 1, size=(n, input_dim))
                    * float(cfg_esn["input_scaling"]))
        W = rng.uniform(-1, 1, size=(n, n))
        mask = rng.random((n, n)) < float(cfg_esn["sparsity"])
        W *= mask
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        if radius > 0:
            W *= float(cfg_esn["spectral_radius"]) / radius
        self.W = W
        self.embed_dim = 2 * n + input_dim

    def embed(self, seq: np.ndarray, mask: np.ndarray) -> np.ndarray:
        N, K, F = seq.shape
        out = np.zeros((N, self.embed_dim), dtype=np.float32)
        for a in range(N):
            h = np.zeros(self.n)
            states = []
            last_x = np.zeros(F)
            for t in range(K):
                if mask[a, t] == 0.0:
                    continue
                x = seq[a, t]
                pre = self.Win @ x + self.W @ h
                h = (1 - self.a) * h + self.a * np.tanh(pre)
                states.append(h.copy())
                last_x = x
            S = np.array(states) if states else np.zeros((1, self.n))
            out[a] = np.concatenate([S[-1], S.mean(axis=0), last_x])
        return out
