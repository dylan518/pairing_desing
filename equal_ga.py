from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np


# ----- Internal helpers (kept local to avoid circular deps) -----
def _row_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim == 1:
        n = float(np.linalg.norm(x))
        return x / (n + eps)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def _sample_matching_softmax(U: np.ndarray, tau: float, rng: np.random.Generator) -> np.ndarray:
    n = U.shape[0]
    iu, ju = np.triu_indices(n, 1)
    logits = U[iu, ju].copy()
    logits[~np.isfinite(logits)] = -1e9
    w = np.exp((logits - logits.max()) / max(tau, 1e-12))
    available = np.ones(n, dtype=bool)
    valid = np.isfinite(U[iu, ju])
    pairs = []
    need = n // 2
    while need > 0:
        mask = valid & available[iu] & available[ju]
        if not np.any(mask):
            rem = np.where(available)[0]
            rng.shuffle(rem)
            for a, b in zip(rem[0::2], rem[1::2]):
                pairs.append((int(a), int(b)))
            break
        ww = np.zeros_like(w, dtype=float)
        ww[mask] = w[mask]
        s = ww.sum()
        if s <= 0 or not np.isfinite(s):
            idxs = np.where(mask)[0]
            pick = int(rng.choice(idxs))
        else:
            pick = int(rng.choice(len(ww), p=ww / s))
        i, j = int(iu[pick]), int(ju[pick])
        if available[i] and available[j]:
            pairs.append((i, j))
            available[i] = available[j] = False
            need -= 1
        valid[pick] = False
    return np.array(pairs, dtype=int)


def _mutual_utility(seek: np.ndarray, provide: np.ndarray) -> np.ndarray:
    sN = _row_norm(seek)
    pN = _row_norm(provide)
    M = pN @ sN.T
    U = 0.5 * (M + M.T)
    np.fill_diagonal(U, -np.inf)
    return U


def _blend(v1: np.ndarray, v2: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    a = np.clip(rng.normal(0.5, 0.1), 0.0, 1.0)
    return a * v1 + (1 - a) * v2 + rng.normal(0, sigma, size=v1.shape)


# ----- Public API -----
@dataclass
class FWSMConfigEq:
    pop_size: int = 60
    gens: int = 40
    trait_dim: int = 48
    embed_dim: int = 3
    tau: float = 0.7
    sigma_traits: float = 0.22
    sigma_embed: float = 0.05
    child_budget: Optional[int] = None
    seed: int = 0


class FWSMGAEqualOffspring:
    def __init__(self, cfg: FWSMConfigEq):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.history: Dict[str, list] = {"best": [], "mean": []}

    def fit(self, fitness_fn: Callable[[np.ndarray], np.ndarray]) -> Dict[str, np.ndarray]:
        N, D, k = self.cfg.pop_size, self.cfg.trait_dim, self.cfg.embed_dim
        B = self.cfg.child_budget or N
        children_per_pair = int(2 * B / N)  # with B=N → 2 per pair

        t = self.rng.normal(0, 1.0, size=(N, D))
        if k > 0:
            s = _row_norm(self.rng.normal(0, 1.0, size=(N, k)))
            p = _row_norm(self.rng.normal(0, 1.0, size=(N, k)))
        else:
            s = np.zeros((N, 1)); p = np.zeros((N, 1))
        f = fitness_fn(t)
        self.history["best"].append(float(f.max()))
        self.history["mean"].append(float(f.mean()))

        for _ in range(self.cfg.gens):
            if k > 0:
                U = _mutual_utility(s, p)
                pairs = _sample_matching_softmax(U, self.cfg.tau, self.rng)
            else:
                idx = np.arange(N)
                self.rng.shuffle(idx)
                pairs = idx.reshape(-1, 2)

            kids_t = []
            for (i, j) in pairs:
                for _ in range(children_per_pair):
                    c_t = _blend(t[i], t[j], self.cfg.sigma_traits, self.rng)
                    kids_t.append(c_t)

            kids_t = np.vstack(kids_t)
            f_kids = fitness_fn(kids_t)
            t_all = np.vstack([t, kids_t])
            f_all = np.hstack([f, f_kids])
            keep = np.argsort(f_all)[-N:]
            t, f = t_all[keep], f_all[keep]
            self.history["best"].append(float(f.max()))
            self.history["mean"].append(float(f.mean()))

        return {k: np.asarray(v) for k, v in self.history.items()}


