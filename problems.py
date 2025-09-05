from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


def _row_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim == 1:
        n = float(np.linalg.norm(x))
        return x / (n + eps)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def _bits_from_traits(t: np.ndarray) -> np.ndarray:
    return (t > 0).astype(np.int8)


@dataclass
class NKProblem:
    n_bits: int
    K: int = 8
    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(100 + self.n_bits)
        neighbors, tables = [], []
        for i in range(self.n_bits):
            others = np.delete(np.arange(self.n_bits), i)
            neigh = rng.choice(others, size=self.K, replace=False)
            neighbors.append(np.concatenate(([i], neigh)))
            tables.append(rng.random(2 ** (self.K + 1)))
        self.neighbors = neighbors
        self.tables = tables

    @property
    def trait_dim(self) -> int:
        return self.n_bits

    def fitness_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        neighbors, tables = self.neighbors, self.tables

        def _nk_fitness(traits: np.ndarray) -> np.ndarray:
            bits = _bits_from_traits(traits)
            pop, n = bits.shape
            K = len(neighbors[0]) - 1
            vals = np.zeros(pop)
            powers = (1 << np.arange(K + 1))
            for i in range(n):
                sub = bits[:, neighbors[i]]
                keys = (sub * powers).sum(axis=1).astype(int)
                vals += tables[i][keys]
            return vals / n

        return _nk_fitness


@dataclass
class Trap5Problem:
    n_bits: int

    def __post_init__(self):
        self.blocks = [list(range(i, min(i + 5, self.n_bits))) for i in range(0, self.n_bits, 5)]

    @property
    def trait_dim(self) -> int:
        return self.n_bits

    def fitness_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        blocks = self.blocks

        def _score(traits: np.ndarray) -> np.ndarray:
            bits = (traits > 0).astype(np.int8)
            out = np.empty(bits.shape[0], dtype=float)
            for i in range(bits.shape[0]):
                total = 0.0
                for blk in blocks:
                    u = int(bits[i, blk].sum())
                    total += 5.0 if u == 5 else (4.0 - u)
                out[i] = total / (5.0 * len(blocks))
            return out

        return _score


def make_problem(task_label: str):
    if task_label.startswith("nk_"):
        n_bits = int(task_label.split("_")[1])
        prob = NKProblem(n_bits=n_bits, K=8)
        return prob.trait_dim, prob.fitness_fn()
    elif task_label.startswith("trap_"):
        n_bits = int(task_label.split("_")[1])
        prob = Trap5Problem(n_bits=n_bits)
        return prob.trait_dim, prob.fitness_fn()
    else:
        raise ValueError("Unknown task label")


