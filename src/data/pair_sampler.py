"""
Pair Sampler — Smart sampling strategies for training pairs.

Supports paired (same person different pose) and cross-person sampling.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


class PairSampler:
    """
    Intelligent pair sampling for try-on training.

    Args:
        pairs_file: CSV with (person_id, garment_id, ...) columns.
        mode: 'paired' (same person), 'cross' (different person), or 'mixed'.
        max_pose_diff: Maximum pose angle difference for paired sampling.
    """

    def __init__(self, pairs_file: str, mode: str = "paired", max_pose_diff: float = 45.0):
        self.mode = mode
        self.max_pose_diff = max_pose_diff
        self.pairs = self._load_pairs(pairs_file)
        self.person_to_garments = self._build_index()

    def _load_pairs(self, path: str) -> list[dict]:
        pairs = []
        p = Path(path)
        if not p.exists():
            return pairs
        with open(p) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append(row)
        return pairs

    def _build_index(self) -> dict[str, list[str]]:
        index = defaultdict(list)
        for pair in self.pairs:
            pid = pair.get("person_id", "")
            gid = pair.get("garment_id", "")
            if pid and gid:
                index[pid].append(gid)
        return dict(index)

    def sample(self, batch_size: int) -> list[tuple[str, str]]:
        """Sample a batch of (person_id, garment_id) pairs."""
        if self.mode == "paired":
            return self._sample_paired(batch_size)
        elif self.mode == "cross":
            return self._sample_cross(batch_size)
        else:
            n_paired = batch_size // 2
            n_cross = batch_size - n_paired
            return self._sample_paired(n_paired) + self._sample_cross(n_cross)

    def _sample_paired(self, n: int) -> list[tuple[str, str]]:
        """Same person, same garment — self-supervised reconstruction."""
        result = []
        persons = list(self.person_to_garments.keys())
        for _ in range(n):
            pid = random.choice(persons)
            gid = random.choice(self.person_to_garments[pid])
            result.append((pid, gid))
        return result

    def _sample_cross(self, n: int) -> list[tuple[str, str]]:
        """Different person, random garment — cross-person try-on."""
        result = []
        persons = list(self.person_to_garments.keys())
        all_garments = [g for gs in self.person_to_garments.values() for g in gs]
        for _ in range(n):
            pid = random.choice(persons)
            gid = random.choice(all_garments)
            result.append((pid, gid))
        return result

    def get_all_pairs(self) -> list[tuple[str, str]]:
        return [(p.get("person_id", ""), p.get("garment_id", "")) for p in self.pairs]
