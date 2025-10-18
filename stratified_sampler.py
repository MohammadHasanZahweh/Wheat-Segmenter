from __future__ import annotations

import numpy as np
import rasterio
from typing import Iterable, Sized, cast
from torch.utils.data import Sampler, Subset, Dataset
import random


class StratifiedRandomSubset(Sampler[int]):
    """
    Stratified random sampler over a dataset (or Subset) that aims to
    sample a given fraction of items while roughly preserving:
      - per-region proportions, and
      - coverage spectrum of the wheat label (via quantile bins).

    Assumptions about dataset items:
      - The base dataset has an attribute `index` list where `index[i]` is a
        dict containing keys: 'label_path' (str) and 'region' (str).
      - This matches the `WheatTilesDataset` in wheat_segmenter.py.

    Parameters
    - dataset: Dataset or Subset wrapping a dataset with the structure above
    - fraction: fraction of samples to pick (0, 1]
    - n_bins: number of quantile bins for wheat coverage stratification
    - seed: RNG seed for reproducibility
    """

    def __init__(self, dataset: Dataset, fraction: float = 0.01, n_bins: int = 5, seed: int | None = 42):
        if not (0.0 < fraction <= 1.0):
            raise ValueError("fraction must be in (0,1].")
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1")

        self.dataset = dataset
        self.fraction = fraction
        self.n_bins = n_bins
        self.rng = random.Random(seed)

        # Helper to map subset index -> base dataset and base index
        def base_and_index(idx: int):
            ds = self.dataset
            if isinstance(ds, Subset):
                base = ds.dataset
                base_idx = int(ds.indices[idx])
            else:
                base = ds
                base_idx = idx
            return base, base_idx

        # Require a map-style dataset with __len__
        n = len(cast(Sized, self.dataset))
        regions: list[str] = []
        label_paths: list[str] = []
        for i in range(n):
            base, bi = base_and_index(i)
            rec = getattr(base, "index")[bi]
            label_paths.append(rec["label_path"])  # type: ignore[index]
            regions.append(rec["region"])          # type: ignore[index]

        # Compute wheat coverage ratios per item from labels
        ratios: list[float] = []
        for lp in label_paths:
            with rasterio.open(lp) as ds:
                lab = ds.read(out_dtype="float32")  # [2,H,W]
            valid = lab[0] > 0.5
            wheat = lab[1] > 0.5
            denom = float(valid.sum())
            ratios.append(float((wheat & valid).sum()) / denom if denom > 0 else 0.0)

        # Global quantile bin edges
        if self.n_bins == 1:
            edges = [0.0, 1.0]
        else:
            qs = np.linspace(0, 1, self.n_bins + 1)
            edges = np.quantile(np.asarray(ratios, dtype=np.float32), qs).tolist()
            for j in range(1, len(edges)):
                if edges[j] < edges[j - 1]:
                    edges[j] = edges[j - 1]

        def bin_id(x: float) -> int:
            
            if self.n_bins == 1:
                return 0
            for b in range(len(edges) - 1):
                lo, hi = edges[b], edges[b + 1]
                if b < len(edges) - 2:
                    if lo <= x < hi:
                        return b
                else:
                    if lo <= x <= hi:
                        return b
            return len(edges) - 2

        # Group indices by (region, bin)
        from collections import defaultdict

        group_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
        region_counts: dict[str, int] = defaultdict(int)
        for i in range(n):
            r = regions[i]
            b = bin_id(ratios[i])
            group_indices[(r, b)].append(i)
            region_counts[r] += 1

        total_target = max(1, round(self.fraction * n))

        # Regional proportional allocation
        all_regions = list(region_counts.keys())
        reg_sizes = np.array([region_counts[r] for r in all_regions], dtype=np.float64)
        reg_weights = reg_sizes / reg_sizes.sum() if reg_sizes.sum() > 0 else np.ones_like(reg_sizes) / max(1, len(reg_sizes))
        reg_alloc = np.floor(reg_weights * total_target).astype(int)
        residual = total_target - int(reg_alloc.sum())
        frac_parts = (reg_weights * total_target) - reg_alloc
        order = np.argsort(-frac_parts)
        for k in range(residual):
            if len(all_regions) == 0:
                break
            reg_alloc[order[k % len(all_regions)]] += 1
        region_target = {r: int(reg_alloc[i]) for i, r in enumerate(all_regions)}
        if total_target >= len(all_regions):
            for r in all_regions:
                if region_target[r] == 0 and region_counts[r] > 0:
                    region_target[r] = 1

        # Within region: allocate across bins proportionally and sample
        chosen: list[int] = []
        for r in all_regions:
            target_r = min(region_target[r], region_counts[r])
            bins_r = [(b, group_indices[(r, b)]) for b in range(self.n_bins) if len(group_indices[(r, b)]) > 0]
            if not bins_r:
                continue
            counts = np.array([len(lst) for _, lst in bins_r], dtype=np.float64)
            weights = counts / counts.sum()
            alloc = np.floor(weights * target_r).astype(int)
            rem = target_r - int(alloc.sum())
            parts = (weights * target_r) - alloc
            ord2 = np.argsort(-parts)
            for k in range(rem):
                if len(bins_r) == 0:
                    break
                alloc[ord2[k % len(bins_r)]] += 1
            for (bin_key, lst), take in zip(bins_r, alloc.tolist()):
                if take <= 0:
                    continue
                take = min(take, len(lst))
                temp = lst.copy()
                self.rng.shuffle(temp)
                chosen.extend(temp[:take])

        # Adjust to exact target size if needed
        chosen = list(dict.fromkeys(chosen))
        if len(chosen) < total_target:
            remaining = [i for i in range(n) if i not in set(chosen)]
            self.rng.shuffle(remaining)
            needed = total_target - len(chosen)
            chosen.extend(remaining[:needed])
        elif len(chosen) > total_target:
            self.rng.shuffle(chosen)
            chosen = chosen[:total_target]

        self.indices = chosen

    def __iter__(self) -> Iterable[int]:
        idxs = self.indices.copy()
        self.rng.shuffle(idxs)
        return iter(idxs)

    def __len__(self) -> int:
        return len(self.indices)


def make_one_percent_sampler(dataset: Dataset, seed: int | None = 42) -> Sampler[int]:
    return StratifiedRandomSubset(dataset, fraction=0.01, n_bins=5, seed=seed)
