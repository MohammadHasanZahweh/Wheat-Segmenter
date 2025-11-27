"""
Consolidated dataset and sampler for wheat segmentation.

This module combines:
- WheatTilesDataset: Multi-temporal Sentinel-2 tile dataset for wheat segmentation
- StratifiedRandomSubset: Stratified sampler for balanced tile selection
- Utility functions for dataset summarization and tile selection

Extracted from wheat_segmenter.py and stratified_sampler.py
"""

from __future__ import annotations

import glob
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Sized, Tuple, cast

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, Sampler, Subset


def load_meta_stats(meta_dir: Path, year: str, months: Sequence[int]) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load precomputed mean/std statistics from meta directory.
    
    Expected structure:
        meta/<YEAR>_<MONTH>.npz containing 'mean' and 'std' arrays
    
    Args:
        meta_dir: Path to meta directory
        year: Year string (e.g., '2020')
        months: Sequence of month integers to load
    
    Returns:
        Dict mapping month -> {'mean': np.ndarray, 'std': np.ndarray}
        Each array has shape (num_bands,)
    
    Example:
        >>> stats = load_meta_stats(Path('./meta'), '2020', [11, 12, 1])
        >>> stats[11]['mean']  # array of shape (13,) with mean for each band in Nov
    """
    meta_stats = {}
    for month in months:
        npz_path = meta_dir / f"{year}_{month}.npz"
        if not npz_path.exists():
            npz_path = meta_dir / f"{int(year)-1}_{month}.npz"
            if not npz_path.exists():
                print(f"[WARN] Meta stats not found: {npz_path}, skipping month {month}")
                continue
        data = np.load(npz_path)
        if 'mean' not in data or 'std' not in data:
            print(f"[WARN] Meta file {npz_path} missing 'mean' or 'std', skipping")
            continue
        meta_stats[month] = {
            'mean': data['mean'].astype(np.float32),
            'std': data['std'].astype(np.float32)
        }
    return meta_stats


class WheatTilesDataset(Dataset):
    """
    Multi-temporal Sentinel-2 dataset for wheat segmentation.
    
    Structure:
        data/<YEAR>/<REGION>/<MONTH>/<TILE_ID>.tif   # ~11–13 bands, 64×64 (usually)
        label/<YEAR>/<REGION>/<TILE_ID>.tif          # 2 layers: [0]=valid, [1]=wheat
        meta/<YEAR>_<MONTH>.npz                      # Optional: precomputed mean/std per month

    LAZY version:
      - Index is built from filenames only (fast).
      - Only a small probe (limit) is used to infer bands & size.
      - Size/band fixes happen at read-time (pad/trim), not during __init__.
    
    Normalization Options:
      1. Per-tile min-max (default): band_stats=None, meta_dir=None
         Each tile normalized independently to [0,1] per band
      
      2. Meta statistics (recommended): band_stats='auto', meta_dir='./meta'
         Uses precomputed mean/std from meta/<year>_<month>.npz files
         Applies z-score normalization: (x - mean) / std per month and band
      
      3. Custom statistics: band_stats={band: (mean, std), ...}
         Provide your own normalization parameters
    
    Example Usage:
        # With meta stats (recommended for training)
        >>> ds = WheatTilesDataset(
        ...     root_preprocessed="./preprocessed_data",
        ...     year="2020",
        ...     normalize=True,
        ...     band_stats='auto',
        ...     meta_dir='./meta'
        ... )
        
        # Without meta stats (per-tile normalization)
        >>> ds = WheatTilesDataset(
        ...     root_preprocessed="./preprocessed_data",
        ...     year="2020",
        ...     normalize=True,
        ...     band_stats=None
        ... )
    """
    
    def __init__(
        self,
        root_preprocessed: str,
        year: str = "2020",
        regions=None,                           # e.g., ["0","1","2","3","4"] or None for all
        month_order=(11, 12, 1, 2, 3, 4, 5, 6, 7),
        temporal_layout=False,                  # True -> [T,B,64,64]; False -> [C,64,64]
        normalize=True,
        band_stats=None,                        # None, 'auto', {band:(mean,std)}, or {(t,b):(mean,std)}
        meta_dir: str | None = None,            # Path to meta directory containing <year>_<month>.npz files
        require_complete=True,                  # only keep tiles with ALL months present
        # band & size handling
        target_bands: int | None = None,        # None => probe few files to detect modal count
        target_size: tuple[int, int] | None = (64, 64),  # None => probe few files to infer
        size_policy: str = "pad",               # "pad" center pad/crop at read-time (recommended)
        probe_limit: int = 20                   # how many samples to open when probing
    ):
        self.root = Path(root_preprocessed)
        self.year = str(year)
        self.DATA = self.root / "data" / self.year
        self.LABEL = self.root / "label" / self.year

        self.months = tuple(month_order)
        self.temporal_layout = temporal_layout
        self.normalize = normalize
        self.require_complete = require_complete
        self.size_policy = size_policy
        
        # Load meta statistics if provided
        self.meta_stats = None
        if meta_dir is not None:
            meta_path = Path(meta_dir)
            if meta_path.exists():
                self.meta_stats = load_meta_stats(meta_path, self.year, self.months)
                if self.meta_stats:
                    print(f"[INFO] Loaded meta stats for {len(self.meta_stats)} months from {meta_path}")
            else:
                print(f"[WARN] meta_dir specified but not found: {meta_path}")
        
        # Handle band_stats: 'auto' uses meta_stats if available
        if band_stats == 'auto':
            if self.meta_stats:
                self.band_stats = 'meta'  # marker to use self.meta_stats in _normalize
                print("[INFO] Using 'auto' normalization with meta statistics")
            else:
                self.band_stats = None
                print("[WARN] band_stats='auto' but no meta_stats loaded, using per-tile min-max")
        else:
            self.band_stats = band_stats

        # Regions (filenames only)
        all_regions = sorted([p.name for p in self.DATA.iterdir() if p.is_dir()])
        self.regions = all_regions if regions is None else [r for r in regions if (self.DATA / r).exists()]

        # Build index from labels (filenames only)
        self.index = self._build_index()
        if not self.index:
            raise RuntimeError("No tiles found. Check ROOT/YEAR/regions structure.")

        # Probe a FEW files to infer bands/size if needed (fast)
        self._probe_bands_size(target_bands, target_size, probe_limit)

        # Sanity: labels should have 2 layers (open ONE label only)
        with rasterio.open(self.index[0]["label_path"]) as dsl:
            if dsl.count != 2:
                raise RuntimeError("Labels must have 2 layers (valid, wheat).")

    # ---------- helpers ----------
    def _build_index(self):
        idx = []
        for region in self.regions:
            label_dir = self.LABEL / region
            if not label_dir.exists():
                print(f"[WARN] missing label dir: {label_dir}")
                continue
            for lab_fp in sorted(glob.glob(str(label_dir / "*.tif"))):
                tile_id = Path(lab_fp).stem
                month_paths = {}
                complete = True
                for m in self.months:
                    m_fp = self.DATA / region / str(m) / f"{tile_id}.tif"
                    if m_fp.exists():
                        month_paths[m] = str(m_fp)
                    else:
                        complete = False
                if self.require_complete and not complete:
                    continue
                if not self.require_complete and len(month_paths) == 0:
                    continue
                idx.append({
                    "region": region,
                    "tile_id": tile_id,
                    "label_path": str(lab_fp),
                    "month_paths": month_paths
                })
        return idx

    def _probe_bands_size(self, target_bands, target_size, limit):
        # Decide bands
        if target_bands is None:
            counts = Counter()
            seen = 0
            for rec in self.index:
                for m in self.months:
                    p = rec["month_paths"].get(m)
                    if p:
                        with rasterio.open(p) as ds:
                            counts[ds.count] += 1
                        seen += 1
                        break
                if seen >= limit:
                    break
            if not counts:
                raise RuntimeError("Could not detect band counts.")
            self.num_bands = counts.most_common(1)[0][0]
        else:
            self.num_bands = int(target_bands)

        # Decide size
        if target_size is None:
            sizes = Counter()
            seen = 0
            for rec in self.index:
                for m in self.months:
                    p = rec["month_paths"].get(m)
                    if p:
                        with rasterio.open(p) as ds:
                            sizes[(ds.height, ds.width)] += 1
                        seen += 1
                        break
                if seen >= limit:
                    break
            if not sizes:
                raise RuntimeError("Could not infer tile size.")
            self.H, self.W = sizes.most_common(1)[0][0]
        else:
            self.H, self.W = target_size

    def __len__(self):
        return len(self.index)

    def _fix_band_count(self, arr):
        B, H, W = arr.shape
        tb = self.num_bands
        if B == tb:
            return arr
        if B > tb:
            return arr[:tb]
        pad = np.zeros((tb - B, H, W), dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)

    def _fix_size(self, arr):
        # Always pad/crop CENTER to target (size_policy="pad")
        H, W = arr.shape[1:]
        th, tw = self.H, self.W
        if (H, W) == (th, tw):
            return arr
        out = np.zeros((arr.shape[0], th, tw), dtype=np.float32)
        h = min(H, th)
        w = min(W, tw)
        sy = (H - h) // 2 if H > h else 0
        sx = (W - w) // 2 if W > w else 0
        dy = (th - h) // 2 if th > h else 0
        dx = (tw - w) // 2 if tw > w else 0
        out[:, dy:dy+h, dx:dx+w] = arr[:, sy:sy+h, sx:sx+w]
        return out

    def _normalize(self, arrTBHW):
        T, B, H, W = arrTBHW.shape
        out = arrTBHW.copy()
        
        # Use meta statistics (per-month normalization)
        if self.band_stats == 'meta' and self.meta_stats:
            for t, month in enumerate(self.months):
                if month not in self.meta_stats:
                    continue  # skip months without stats
                mean = self.meta_stats[month]['mean']
                std = self.meta_stats[month]['std']
                # Ensure we don't exceed available bands
                num_bands_to_norm = min(B, len(mean))
                for b in range(num_bands_to_norm):
                    s = std[b] if std[b] > 0 else 1.0
                    out[t, b] = (out[t, b] - mean[b]) / s
            return out
        
        if self.band_stats is None:
            # per-tile min-max per band across time
            for b in range(B):
                band = out[:, b]
                vmin = np.nanmin(band)
                vmax = np.nanmax(band)
                out[:, b] = 0.0 if vmax <= vmin else (band - vmin) / (vmax - vmin)
            return out
        
        keyed_tb = any(isinstance(k, tuple) and len(k) == 2 for k in self.band_stats.keys())
        if keyed_tb:
            for t in range(T):
                for b in range(B):
                    mean, std = self.band_stats.get((t, b), (0.0, 1.0))
                    if std == 0:
                        std = 1.0
                    out[t, b] = (out[t, b] - mean) / std
        else:
            for b in range(B):
                mean, std = self.band_stats.get(b, (0.0, 1.0))
                if std == 0:
                    std = 1.0
                out[:, b] = (out[:, b] - mean) / std
        return out

    def _read_stack(self, month_paths):
        imgs = []
        for m in self.months:
            if m not in month_paths:
                arr = np.zeros((self.num_bands, self.H, self.W), dtype=np.float32)
            else:
                with rasterio.open(month_paths[m]) as ds:
                    arr = ds.read(out_dtype="float32")     # [B,H,W]
                arr = self._fix_band_count(arr)
                arr = self._fix_size(arr)
            imgs.append(arr)
        arrTBHW = np.stack(imgs, axis=0)                    # [T,B,H,W]
        if self.normalize:
            arrTBHW = self._normalize(arrTBHW)
        if self.temporal_layout:
            return arrTBHW
        T, B, H, W = arrTBHW.shape
        return arrTBHW.reshape(T * B, H, W)                 # [C,H,W]

    def _read_labels(self, label_path):
        with rasterio.open(label_path) as ds:
            lab = ds.read(out_dtype="float32")              # [2,H,W]
        lab = np.clip(lab, 0, 1)
        lab = self._fix_size(lab)
        return lab[0:1], lab[1:2]

    def __getitem__(self, i):
        rec = self.index[i]
        x = self._read_stack(rec["month_paths"])
        valid, wheat = self._read_labels(rec["label_path"])
        x = np.nan_to_num(x, nan=0.0)
        valid = np.nan_to_num(valid, nan=0.0)
        wheat = np.nan_to_num(wheat, nan=0.0)
        return {
            "x": torch.from_numpy(x),
            "valid_mask": torch.from_numpy(valid),
            "wheat_mask": torch.from_numpy(wheat),
            "tile_id": rec["tile_id"],
            "region": rec["region"]
        }


class StratifiedRandomSubset(Sampler[int]):
    """
    Stratified random sampler over a dataset (or Subset) that aims to
    sample a given fraction of items while roughly preserving:
      - per-region proportions, and
      - coverage spectrum of the wheat label (via quantile bins).

    Assumptions about dataset items:
      - The base dataset has an attribute `index` list where `index[i]` is a
        dict containing keys: 'label_path' (str) and 'region' (str).
      - This matches the `WheatTilesDataset` structure.

    Parameters
    - dataset: Dataset or Subset wrapping a dataset with the structure above
    - fraction: fraction of samples to pick (0, 1]
    - n_bins: number of quantile bins for wheat coverage stratification
    - seed: RNG seed for reproducibility
    """

    def __init__(
        self,
        dataset: Dataset,
        fraction: float = 0.01,
        n_bins: int = 5,
        seed: int | None = 42
    ):
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
        reg_weights = (
            reg_sizes / reg_sizes.sum()
            if reg_sizes.sum() > 0
            else np.ones_like(reg_sizes) / max(1, len(reg_sizes))
        )
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
            bins_r = [
                (b, group_indices[(r, b)])
                for b in range(self.n_bins)
                if len(group_indices[(r, b)]) > 0
            ]
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
    """Create a stratified sampler for 1% of the dataset."""
    return StratifiedRandomSubset(dataset, fraction=0.01, n_bins=5, seed=seed)


# ---- Utility and CLI helpers ----
def tiles_in(folder: Path):
    """Return set of tile IDs (stems) from .tif files in folder."""
    return set(Path(fp).stem for fp in glob.glob(str(folder / "*.tif")))


def _assert_root_year(root: str, year: str):
    """Validate that data/ and label/ directories exist."""
    data_root = Path(root) / "data" / str(year)
    label_root = Path(root) / "label" / str(year)
    if not data_root.exists() or not label_root.exists():
        raise FileNotFoundError(
            f"Expected directories not found. Check: '{data_root}' and '{label_root}'. "
            f"Pass --root to the folder that contains 'data' and 'label'."
        )
    return data_root, label_root


def summarize_dataset(root: str, year: str, regions, months):
    """Print dataset summary statistics."""
    data_root, label_root = _assert_root_year(root, year)
    regions_eff = regions or sorted([p.name for p in data_root.iterdir() if p.is_dir()])
    print("Regions:", regions_eff)
    for r in regions_eff:
        print(f"\n== Region {r} ==")
        month_sets = {}
        for m in months:
            mdir = data_root / r / str(m)
            month_sets[m] = tiles_in(mdir) if mdir.exists() else set()
        inter = set.intersection(*[s for s in month_sets.values()]) if month_sets else set()
        union = set.union(*[s for s in month_sets.values()]) if month_sets else set()
        print(" tiles present in ALL months:", len(inter))
        print(" tiles in ANY month:", len(union))
        sample = next(iter(glob.iglob(str(data_root / r / str(months[0]) / "*.tif"))), None)
        if sample:
            with rasterio.open(sample) as ds:
                print(" image bands:", ds.count, "| size:", (ds.height, ds.width))
        # basic label shape check
        label_dir = label_root / r
        two_ok = True
        for lf in list(glob.iglob(str(label_dir / "*.tif")))[:10]:
            with rasterio.open(lf) as dsl:
                if dsl.count != 2:
                    print(" !! non-2-layer label:", lf)
                    two_ok = False
                    break
        print(" labels have 2 layers:", two_ok)


def pick_complete_tiles_per_region(
    root: str,
    year: str,
    regions,
    months,
    k_per_region: int = 32
):
    """Select k complete tiles per region for demo purposes."""
    data_root, label_root = _assert_root_year(root, year)
    regions_eff = regions or sorted([p.name for p in data_root.iterdir() if p.is_dir()])
    keep = set()
    counts = {}
    for r in regions_eff:
        label_ids = tiles_in(label_root / r)
        month_sets = []
        for m in months:
            mdir = data_root / r / str(m)
            month_sets.append(tiles_in(mdir) if mdir.exists() else set())
        complete = set.intersection(label_ids, *month_sets) if month_sets else set()
        chosen = sorted(list(complete))[:k_per_region]
        counts[r] = len(chosen)
        for t in chosen:
            keep.add((r, t))
    return keep, counts


def demo_loader(
    root: str,
    year: str,
    regions,
    months,
    batch_size: int = 8,
    k_per_region: int = 32
):
    """Demonstration DataLoader for quick testing."""
    from torch.utils.data import DataLoader
    
    _assert_root_year(root, year)
    keep, counts = pick_complete_tiles_per_region(root, year, regions, months, k_per_region)
    print("Picked per region:", counts, "| total:", sum(counts.values()))
    ds_full = WheatTilesDataset(
        root_preprocessed=root,
        year=year,
        regions=regions,
        month_order=months,
        temporal_layout=True,
        normalize=True,
        band_stats=None,
        require_complete=True,
        target_bands=None,
        target_size=(64, 64),
        size_policy="pad",
        probe_limit=12
    )
    print("Full tiles (after indexing):", len(ds_full))
    keep_set = set(keep)
    keep_idx = [
        i for i, rec in enumerate(ds_full.index)
        if (rec["region"], rec["tile_id"]) in keep_set
    ]
    ds = Subset(ds_full, keep_idx)
    if len(ds) == 0:
        print("No tiles selected; consider reducing require_complete or months.")
        return
    t0 = time.time()
    s = cast(Dict[str, Any], ds[0])
    t1 = time.time()
    print(
        "One sample:", round(t1 - t0, 3), "sec | x:", s["x"].shape,
        "| valid:", s["valid_mask"].shape, "| wheat:", s["wheat_mask"].shape
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    b = next(iter(loader))
    print("Batch x:", b["x"].shape)
    print("Batch valid:", b["valid_mask"].shape)
    print("Batch wheat:", b["wheat_mask"].shape)

