# Extracted from Colab notebook
# Colab magics and Drive mounts removed.
# Update DATA_ROOT below to point to your local data directory.
DATA_ROOT = r'./data'  # change if needed

#
"""
Minimal, runnable extraction of the Colab notebook logic.

Key variables you can pass via CLI or edit below:
  - ROOT: path pointing to your preprocessed root containing `data/` and `label/`.
  - YEAR: subfolder under those roots, e.g. "2020".
  - REGIONS: list of region ids (strings) or None for all.
  - MONTHS: months sequence to stack.
"""

# Default configuration (can be overridden by CLI)
ROOT = "."
YEAR = "2020"
REGIONS = None
MONTHS = (11,12,1,2,3,4,5,6,7)

# ==== Cell ====
# ==== Step 2 (REPLACE THIS WHOLE CELL) ====
from pathlib import Path
import glob
import time
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from typing import Dict, Any, cast
import rasterio
from collections import Counter

class WheatTilesDataset(Dataset):
    """
    data/<YEAR>/<REGION>/<MONTH>/<TILE_ID>.tif   # ~11–13 bands, 64×64 (usually)
    label/<YEAR>/<REGION>/<TILE_ID>.tif          # 2 layers: [0]=valid, [1]=wheat

    LAZY version:
      - Index is built from filenames only (fast).
      - Only a small probe (limit) is used to infer bands & size.
      - Size/band fixes happen at read-time (pad/trim), not during __init__.
    """
    def __init__(self,
        root_preprocessed: str,
        year: str = "2020",
        regions=None,                           # e.g., ["0","1","2","3","4"] or None for all
        month_order=(11,12,1,2,3,4,5,6,7),
        temporal_layout=False,                  # True -> [T,B,64,64]; False -> [C,64,64]
        normalize=True,
        band_stats=None,                        # None or {band:(mean,std)} or {(t,b):(mean,std)}
        require_complete=True,                  # only keep tiles with ALL months present
        # band & size handling
        target_bands: int | None = None,        # None => probe few files to detect modal count
        target_size: tuple[int,int] | None = (64, 64),  # None => probe few files to infer
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
        self.band_stats = band_stats
        self.require_complete = require_complete
        self.size_policy = size_policy

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
                print(f"[WARN] missing label dir: {label_dir}"); continue
            for lab_fp in sorted(glob.glob(str(label_dir / "*.tif"))):
                tile_id = Path(lab_fp).stem
                month_paths = {}
                complete = True
                for m in self.months:
                    m_fp = self.DATA / region / str(m) / f"{tile_id}.tif"
                    if m_fp.exists(): month_paths[m] = str(m_fp)
                    else: complete = False
                if self.require_complete and not complete:
                    continue
                if not self.require_complete and len(month_paths) == 0:
                    continue
                idx.append({"region": region, "tile_id": tile_id,
                            "label_path": str(lab_fp), "month_paths": month_paths})
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
                if seen >= limit: break
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
                if seen >= limit: break
            if not sizes:
                raise RuntimeError("Could not infer tile size.")
            self.H, self.W = sizes.most_common(1)[0][0]
        else:
            self.H, self.W = target_size

    def __len__(self): return len(self.index)

    def _fix_band_count(self, arr):
        B,H,W = arr.shape
        tb = self.num_bands
        if B == tb: return arr
        if B > tb:  return arr[:tb]
        pad = np.zeros((tb - B, H, W), dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)

    def _fix_size(self, arr):
        # Always pad/crop CENTER to target (size_policy="pad")
        H,W = arr.shape[1:]
        th, tw = self.H, self.W
        if (H,W) == (th,tw): return arr
        out = np.zeros((arr.shape[0], th, tw), dtype=np.float32)
        h = min(H, th); w = min(W, tw)
        sy = (H - h)//2 if H>h else 0
        sx = (W - w)//2 if W>w else 0
        dy = (th - h)//2 if th>h else 0
        dx = (tw - w)//2 if tw>w else 0
        out[:, dy:dy+h, dx:dx+w] = arr[:, sy:sy+h, sx:sx+w]
        return out

    def _normalize(self, arrTBHW):
        T,B,H,W = arrTBHW.shape
        out = arrTBHW.copy()
        if self.band_stats is None:
            # per-tile min-max per band across time
            for b in range(B):
                band = out[:, b]
                vmin = np.nanmin(band); vmax = np.nanmax(band)
                out[:, b] = 0.0 if vmax <= vmin else (band - vmin)/(vmax - vmin)
            return out
        keyed_tb = any(isinstance(k, tuple) and len(k) == 2 for k in self.band_stats.keys())
        if keyed_tb:
            for t in range(T):
                for b in range(B):
                    mean, std = self.band_stats.get((t,b), (0.0,1.0))
                    if std == 0: std = 1.0
                    out[t,b] = (out[t,b] - mean)/std
        else:
            for b in range(B):
                mean, std = self.band_stats.get(b, (0.0,1.0))
                if std == 0: std = 1.0
                out[:,b] = (out[:,b] - mean)/std
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
        if self.normalize: arrTBHW = self._normalize(arrTBHW)
        if self.temporal_layout: return arrTBHW
        T,B,H,W = arrTBHW.shape
        return arrTBHW.reshape(T*B, H, W)                   # [C,H,W]

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

# ---- Utility and CLI helpers ----
def tiles_in(folder: Path):
    return set(Path(fp).stem for fp in glob.glob(str(folder / "*.tif")))

def _assert_root_year(root: str, year: str):
    data_root = Path(root) / "data" / str(year)
    label_root = Path(root) / "label" / str(year)
    if not data_root.exists() or not label_root.exists():
        raise FileNotFoundError(
            f"Expected directories not found. Check: '{data_root}' and '{label_root}'. "
            f"Pass --root to the folder that contains 'data' and 'label'."
        )
    return data_root, label_root

def summarize_dataset(root: str, year: str, regions, months):
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
                    print(" !! non-2-layer label:", lf); two_ok = False; break
        print(" labels have 2 layers:", two_ok)

def pick_complete_tiles_per_region(root: str, year: str, regions, months, k_per_region: int = 32):
    data_root, label_root = _assert_root_year(root, year)
    regions_eff = regions or sorted([p.name for p in data_root.iterdir() if p.is_dir()])
    keep = set(); counts = {}
    for r in regions_eff:
        label_ids = tiles_in(label_root / r)
        month_sets = []
        for m in months:
            mdir = data_root / r / str(m)
            month_sets.append(tiles_in(mdir) if mdir.exists() else set())
        complete = set.intersection(label_ids, *month_sets) if month_sets else set()
        chosen = sorted(list(complete))[:k_per_region]
        counts[r] = len(chosen)
        for t in chosen: keep.add((r,t))
    return keep, counts

# (Removed the old top-level subset code that ran at import.)

def demo_loader(root: str, year: str, regions, months, batch_size: int = 8, k_per_region: int = 32):
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
        target_size=(64,64),
        size_policy="pad",
        probe_limit=12
    )
    print("Full tiles (after indexing):", len(ds_full))
    keep_set = set(keep)
    keep_idx = [i for i, rec in enumerate(ds_full.index)
                if (rec["region"], rec["tile_id"]) in keep_set]
    ds = Subset(ds_full, keep_idx)
    if len(ds) == 0:
        print("No tiles selected; consider reducing require_complete or months.")
        return
    t0 = time.time()
    s = cast(Dict[str, Any], ds[0])
    t1 = time.time()
    print("One sample:", round(t1-t0,3),"sec | x:", s["x"].shape,
          "| valid:", s["valid_mask"].shape, "| wheat:", s["wheat_mask"].shape)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    b = next(iter(loader))
    print("Batch x:", b["x"].shape)
    print("Batch valid:", b["valid_mask"].shape)
    print("Batch wheat:", b["wheat_mask"].shape)

# ==== Cell ====
# Optional: band stats computation existed in the notebook but is not
# defined here. You can plug yours in and pass `band_stats` to the dataset.

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Wheat tiles dataset loader demo")
    p.add_argument("--root", default=ROOT, help="Preprocessed root containing data/ and label/")
    p.add_argument("--year", default=YEAR, help="Year subfolder under data/ and label/")
    p.add_argument("--regions", nargs="*", default=None, help="Region ids (strings). If omitted, use all.")
    p.add_argument("--months", nargs="*", type=int, default=list(MONTHS), help="Months to include, e.g. 11 12 1 2 ...")
    p.add_argument("--batch", type=int, default=8, help="Batch size for demo loader")
    p.add_argument("--subset_k", type=int, default=32, help="Tiles per region for demo")
    p.add_argument("--summary", action="store_true", help="Print dataset summary and exit")
    args = p.parse_args()

    if args.summary:
        summarize_dataset(args.root, args.year, args.regions, tuple(args.months))
    else:
        demo_loader(args.root, args.year, args.regions, tuple(args.months), batch_size=args.batch, k_per_region=args.subset_k)
