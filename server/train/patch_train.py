import joblib
from pathlib import Path
from typing import Dict, Any
import numpy as np
from torch.utils.data import Dataset

from server.server.config import ModelType, SklearnType, PATCH_SPLIT_DATA_PATH, PixelTrainRequest, MODELS_PATH, META_PATH
from server.dataset.PatchDataset import WheatTilesDataset, StratifiedRandomSubset, Subset
from server.model.torch_pixel_model import TorchPixelPatchModel
from server.model.RNNmodel import RNNPixelPatchModel
from torch.utils.data import DataLoader
import os

year = 2020
aois = [0,1,2,3,4]
months = [11,12,1,2,3,4,5,6,7,]

def get_mean(path):
    meta = [np.load(path/f) for f in os.listdir(path)]
    return np.array([m["mean"] for m in meta])

def get_std(path):
    meta = [np.load(path/f) for f in os.listdir(path)]
    return np.array([m["std"] for m in meta])


import numpy as np
def build_xy_from_tiles(
    dataset: WheatTilesDataset,
    tile_indices,
    seed: int,
):
    """Aggregate per-pixel features and labels over a set of tiles."""
    # rng = np.random.default_rng(seed)
    xs = []
    ys = []
    vs = []
    for i in tile_indices:
        item = dataset[i]
        x = item["x"].numpy()
        valid = item["valid_mask"].numpy()[0]
        wheat = item["wheat_mask"].numpy()[0]

        if valid.sum():
            xs.append(x)
            vs.append(valid)
            ys.append(wheat)
        # print(x.shape)

    X = np.stack(xs, axis=0)
    y = np.stack(ys, axis=0)
    v = np.stack(vs, axis=0)
    return X, y, v

def sample_tiles(ds: WheatTilesDataset, train_fraction = 0.01,test_fraction = 0.5 , seed = 42) -> tuple[list[int], list[int]]:
    """Sample train and test tiles using stratified sampling."""
    print(f"Sampling TRAIN ~{train_fraction*100:.2f}% of tiles (stratified)...")
    train_tiles = list(iter(StratifiedRandomSubset(ds, fraction=train_fraction, n_bins=5, seed=seed)))
    print(f"Train tiles: {len(train_tiles)}")
    
    if len(train_tiles) == 0:
        raise RuntimeError("Train sampler returned 0 tiles. Increase train_fraction or check data.")
    
    all_ids = set(range(len(ds)))
    remaining = sorted(all_ids.difference(set(train_tiles)))
    
    if len(remaining) == 0:
        raise RuntimeError("No remaining tiles to sample test set from. Lower train_fraction.")
    
    rem_subset = Subset(ds, remaining)
    print(f"Sampling TEST ~{test_fraction*100:.2f}% of remaining tiles (stratified)...")
    test_sampler = StratifiedRandomSubset(rem_subset, fraction=test_fraction, n_bins=5, seed=seed + 7)
    val_tiles = [remaining[i] for i in iter(test_sampler)]
    print(f"Test tiles: {len(val_tiles)} (sampled from {len(remaining)} remaining)")
    
    return train_tiles, val_tiles

class XYV_dataset(Dataset):
    def __init__(self,X,y,v):
        super().__init__()
        self.X = X
        self.y = y
        self.v = v
    
    def __len__(self):
        return len(self.v)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index], self.v[index]

# -------------------------------------------------------
# Train routine
# -------------------------------------------------------
def train_patch_model(req: PixelTrainRequest) -> Dict[str, Any]:
    print("[INFO] Loading datasets...")

    root = PATCH_SPLIT_DATA_PATH
    

    ds = WheatTilesDataset(
        root_preprocessed=PATCH_SPLIT_DATA_PATH,
        year=2020,
        regions=[str(a) for a in [0,1,2,3,4]],
        month_order=months,
        temporal_layout=True,
        normalize=True,
        band_stats='auto',
        meta_dir=META_PATH,
        require_complete=True,
        target_bands=None,
        target_size=(64, 64),
        size_policy="pad",
        probe_limit=12,
    )
    train_tiles, val_tiles = sample_tiles(ds)

    X_train, y_train, v_train = build_xy_from_tiles(ds, train_tiles, 42)
    train_ds = XYV_dataset(X_train, y_train, v_train)
    
    X_val, y_val, v_val = build_xy_from_tiles(ds, val_tiles, 42)
    val_ds = XYV_dataset(X_val, y_val, v_val)

    print(f"Total size: {len(train_ds)}")
    print(f"Val size:   {len(val_ds)}")
    # print(f"Test size:  {len(test_ds)}")

    status = {
        "status": "completed",
    }

    # -----------------------
    # Build model
    # -----------------------
    if req.model_type == ModelType.TORCH:
        save_path = MODELS_PATH / (req.run_save_name + ".torch.joblib")
        model = TorchPixelPatchModel(["nw","w"], mean=get_mean(META_PATH), std=get_std(META_PATH))
    
    elif req.model_type == ModelType.RNN:
        save_path = MODELS_PATH / (req.run_save_name + ".rnn.joblib")
        model = RNNPixelPatchModel(["nw","w"], mean=get_mean(META_PATH), std=get_std(META_PATH))
    
    else:
        raise NotImplementedError(f"Model type {req.model_type} not supported yet.")

    # -----------------------
    # Fit
    # -----------------------
    print("[INFO] Training...")
    model.fit_patch(DataLoader(train_ds, batch_size = 16))

    status.update(model.val_patch_dataset(val_ds,"val_"))
    print(f"[INFO] Validation accuracy = {status["val_accuracy"]:.4f}")

    # -----------------------
    # Save
    # -----------------------
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    # joblib.dump(model, save_path)
    status["save_path"] = str(save_path)



    # status.update(model.val_pixel_dataset(test_ds, "test_"))


    return status

if __name__ == "__main__":
    # import sys
    # sys.path.append("..")
    import os
    os.chdir("../..")
    print(os.curdir())

    a = train_patch_model(PixelTrainRequest(
        run_save_name="t1",
        model_type=ModelType.SKLEARN,
        sub_model_type=SklearnType.LR,
        val_batches=1,
        train_batches=1,
        class_root = "wheat",
        ))
    print(a)