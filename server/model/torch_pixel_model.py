import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .base_model import AbstractModel
from torch import optim
from pathlib import Path

class TorchPixelPatchModel(AbstractModel):
    """
    A simple fully-connected model implemented in PyTorch that works for both
    pixel and patch predictions.
    """

    def __init__(self, classes, hidden_dims = [64,32], lr=1e-3, device=None, mean = np.array([0]), std = np.array([1])):
        self.num_classes = int(len(classes))
        self.classes = classes
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dims = hidden_dims
        self.mean = mean
        self.std = std

        input_dim = 9 * 13  # each feature is a 9x13 patch that we flatten

        assert len(hidden_dims) >= 1
        
        if len(hidden_dims) == 1:
            # simple MLP
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], self.num_classes),
            ).to(self.device)
        else:
            input_dim = 9 * 13
            self.net = nn.Sequential()
            for i in hidden_dims:
                self.net.append(nn.Linear(input_dim,i))
                self.net.append(nn.ReLU())
                input_dim = i
            self.net.append(nn.Linear(input_dim, self.num_classes))
            self.net.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    # ---------- Saving / Loading ----------

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "num_classes": self.num_classes,
                "hidden_dims": self.hidden_dims,
                "classes": self.classes,
                "lr": self.lr,
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
            },
            path,
        )

    @classmethod
    def load(cls, path, device=None):
        checkpoint = torch.load(path, map_location=device or "cpu")
        model = cls(
            classes=checkpoint["classes"],
            hidden_dims = checkpoint["hidden_dims"],
            lr=checkpoint.get("lr", 1e-3),
            device=device,
            mean = np.array(checkpoint.get("mean", [0])),
            std = np.array(checkpoint.get("std", [1])),
        )
        model.net.load_state_dict(checkpoint["state_dict"])
        return model

   
    # ---------------------------------------------------------------------
    #  ✔ PREDICT PIXEL  (same API as AbstractModel)
    # ---------------------------------------------------------------------
    def predict_pixel(self, array: np.ndarray, normalize = False):
        """array shape: (N, 9, 13) -> output shape: (N,)"""
        self.net.eval()
        if normalize:
            x = torch.tensor((array - self.mean)/self.std, dtype=torch.float32, device=self.device)
        else:
            x = torch.tensor(array, dtype=torch.float32, device=self.device)
        x = x.view(x.shape[0], -1)

        with torch.no_grad():
            logits = self.net(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        return preds

    # ---------------------------------------------------------------------
    #  ✔ PREDICT PATCH (9×13×H×W → H×W)
    # ---------------------------------------------------------------------
    def predict_patch(self, array: np.ndarray, normalize = False):
        """
        array: shape (9, 13, H, W)
        output: shape (H, W)
        """
        self.net.eval()

        H, W = array.shape[2], array.shape[3]

        if normalize:
            x = torch.tensor((array - self.mean.reshape(self.mean.shape[0], self.mean.shape[1],1,1))/self.std.reshape(self.mean.shape[0], self.mean.shape[1],1,1), 
                            dtype=torch.float32, device=self.device)
        else:
            x = torch.tensor(array, dtype=torch.float32, device=self.device)
        x = x.view(9 * 13, H * W).T          # (H*W, 117)

        with torch.no_grad():
            logits = self.net(x)
            preds = torch.argmax(logits, dim=1)

        return preds.view(H, W).cpu().numpy()

    # ---------------------------------------------------------------------
    #  ✔ TRAIN PIXEL (same logic as sklearn version you provided)
    # ---------------------------------------------------------------------
    def fit_pixel(self, dataset):
        """
        dataset: iterable of (X, y)
            X: shape (k, 9, 13)
            y: class index for entire chunk
        """

        # same logic as your sklearn version
        self.num_classes = len(dataset.class_names)

        X_train = np.concatenate([x for x, _ in dataset])   # (M, 9, 13)
        X_train = X_train.reshape((X_train.shape[0], -1))   # (M, 117)
        y_train = np.concatenate([y * np.ones((x.shape[0],), dtype=np.int64)
                                  for x, y in dataset])

        # build network if first time
        # if self.net is None:
        #     self.net = self._build_model()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        # convert to torch
        X = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_train, dtype=torch.long, device=self.device)

        # train
        self.net.train()
        for _ in range(10):  # epochs
            idx = torch.randperm(X.shape[0])
            Xb, yb = X[idx], y[idx]

            logits = self.net(Xb)
            loss = self.criterion(logits, yb)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # ---------------------------------------------------------------------
    #  ✔ TRAIN PATCH (mirrors pixel logic but flatten patches)
    # ---------------------------------------------------------------------
    def fit_patch(self, dataset, epochs = 10):
        """
        dataset: iterable of (X, y)
           X: (k, 9, 13, H, W)
           y: (k, H, W)
        """
        self.num_classes = len(dataset.class_names)

        # collect samples
        X_list = []
        y_list = []

        for X, Y in dataset:

            k, _, _, H, W = X.shape

            # flatten all spatial pixels
            X_flat = X.reshape(k, 9 * 13, H * W).transpose(0, 2, 1).reshape(-1, 9 * 13)
            Y_flat = Y.reshape(-1)

            X_list.append(X_flat)
            y_list.append(Y_flat)

        X_train = np.concatenate(X_list)
        y_train = np.concatenate(y_list)

        # build net if needed
        if self.net is None:
            self.net = self._build_net()
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
            self.criterion = nn.CrossEntropyLoss()

        X = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_train, dtype=torch.long, device=self.device)

        # train
        self.net.train()
        for _ in range(epochs):
            idx = torch.randperm(X.shape[0])
            Xb, yb = X[idx], y[idx]

            logits = self.net(Xb)
            loss = self.criterion(logits, yb)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # ---------------------------------------------------------------------
    #  ✔ VALIDATION PIXEL (identical output to your sklearn code)
    # ---------------------------------------------------------------------
    def val_pixel_dataset(self, dataset, prefix=""):
        conf = np.zeros((self.num_classes, self.num_classes))

        for X, y in dataset:
            Xf = X.reshape((X.shape[0], -1))
            yp = self.predict_pixel(X).astype(np.int16)

            for j in range(yp.min(), yp.max() + 1):
                conf[y, j] += (yp == j).sum()

        results = {
            prefix + "confusion_matrix": conf.tolist(),
            prefix + "accuracy": conf.diagonal().sum() / conf.sum(),
            prefix + "F1_score": 2 * conf.diagonal() /
                (conf.sum(axis=1) + conf.sum(axis=0)),
        }
        return results

    # ---------------------------------------------------------------------
    #  ✔ VALIDATION PATCH
    # ---------------------------------------------------------------------
    def val_patch_dataset(self, dataset):
        conf = np.zeros((self.num_classes, self.num_classes))

        for X, Y in dataset:
            preds = self.predict_patch(X)

            flat_y = Y.reshape(-1)
            flat_p = preds.reshape(-1)

            for c in range(self.num_classes):
                mask = flat_y == c
                if mask.sum() > 0:
                    hist = np.bincount(flat_p[mask], minlength=self.num_classes)
                    conf[c] += hist

        return {
            "confusion_matrix": conf.tolist(),
            "accuracy": conf.diagonal().sum() / conf.sum(),
            "F1_score": 2 * conf.diagonal() /
                (conf.sum(axis=1) + conf.sum(axis=0)),
        }