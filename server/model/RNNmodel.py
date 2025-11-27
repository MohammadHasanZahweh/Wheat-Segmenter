import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .base_model import AbstractModel
from torch import optim
from pathlib import Path
import torch.nn.functional as F

class SimpleRNNClassifier(nn.Module):
    def __init__(self, input_size=13, hidden_size=32, num_layers=1, num_classes=2):
        super(SimpleRNNClassifier, self).__init__()
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [B, 9, 13]
        out, h_n = self.rnn(x)        # out: [B, 9, hidden], h_n: [1, B, hidden]
        last_hidden = h_n[-1]         # [B, hidden]
        logits = self.fc(last_hidden) # [B, 2]
        return logits

class RNNPixelPatchModel(AbstractModel):
    """
    A simple fully-connected model implemented in PyTorch that works for both
    pixel and patch predictions.
    """

    def __init__(self, classes, hidden_dims = 8, layer_count = 3, lr=1e-3, device=None, mean = np.array([0]), std = np.array([1])):
        self.num_classes = int(len(classes))
        self.classes = classes
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dims = hidden_dims
        self.mean = mean
        self.std = std
        self.layer_count = 3

        input_dim = 13  # each feature is a 9x13 patch that we flatten

        # simple MLP
        self.net = SimpleRNNClassifier(input_size=input_dim, hidden_size=hidden_dims, num_layers=layer_count, num_classes=2).to(self.device)
        
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
                "layer_count":self.layer_count,
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
            layer_count = checkpoint["layer_count"],
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
        # x = x.view(x.shape[0], -1)

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
        x = x.view(9 , 13, H * W).permute(2,0,1)          # (H*W, 9, 13)

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
        # X_train = X_train.reshape((X_train.shape[0], -1))   # (M, 117)
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
    def fit_patch(self, dataset, epochs = 10, class_names = ["non_wheat", "wheat"]):
        """
        dataset: iterable of (X, y)
           X: (k, 9, 13, H, W)
           y: (k, H, W)
        """
        # self.num_classes = len(dataset.class_names)

        # collect samples
        X_list = []
        y_list = []

        for X, Y, v in dataset:
            X = X.detach().cpu().numpy()
            Y = Y.detach().cpu().numpy()
            v = v.detach().cpu().numpy()
            # print(X.shape)
            # print(X.shape)
            # print(X.shape)
            # print(v.dtype)
            # print(v.min(), v.max())
            # print(v)

            k, _, _, H, W = X.shape

            # flatten all spatial pixels
            X_flat = X.reshape(k, 9 , 13, H * W).transpose(0, 3, 1, 2).reshape(-1, 9 , 13)
            Y_flat = Y.reshape(-1)
            v_flat = v.reshape(-1) != 1

            X_list.append(X_flat[v_flat])
            y_list.append(Y_flat[v_flat])

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
    def val_patch_dataset(self, dataset, prefix=""):
        conf = np.zeros((self.num_classes, self.num_classes))

        for X, Y, v in dataset:
            preds = self.predict_patch(X)[v!=0]

            flat_y = Y[v!=0].reshape(-1)
            flat_p = preds.reshape(-1)

            for c in range(self.num_classes):
                mask = flat_y == c
                if mask.sum() > 0:
                    hist = np.bincount(flat_p[mask], minlength=self.num_classes)
                    conf[c] += hist

        return {
            prefix + "confusion_matrix": conf.tolist(),
            prefix + "accuracy": conf.diagonal().sum() / conf.sum(),
            prefix + "F1_score": 2 * conf.diagonal() /
                (conf.sum(axis=1) + conf.sum(axis=0)),
        }