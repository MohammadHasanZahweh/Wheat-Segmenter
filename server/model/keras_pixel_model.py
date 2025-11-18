from .base_model import AbstractModel
import numpy as np
from server.config import SklearnType
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# -------------------------------------------------------
# Build sklearn sub-model
# -------------------------------------------------------

def build_sklearn_model(sub: SklearnType):
    if sub == SklearnType.KNN:
        return KNeighborsClassifier(n_neighbors=3)
    if sub == SklearnType.LR:
        return LogisticRegression(max_iter=500)

    raise NotImplementedError(f"Sub-model type {sub} is not implemented.")

class KerasModel(AbstractModel):
    def __init__(self):
        super().__init__()

    
    def save(self,path):
        self.num_classes = -1
        raise NotImplementedError("please write saving code")
    
    @classmethod
    def load(cls,path):
        raise NotImplementedError("please write loading code")
    
    def predict(self,array:np.ndarray ):
        """
        takes a 9x13xnxn array and produce a nxn output
        """
        raise NotImplementedError("please implement inference code")
    
    def train_pixel_based(self, X, y):
        """
        train pixel based methods in the dataset formate mx9x13
        """
        raise NotImplementedError("please implement inference code")
    
    def train_patch_based(self, dataset):
        """
        train patch based methods in the dataset formate mx9x13
        """
        raise NotImplementedError("please implement inference code")