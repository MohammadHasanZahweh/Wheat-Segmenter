from .base_model import AbstractModel
import numpy as np
from server.server.config import SklearnType
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

class SklearnModel(AbstractModel):
    def __init__(self, sub_model_type):
        super().__init__()
        self.num_classes = -1
        self.model = build_sklearn_model(sub_model_type)

    
    def save(self,path):
        self.num_classes = -1
        raise NotImplementedError("please write saving code")
    
    @classmethod
    def load(cls,path):
        raise NotImplementedError("please write loading code")
    
    def predict_patch(self, array:np.ndarray ):
        """
        takes a 9x13xnxn array and produce a nxn output
        """
        raise NotImplementedError("please implement inference code")
    
    def fit_pixel(self, dataset):
        """
        train pixel based methods in the dataset formate mx9x13
        """
        self.num_classes = len(dataset.class_names)
        X_train = np.concatenate([x for x, _ in dataset])
        X_train = X_train.reshape((X_train.shape[0],-1))
        y_train = np.concatenate([y * np.ones(((x.shape[0]))) for x, y in dataset])
        self.model.fit(X_train, y_train)
    
    def fit_patch(self, dataset):
        """
        train patch based methods in the dataset formate mx9x13
        """
        raise NotImplementedError("please implement inference code")
    
    def val_patch_dataset(self, dataset):
        raise NotImplementedError("please implement inference code")
    
    def val_pixel_dataset(self, dataset, prefix = ""):
        conf = np.zeros(shape=(self.num_classes,self.num_classes))
        for X,y in dataset:
            yp = self.model.predict(X.reshape((X.shape[0],-1))).astype(np.int16)
            for j in range(yp.min(), yp.max()+1):
                conf[y,j] += ((yp==j)).sum()

        results = {
            prefix + "confusion_matrix" : conf.tolist(),
            prefix + "accuracy" : conf.diagonal().sum()/conf.sum(),
            prefix + "F1_score" : 2*conf.diagonal() / (conf.sum(axis=1) + conf.sum(axis=0)),
            # prefix + "confusion_matrix" : conf.tolist(),
        }
        return results
