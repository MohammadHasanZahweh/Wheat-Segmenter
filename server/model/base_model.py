import numpy as np

class AbstractModel :
    
    def save(self,path):
        self.num_classes = -1
        raise NotImplementedError("please write saving code")
    
    def load(self,path):
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
    
