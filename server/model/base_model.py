import numpy as np

class AbstractModel :
    
    def save(self,path):
        self.num_classes = -1
        raise NotImplementedError("please write saving code")
    
    def load(self,path):
        raise NotImplementedError("please write loading code")
    
    def predict_pixel(self,array:np.ndarray ):
        """
        takes a nx9x13 array and produce a n output
        """
        raise NotImplementedError("please implement inference code")
    
    def predict_patch(self,array:np.ndarray ):
        """
        takes a 9x13xnxn array and produce a nxn output
        """
        raise NotImplementedError("please implement inference code")
    
    def fit_pixel(self, dataset):
        """
        train pixel based methods in the dataset formate mx9x13
        """
        raise NotImplementedError("please implement inference code")
    
    def fit_patch(self, dataset):
        """
        train patch based methods in the dataset formate mx9x13
        """
        raise NotImplementedError("please implement inference code")
    
    def eval_patch_dataset(self, dataset):
        raise NotImplementedError("please implement inference code")
    
    def eval_pixel_dataset(self, dataset):
        raise NotImplementedError("please implement inference code")
    
