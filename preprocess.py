import json
import numpy as np 

class Preprocess():
    def __init__(self, file_path):
        self.file_path = file_path

        self.x = None
        self.y = None

        self._load_data()

    def _load_data(self):
        with open(self.file_path, 'r') as f:
            data = json.load(f)

        self.X = np.array(data['X'])
        self.y = np.array(data['y'])

    def return_data(self):
        return self.X, self.y

    def minmax(self):        
        x_min = np.min(self.X, axis=0)
        x_max = np.max(self.X, axis=0)

        X_scaled = (self.X - x_min) / (x_max - x_min)
        self.X = X_scaled

    def standard_scaler(self):
        mean = np.mean(self.X, axis=0)
        std = np.std(self.X, axis=0)

        self.X = (self.X - mean) / std
        


    

        