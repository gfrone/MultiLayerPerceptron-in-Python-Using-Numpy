from MLP import *
from callbacks import *
from preprocess import *

def testa_convergencia_neuronios(X_train, y_train, y_test, min_camadas, max_camadas):
    best_numero_neuronios = 0
    for i in range(min_camadas, max_camadas):
        model = MLP([X_train.shape[1], i, y_train.shape[1]], "relu", 42, "bce")
