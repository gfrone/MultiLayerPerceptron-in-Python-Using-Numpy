"""
Implementação das camadas ocultas de um MLP
Semelhante ao keras
Foi feito dessa forma porque penso que é mais facil para inicializar
uma camada individualmente, com uma própria ativação.

Assim, será possivel contruir neste formato:
    model = MLP()
    model.add(Dense(n_units, initialization, activation))

A última camada Também sera considerada densa, que nem da pra implementar no keras
"""

import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_backward(a):
    return (a > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_backward(a):
    return 1 - np.square(a)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_backward(a):
    return a * (1 - a)

ACTIVATIONS = {
    "relu": (relu, relu_backward),
    "tanh": (tanh, tanh_backward),
    "sigmoid" : (sigmoid, sigmoid_backward)
}

class Layer:
    def __init__(self, units : int, initialization : str = "he", activation : str = None):
        self.units = units
        self.initialization = initialization
        self.activation, self.act_grad = ACTIVATIONS[activation]
        self.W = None
        self.b = None

    def build(self, n_in, n_out, rng):
        if self.initialization == "he":
            sigma = np.sqrt(2.0 / n_in)
            W = rng.normal(0.0, sigma, size=(n_in, n_out))
            b = np.zeros((1, n_out))
            self.W, self.b = W, b

        elif self.initialization == "random":
            std = 0.01
            W = rng.normal(0.0, std, size=(n_in, n_out))
            b = np.zeros((1, n_out))
            self.W, self.b = W, b

        else:
            raise ValueError("Choose a valid initialization: 'he' or 'random'")

    def forward(self, X):
        self.X = X
        Z = X @ self.W + self.b
        A = self.activation(Z)
        self.Z, self.A = Z, A

        return self.A

    def backward(self, dA):
        dZ = dA * self.act_grad(self.A)
        self.dW = self.X.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)

        dX = dZ @ self.W.T
        return dX

    def update_params(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db
