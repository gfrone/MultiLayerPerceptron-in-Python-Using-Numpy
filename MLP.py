import math
import numpy as np
from callbacks import *

"""
    Funções de ativação recebem x
    A derivada dessas funções no backpropagation recebem a(ativação da camada anterior)
"""

## ------------------------------------- FUNÇÕES DE ATIVAÇÃO ------------------------------------- ##

def relu(x):
    return np.maximum(0, x)

def relu_backward(a):
    return (a > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_backward(a):
    return a * (1 - a)


## --------- Losses: Funções de perda ------------------------------------------ ##

def bce_loss(y_true, y_pred):
    # ETAPA 1: Pegar todos os valores de y_pred e evitar que resulte em log(0), ou quase zero
    # Para isso, usamos o np.clip. O valor mínimo é 1e-15 e o valor maximo é 1 - 1e-15. 
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss
    
def bce_loss_grad(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (-y_true / y_pred + (1 - y_true) / (1 - y_pred)) / y_true.size

def focal_loss():
    pass
def focal_loss_grad():
    pass


## -------------------------------------- METRICAS DE AVALIAÇÃO --------------------------------------------

def accuracy(y_pred, y_true):
    y_pred = np.asarray(y_pred).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    return float(np.mean(y_pred == y_true))

LOSSES = {
    'bce':   (bce_loss, bce_loss_grad),
    'focal': (focal_loss, focal_loss_grad),
}

METRICS = {
    'accuracy': accuracy,
    # 'precision': precision,
    # 'recall': recall,
    # 'f1': f1_score,
}

class MLP:
    def __init__(self, layers_size : list, hidden_activation, seed : int, loss : str):
        """
        layer_sizes: lista, ex [2, H, 1] -> entrada, oculta(s), saída
        hidden_activation: string, ex 'relu' -> resolve pra função via dict interno
        seed: int, seed fixa pro rng da instância
        loss: string, diz o nome da loss - retorna a loss e o gradiente da loss para a derivada - "bce" ou "focal"
        """
        self.layers_size = layers_size
        self.hidden_activation = hidden_activation
        self.rng = np.random.default_rng(seed)
        
        self.weights = []
        self.bias = []
        self._init_params()

        self.cache = {}
        self.loss_history_train = []
        self.loss_history_val = []
        self.loss_fn, self.loss_grad = LOSSES[loss]

    def _he_init_layer(self, n_in, n_out):
        """Retorna (W, b) para uma única camada com He init."""
        sigma = np.sqrt(2.0 / n_in)
        W = self.rng.normal(0.0, sigma, size=(n_in, n_out))
        b = np.zeros((1, n_out))
        return W, b

    def _init_params(self):
        """Inicializa pesos e bias para cada par de camadas consecutivas."""
        for n_in, n_out in zip(self.layers_size[:-1], self.layers_size[1:]):
            matrix, bias = self._he_init_layer(n_in, n_out)
            self.bias.append(bias)
            self.weights.append(matrix)

    def forward(self, X):
        """
        X -> vetor X da entrada: shape (N, n_features)
        Propaga camada por camada e guarda Z e A em self.cache.
        """
        A = X
        cache = {'A0': X}
        for i, (W, b) in enumerate(zip(self.weights, self.bias)):
            Z = A @ W + b
            A = sigmoid(Z) if i == len(self.weights) - 1 else relu(Z)
            cache[f'Z{i + 1}'] = Z
            cache[f'A{i + 1}'] = A

        self.cache = cache
        return A

    def compute_loss(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)

    def backward(self, X, y_true):
        L = len(self.weights) # Número de camadas

        dWs = [None] * L  # Vetor dos gradiente dos pesos
        dWb = [None] * L  # Vetor do gradiente dos biases

        # Para a última camada, calculamos seu gradiente
        # Com o ultimo gradiente podemos iniciar o loop
        A_out = self.cache[f'A{L}']
        delta = self.loss_grad(y_true, A_out) * sigmoid_backward(A_out)

        for i in reversed(range(L)):
            previous_activation = self.cache[f'A{i}']

            # Calcula o gradiente da camada atual, com o delta da camada anterior
            # dWs{l+1} = Activation{l+1}.Transposta * delta{l}
            dWs[i] = previous_activation.T @ delta 
            dWb[i] = np.sum(delta, axis=0, keepdims=True)

            if i > 0:  # Para todas camadas, exceto a de input
                # Atualiza o valor de delta para as proximas iterações - propagação do gradiente
                delta = (delta @ self.weights[i].T) * (relu_backward(previous_activation))

        return dWs, dWb

    def update_params(self, dWs, dWb, lr):
        """
        Função que atualiza os pesos da matriz e bias
        lr = Learning Rate - tamanho do passo que está sendo dado
        """
        n = len(self.weights)
        for i in reversed(range(n)):
            self.weights[i] = self.weights[i] - (lr * dWs[i])
            self.bias[i] = self.bias[i] - (lr * dWb[i])

    def predict(self, X_train, threshold = 0.5):
        # Retorna a predição nas classes 0 ou 1, com base no threshold
        y_pred = self.forward(X_train)
        return (y_pred >= threshold).astype(int)

    def train(self, X_train, y_train, X_val, y_val, epochs, lr, metrics : list, verbose_every=None,
              Callbacks = []):
        """
        Args:
        - X_train, y_train: Vetor de treinamento X e y
        - X_val, y_val: Vetores utilizados apenas para validação
        - epochs: Número máximo de épocas que será realizado o treinamento
        - lr: Learning Rate
        - metrics: lista de métricas que serão usadas para treinar
        - verbose_every: Se nao especificado nao reporta, mas se sim a cada x iterações
        - Callbacks: Pode receber uma lista de callbacks, como earlystopping

        Realiza o treinamento da rede seguindo os passos:
        1 Passo: Foward do X_train
        2: Passo: Calculo da loss e backpropagation
        3: Passo: Atualização do peso do treinamento
        Vai seguindo estes passos seguidamente segundo o SGD
        A métrica da validação será apenas calculada depois destes passos e não será treinada

        """
        
        for epoch in range(epochs):

            y_pred_train = self.forward(X_train)
            train_loss = self.compute_loss(y_train, y_pred_train)
            self.loss_history_train.append(train_loss)

            dWs, dWb = self.backward(X_train, y_train)
            self.update_params(dWs, dWb, lr)    

            # Não treina validação
            y_pred_val = self.forward(X_val)
            val_loss = self.compute_loss(y_val, y_pred_val)
            self.loss_history_val.append(val_loss)

            if Callbacks:
                for callback in Callbacks:
                    if isinstance(callback, EarlyStopping):
                        callback.on_epoch_end(self, epoch, val_loss)

                        if callback.stop_training:
                            callback.restore(self)
                            return
                    if isinstance(callback, ReduceLRONPlateau):
                        callback.verify(val_loss)
                        if callback.reduce:
                            lr = callback.apply_reduction(lr)


            if verbose_every is not None and (epoch % verbose_every == 0 or epoch == 0 or epoch == epochs - 1):
                y_pred_train_class = self.predict(X_train)
                y_pred_val_class = self.predict(X_val)

                metric_strs = []
                for metric in metrics:
                    metric_fn = METRICS[metric]
                    val_train = metric_fn(y_train, y_pred_train_class)
                    val_val = metric_fn(y_val, y_pred_val_class)
                    metric_strs.append(f"train_{metric}: {val_train:.4f} | val_{metric}: {val_val:.4f}|")

                print(f"época {epoch:4d} | train_loss: {train_loss:.4f}| val_loss: {val_loss:.4f} | " + " | ".join(metric_strs))
                