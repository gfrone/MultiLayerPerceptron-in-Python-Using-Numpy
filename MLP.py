import math
import numpy as np
from callbacks import *
import matplotlib.pyplot as plt
from Layers import Layer

"""
    Funções de ativação recebem x
    A derivada dessas funções no backpropagation recebem a(ativação da camada anterior)
"""

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

def focal_loss(y_true, y_pred, gama, beta):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
def focal_loss_grad():
    pass


## -------------------------------------- METRICAS DE AVALIAÇÃO --------------------------------------------

def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(y_pred == y_true))

def precision(y_true, y_pred, label=1):
    # Proporção de verdadeiros positivos para todas as predições positivas
    # precision = TP / TP + FP
    true_positives = np.sum((y_pred == y_true) & (y_pred == label))
    false_positives = np.sum((y_pred != y_true) & (y_pred == label))
    return true_positives / (true_positives + false_positives)

def recall(y_true, y_pred, label=1):
    # Proporção de verdadeiros positivos em relação a todos os classes positivas existentes
    # recall = TP / TP + FN ou TP/ALL_POSITIVES_LABELS
    true_positives = np.sum((y_pred == y_true) & (y_pred == label))
    all_labels_positives = np.sum(y_true == label)
    return true_positives / all_labels_positives

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r)

def plot_confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tp = np.sum((y_pred == 1) & (y_true == 1))

    matrix = np.array([
        [tn, fp],
        [fn, tp]
    ])

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j, i,
                matrix[i, j],
                ha="center",
                va="center"
            )

    plt.colorbar()
    plt.tight_layout()
    plt.show()


def classification_report(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    # Métricas para classe 0
    precision_0 = precision(y_true, y_pred, label=0)
    recall_0 = recall(y_true, y_pred, label=0)
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0)

    # Métricas para classe 1
    precision_1 = precision(y_true, y_pred, label=1)
    recall_1 = recall(y_true, y_pred, label=1)
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)

    acc = accuracy(y_true, y_pred)

    print("              precision    recall    f1-score")
    print(f"0             {precision_0:.4f}      {recall_0:.4f}      {f1_0:.4f}")
    print(f"1             {precision_1:.4f}      {recall_1:.4f}      {f1_1:.4f}")
    print()
    print(f"accuracy                          {acc:.4f}")


LOSSES = {
    'bce':   (bce_loss, bce_loss_grad),
    'focal': (focal_loss, focal_loss_grad),
}

METRICS = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1_score,
}

class MLP:
    def __init__(self, input_dim, seed : int, loss : str):

        self.rng = np.random.default_rng(seed)
        self.loss_history_train = []
        self.loss_history_val = []
        self.loss_fn, self.loss_grad = LOSSES[loss]
        self.input_dim = input_dim
        self.layers = []

    def add(self, layer : Layer):
        n_in = self.input_dim if not self.layers else self.layers[-1].units
        layer.build(n_in, layer.units, self.rng)
        self.layers.append(layer)

    def forward(self, X):
        x = X
        # bota umas celula de verificaççao pq ta dando erro aqui eu acho
        if X.ndim != 2:
            raise ValueError(
                f"X deve ser uma matriz 2D, mas recebeu shape {X.shape}"
            )

        if X.shape[1] != self.input_dim:
            raise ValueError("A dimensão de X_train é diferente da especificada no input")

        for layer in self.layers:
            x = layer.forward(x)
        return x

    def compute_loss(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)

    def backward(self, dA):
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def predict(self, X_train, threshold = 0.5):
        # Retorna a predição nas classes 0 ou 1, com base no threshold
        y_pred = self.forward(X_train)
        return (y_pred >= threshold).astype(int)

    def train(self, X_train, y_train, X_val, y_val, epochs, lr, metrics : list, verbose_every=None,
              batch_size=None, Callbacks = None):
        """
            Caso nao seja especificado batch_size, então é gradient descent, passando todo o dataset
        """
        
        X_train, y_train = np.asarray(X_train), np.asarray(y_train)
        N = X_train.shape[0]
        if batch_size is None:
            batch_size = N

        for epoch in range(epochs):
            indices = self.rng.permutation(N)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for index in range(0, N, batch_size): # Passa o dataset em X batches, tipo em 32 em 32, por exemplo
                Xb = X_shuffled[index:index + batch_size]
                yb = y_shuffled[index:index + batch_size]

                y_pred_b = self.forward(Xb)
                dA = self.loss_grad(yb, y_pred_b)
                self.backward(dA)

                for layer in self.layers:
                    layer.update_params(lr)

            y_pred_train = self.forward(X_train)
            train_loss = self.compute_loss(y_train, y_pred_train)
            self.loss_history_train.append(train_loss)
            
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
                    val_train = metric_fn(y_pred_train_class, y_train)
                    val_val = metric_fn(y_pred_val_class, y_val)
                    metric_strs.append(f"train_{metric}: {val_train:.4f} | val_{metric}: {val_val:.4f}|")

                print(f"época {epoch:4d} | train_loss: {train_loss:.4f}| val_loss: {val_loss:.4f} | " + " | ".join(metric_strs))

    # ------------------ Plot das curvas de treinamento ----------------------------------

    def plot_train_val_loss(self, figsize = (12,6)):
        epochs_train = range(len(self.loss_history_train))
        epochs_val = range(len(self.loss_history_val))

        plt.figure(figsize=(figsize))
        plt.plot(epochs_train, self.loss_history_train, label='Loss de Treino', color='blue', marker='o')
        plt.plot(epochs_val, self.loss_history_val, label='Loss da Validação', color='orange', marker='o')
        plt.title("Log loss treino/validação")
        plt.xlabel("Epocas")
        plt.ylabel("Valor Loss")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.close()
