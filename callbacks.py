import copy

"""
Aplicação limitada
Não é avaliado de acordo com cada métrica, apenas se reduziu ou não, que é o objetivo das métricas
avaliadas
"""

class EarlyStopping():
    def __init__(self, patience : int, restore_best=True, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best_value = float("inf")
        self.wait = 0
        self.stop_training = False
        self.best_weights = None
        self.best_bias = None
        self.best_epoch = None

    def on_epoch_end(self, model, epoch, monitored_value):
        improved = (monitored_value < self.best_value - self.min_delta).astype(bool)

        if improved:
            self.best_value = monitored_value
            self.wait = 0
            self.best_epoch = epoch

            self.best_weights = copy.deepcopy(model.weights)
            self.best_bias = copy.deepcopy(model.bias)

        else:
            self.wait += 1
            if self.wait > self.patience:
                self.stop_training=True
                print(f"EarlyStopping: Interrompendo o treinamento na época {epoch}")


    def restore(self, model):
        if self.restore_best:
            print(f"EarlyStopping: Restaurando os melhores pesos do treinamento: época {self.best_epoch}")
            model.weights = self.best_weights
            model.bias = self.best_bias


class ReduceLRONPlateau():
    def __init__(self, patience, factor=0.5):
        self.patience = patience
        self.wait = 0
        self.best_value = float("inf")
        self.factor = factor
        self.reduce = False

    def verify(self, monitored_value):
        self.reduce = False
        improved = monitored_value < self.best_value

        if improved:
            self.best_value = monitored_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.reduce = True
                self.wait = 0
                print(f"ReduceLRONPlateau: Reduzindo a learning rate em um fator de {self.factor}")

    def apply_reduction(self, lr):
        return lr * self.factor
