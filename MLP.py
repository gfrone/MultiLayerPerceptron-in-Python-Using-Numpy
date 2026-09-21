import math
import numpy as np

def relu(x):
    return max(0, x)

def relu_d(x):
    return 0 if x < 0 else 1

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_d(x):
    return - math.exp(-x) / ((1 + math.exp(-x)) * (1 + math.exp(-x)))

def bce_loss(y_true, y_pred):
    """
        Implementação da loss binary cross entropy. shape (N, 1). Retorna escalar (perda média)
    """
def bce_loss_grad():
    pass

def focal_loss():
    pass
def focal_loss_grad():
    pass

LOSSES = {
    'bce':   (bce_loss, bce_loss_grad),
    'focal': (focal_loss, focal_loss_grad),
}

class MLP:
    def __init__(self, layers_size : list, hidden_activation, seed : int, loss : str):
        """
        layer_sizes: lista, ex [2, H, 1] -> entrada, oculta(s), saída
        hidden_activation: string, ex 'relu' -> resolve pra função via dict interno
        seed: int, seed fixa pro rng da instância
        loss: string, diz o nome da loss - retorna a loss e o gradiente da loss para a derivada
        """
        self.layers_size = self.layers_size
        self.hidden_activation = hidden_activation
        self.rng = np.random.default_rng(seed)
        
        self.weights = []
        self.bias = []
        self._init_params()

        self.cache = {}
        self.loss_history_train = []
        self.loss_history_val = []
        self.loss, self.loss_grad = LOSSES[loss]


        def _he_init_layer(self, n_in, n_out):
            """Retorna (W, b) para uma única camada com He init."""
            sigma = np.sqrt(2.0 / n_in)
            W = self.rng.normal(0.0, sigma, size=(n_in, n_out))
            b = np.zeros((1, n_out))
            return W, b

        def _init_params(self):
            """
            Percorre o layers_size e inicia as matrizes dos pesos e bias com he_init
            """
            for n_in, n_out in zip(self.layers_size[:, -1]. self.layers_size[1:]):
                matrix, bias = _he_init_layer(n_in, n_out)
                self.bias.append(bias)
                self.weights.append(matrix)
      
        def forward(self, X):
            """
            X -> vetor X da entrada: shape (N, n_features)
            Propaga camada por camada, guarda z e a de cada camada em self.cache.
            Retorna y_pred: shape (N, 1)
            """
            A = X
            cache = {'A0' : X}
            for i, (W, b) in enumerate(zip(self.weights, self.bias)):
                Z = A @ W + b
                if(i == len(self.weights) - 1):
                    A = sigmoid(Z)
                else:
                    A = relu(Z)

                cache[f'Z{i+1}'] = Z
                cache[f'A{i+1}'] = A

            self.cache = cache
            return A    

        def compute_loss(y_true, y_pred):
            # Como foward retorna y_pred como X, vamos utiliza-lo como argumento
            # y_true shape = (N, 1)
            n = len(y_true[0])
            loss = 0.0
            for i in range (n):
                loss += bce_loss(y_true[i], y_pred[i])
            return loss / n
 
                
            