from MLP import MLP
from Layers import Layer
from preprocess import Preprocess
import numpy as np
from MLP import classification_report, plot_confusion_matrix
from callbacks import EarlyStopping

data = Preprocess('data/dataset_2.json')
data.standard_scaler()
X, y = data.return_data()

X = np.asarray(X, dtype=float)

y = np.asarray(y, dtype=float)
if y.ndim == 1:
    y = y.reshape(-1, 1)

# Mantém a divisão determinística, mas evita que os dados dependam da ordem
# original do arquivo.
rng = np.random.default_rng(42)
indices = rng.permutation(len(X))
X = X[indices]
y = y[indices]

prop = int(len(X) * 0.8)
X_train, X_val = X[:prop], X[prop:]
y_train, y_val = y[:prop], y[prop:]

model = MLP(X_train.shape[1], 42, "bce")
for _ in range(3):
    model.add(Layer(126, activation="tanh"))
model.add(Layer(1, initialization="he", activation="sigmoid"))

callbacks = []
callbacks.append(EarlyStopping(15, True, 1e-4))

model.train(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=300,
    lr=5e-3,
    metrics=["accuracy"],
    verbose_every=10,
)

print("loss final de treino:", model.loss_history_train[-1])
print("loss final de validação:", model.loss_history_val[-1])


y_pred = model.predict(X_train)
classification_report(y_train, y_pred)