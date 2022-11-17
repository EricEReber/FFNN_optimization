from utils import *
from sklearn.datasets import load_diabetes

np.random.seed(42069)

from sklearn.neural_network import MLPRegressor

inputs = 10
diabetes = load_diabetes()

mlp = MLPRegressor(
    hidden_layer_sizes=(10, 10),
    activation="logistic",
    solver="adam",
    max_iter=10000,
    learning_rate="adaptive",
    tol=0,
    n_iter_no_change=10000,
)

X = diabetes.data
t = diabetes.target

X_train, X_test, t_train, t_test = train_test_split(X, t)

t_train = t_train.reshape(t_train.shape[0], 1)
t_test = t_test.reshape(t_test.shape[0], 1)

t_train /= 10
t_test /= 10

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)

mlp.fit(X_train, np.ravel(t_train))

t_pred = mlp.predict(X_test)

print(MSE(t_test, t_pred))
