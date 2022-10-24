from utils import *
from autograd import grad, elementwise_grad
import autograd.numpy as np

dims = (4, 3, 3)
np.random.seed(0)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def nothing(x):
    return x


def CostOLS(X, target):
    def func(beta):
        return (1.0 / target.shape[0]) * np.sum((target - X @ beta) ** 2)

    return func


# beta = np.array([0.5, 0.5], dtype="float64")
# target = np.array([[1], [2]], dtype="float64")
# X = np.array([[1, 1], [2, 2]], dtype="float64")
#
# func = CostOLS(X, target)
#
# deri = grad(func)
# print(deri)
#
# print(deri(beta))
# print(X @ beta)

neural = FFNN(dims, epochs=1000)

X = np.array([[1, 1, 1, 1], [0, 1, 1, 1], [3, 5, 6, 7]])

target = np.array([[1, 1], [1, 2], [3, 3]])

# print(neural.predict(X))

# neural.fit(X, target)

x = np.array([2, 3, 4, 5])
# t = np.array([1, 5, 2, 3])
t = np.array([1, 5, 3])

print(f"{neural.predict(x)=}")

neural.fit(x, t)


print(f"{neural.predict(x)=}")

# x = np.array([2, 3, 4, 5])
# print(neural.hidden_outs)
