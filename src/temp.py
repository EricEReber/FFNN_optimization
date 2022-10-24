from utils import *
from autograd import grad, elementwise_grad
import autograd.numpy as np

dims = (4, 3, 5, 4, 3)
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

neural = FFNN(dims, epochs=4500)

X = np.array(
    [
        [0.83, 0.541, 0.455, 0.4124],
        [0.414, 0.4351, 0.4156, 0.8888],
        [0.111, 0.312, 0.77, 0.93],
    ]
)

target = np.array([[3, 2, 4], [3, 2, 5], [0, 2, 5]])

print(f"{neural.predict(X)=}")
neural.fit(X, target, scheduler=Scheduler(0.01))
print(f"{neural.predict(X)=}")
print(f"{target=}")
print(f"{neural.predict(X) - target=}")

neural.fit(X, target)

# neural = FFNN(dims, epochs=1)

x = np.array([[2, 3, 4, 5], [2, 3, 4, 5]])
t = np.array([[1, 5], [2, 3]])

# print(f"{neural.predict(x)=}")
# neural.fit(x, t)
# print(f"{neural.predict(x)=}")


# x = np.array([2, 3, 4, 5])
# print(neural.hidden_outs)
