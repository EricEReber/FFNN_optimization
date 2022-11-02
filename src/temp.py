from utils import *
from autograd import grad, elementwise_grad
import autograd.numpy as np
import matplotlib.pyplot as plt

dims = (2, 10, 10, 10, 1)
np.random.seed(0)


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

neural = FFNN(dims, epochs=1)

X = np.array(
    [
        [0.5, 0.5],
        [0.3, 0.5],
        [0.2, 0.5],
        [0.1, 0.5],
        [0.5, 0.3],
        [0.3, 0.3],
        [0.2, 0.3],
        [0.1, 0.3],
    ]
)

target = (X[:, 0] ** 2 + X[:, 1] ** 3).reshape(X.shape[0], 1)
# np.seterr(all="raise")
learnings = np.logspace(-5, 2, 10)
print(learnings[5])
print(f"{neural.predict(X)=}")
neural.fit(X, target, scheduler=Scheduler(learnings[5]))
print(f"{neural.predict(X)=}")
print(f"{target=}")
print("-" * 30)
print(f"{neural.predict(X) - target=}")

error = np.zeros(learnings.shape)
for i in range(len(learnings)):
    neural = FFNN(dims, epochs=2000)
    neural.fit(X, target, scheduler=Scheduler(learnings[i]))

    error[i] = MSE(target, neural.predict(X))

plt.plot(error)
plt.xlabel("iterations")
plt.ylabel("mse")
plt.show()

print(error)

# neural.fit(X, target)

# neural = FFNN(dims, epochs=1)

x = np.array([[2, 3, 4, 5], [2, 3, 4, 5]])
t = np.array([[1, 5], [2, 3]])

# print(f"{neural.predict(x)=}")
# neural.fit(x, t)
# print(f"{neural.predict(x)=}")


# x = np.array([2, 3, 4, 5])
# print(neural.hidden_outs)
