from FFNN import FFNN
from utils import *
from Schedulers import *


# data
x = np.arange(0, 1, 0.01)
y = 3 + x + x**2 + x**4 + x**5 + x

x = x.reshape(x.shape[0], 1)
y = y.reshape(y.shape[0], 1)

X = np.c_[np.ones((x.shape[0], 1)), x, x**2]

print(X.shape)
print(y.shape)
scores, beta = hessian_cv(
    5,
    X,
    y,
    epochs=5,
)

print(scores)

plt.plot(scores["train_errors"], label="train")
plt.plot(scores["test_errors"], label="test")
plt.show()
