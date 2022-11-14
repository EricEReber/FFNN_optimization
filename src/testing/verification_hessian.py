from FFNN import FFNN
from utils import *
from Schedulers import *


# data
x = np.arange(0, 1, 0.01)
y = 3 + x + x**2

x = x.reshape(x.shape[0], 1)
y = y.reshape(y.shape[0], 1)

X = np.c_[np.ones((x.shape[0], 1)), x, x**2]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scores, beta = hessian(
    x_train,
    y_train,
    epochs=1000,
    X_test=x_test,
    t_test=y_test,
)

plt.plot(scores["train_errors"], label="train")
plt.plot(scores["test_errors"], label="test")
plt.show()
