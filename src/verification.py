from FFNN import FFNN
from utils import *
from Schedulers import *


# data
x = np.arange(0, 1, 0.01)
y = 3 + x + x**2

x = x.reshape(x.shape[0], 1)
y = y.reshape(y.shape[0], 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# network
dims = (1, 10, 1)
sched = Constant
eta = 0.01
rho = 0.9
rho2 = 0.999
momentum = 0.5
sched_params = [eta, rho, rho2]
sched_params = [eta]

neural = FFNN(dims, seed=1337)

scores = neural.fit(
    x_train,
    y_train,
    sched,
    *sched_params,
    epochs=10000,
    X_test=x_test,
    t_test=y_test,
    batches=1
)

y_pred = neural.predict(x)

plt.plot(scores["train_errors"], label="train")
plt.plot(scores["test_errors"], label="test")
plt.show()

plt.title("Fitting of a simple second degree polynomial", fontsize="xx-large")
plt.plot(y, label="y", lw=4)
plt.plot(y_pred, "r--", label="predicted y", lw=4)
plt.xlabel("x", fontsize="xx-large")
plt.ylabel("y", fontsize="xx-large")
plt.legend(loc="upper left", fontsize="xx-large")
plt.show()
