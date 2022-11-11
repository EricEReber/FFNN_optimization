import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import *
from Schedulers import *
from FFNN import FFNN
from sklearn.neural_network import MLPRegressor

np.random.seed(42069)
(
    betas_to_plot,
    N,
    X,
    X_train,
    X_test,
    z,
    z_train,
    z_test,
    centering,
    x,
    y,
    z,
) = read_from_cmdline()

z_train = z_train.reshape(z_train.shape[0], 1)
z_test = z_test.reshape(z_test.shape[0], 1)

# ------------------------- Params -------------------------
eta = 0.00005
momentum = 0.5
rho = 0.9
rho2 = 0.99
sched = Adam
# sched = Momentum
params = [eta, rho, rho2]
opt_params = [rho, rho2]
# params = [eta, momentum]

dims = (2, 224, 112, 1)
train_epochs = 10

eta = np.logspace(-5, -1, 5)
lams = np.logspace(-5, -1, 5)
batch_sizes = np.linspace(1, X.shape[0] // 2, 5, dtype=int)
hidden_func_list = [sigmoid, RELU, LRELU]

# ------------------------- FFNN -------------------------

for i in range(len(hidden_func_list)):
    neural = FFNN(dimensions=dims, hidden_func=hidden_func_list[i], seed=42069)

    optimal_params, optimal_lambda, _ = neural.optimize_scheduler(
        X_train[:, 1:3],
        z_train,
        X_test[:, 1:3],
        z_test,
        sched,
        eta,
        lams,
        opt_params,
        batches=10,
        epochs=20,
    )

    params = [optimal_params[0], rho, rho2]

    optimal_batch = neural.optimize_batch(
        X_train[:, 1:3],
        z_train,
        X_test[:, 1:3],
        z_test,
        sched,
        optimal_lambda,
        *params,
        batches_list=batch_sizes,
        epochs=20,
    )

    scores = neural.fit(
        X_train[:, 1:3],
        z_train,
        sched,
        *params,
        batches=optimal_batch[0],
        epochs=train_epochs,
        lam=optimal_lambda,
        X_test=X_test[:, 1:3],
        t_test=z_test,
    )

    test_error = scores["train_error"]
    plt.plot(test_error, label=f"{hidden_func_list[i].__name__}")
    plt.legend(loc=(1.04, 0))

plt.xlabel("Epoch", fontsize=18)
plt.ylabel("MSE", fontsize=18)
plt.title(
    "MSE over epochs for different activation functions in hidden layers", fontsize=22
)
plt.show()
