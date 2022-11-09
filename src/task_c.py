import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.patches import Rectangle
from utils import *
from Schedulers import *
from FFNN import FFNN


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

opti_epochs = 20
train_epochs = 1000
etas = np.logspace(-5, -1, 5)
lams = np.zeros(etas.shape)
print(lams)  # np.logspace(-5, -1, 5)
rho = 0.9
rho2 = 0.999

dims = (X.shape[1], 1)
hidden_func_list = [sigmoid, RELU, LRELU]
batch_sizes = np.linspace(1, X.shape[0] // 2, 5, dtype=int)
scheduler = [Adam]
scheduler_params = [[rho, rho2]]

optimal_eta = np.zeros(len(scheduler))
optimal_lam = np.zeros(len(scheduler))
optimal_batch = np.zeros(len(scheduler))
optimal_error = np.zeros(len(scheduler))

for i in range(len(hidden_func_list)):
    neural = FFNN(dimensions=dims, hidden_func=hidden_func_list[i], seed=42069)

    optimal_params, optimal_lambda, optimal_batch, _, _ = neural.optimize_scheduler(
        X_train,
        z_train,
        X_test,
        z_test,
        scheduler[0],
        etas,
        lams,
        batch_sizes,
        scheduler_params[0],
        batches=X.shape[0],
        epochs=opti_epochs,
    )

    scores = neural.fit(
        X=X_train,
        t=z_train,
        scheduler_class=scheduler[0],
        *optimal_params,
        batches=optimal_batch,
        epochs=train_epochs,
        lam=optimal_lambda,
        X_test=X_test,
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
