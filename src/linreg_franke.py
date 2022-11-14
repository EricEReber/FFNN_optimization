import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate
from matplotlib.patches import Rectangle
from utils import *
from Schedulers import *
from FFNN import FFNN

np.random.seed(1337)

# read in data
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

# epochs to run for
epochs = 20
folds = 5

# no hidden layers, no activation function
dims = (X.shape[1], 1)
neural = FFNN(dims, seed=1337)

# optimal params written from task_a.py
scheduler = Adam
rho = 0.9
rho2 = 0.999
params = [rho, rho2]
eta = np.logspace(-5, -1, 5)
lam = np.logspace(-5, -1, 5)

batches_list = np.logspace(
    0, np.log(X_train.shape[0] + 1), 7, base=np.exp(1), dtype=int
)

# from project 1
best_MSE_analytically = np.zeros(epochs)
best_MSE_analytically[:] = 0.00328322949832417

# optimize Adam
optimal_params, optimal_lambda, _ = neural.optimize_scheduler(
    X_train,
    z_train,
    scheduler,
    eta,
    lam,
    params,
    batches=batches_list[3],
    epochs=epochs // 2,
    folds=folds,
)

# fit using optimal params
scores = neural.cross_val(
    folds,
    X_train,
    z_train,
    scheduler,
    *optimal_params,
    batches=batches_list[3],
    epochs=epochs,
    lam=optimal_lambda,
)
test_errors = scores["test_errors"]
train_errors = scores["train_errors"]

# test against sklearn
scikit_MLP = MLPRegressor(
    hidden_layer_sizes=[],
    activation="identity",
    solver="adam",
    alpha=optimal_lambda,
    batch_size=batches_list[3],
    learning_rate_init=optimal_params[0],
    max_iter=epochs,
    tol=0,
    n_iter_no_change=1000000,
)

# simple cross validation
scikit_train_errors = np.zeros(epochs)
for i in range(folds):
    scikit_MLP.fit(X_train, z_train)
    scikit_train_errors[:] += scikit_MLP.loss_curve_
scikit_train_errors /= folds

# plot
plt.plot(scikit_train_errors, label="scikit Adam train")
plt.plot(train_errors, label="Adam train")
plt.plot(test_errors, label="Adam test")
plt.plot(best_MSE_analytically, label="analytical MSE")
plt.legend(loc="upper right")
plt.xlabel("Epochs", fontsize=18)
plt.ylabel("MSE", fontsize=18)
plt.title("MSE over Epochs for optimized Adam", fontsize=22)
plt.show()
