# Our own library of functions
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
epochs = 200
folds = 5

# no hidden layers, no activation function
dims = (X.shape[1], 1)
neural = FFNN(dims, seed=1337)

# parameters to test for
eta = np.logspace(-5, -1, 5)
lam = np.logspace(-5, -1, 5)
momentums = np.linspace(0, 0.1, 5)
lam[0] = 0
rho = 0.9
rho2 = 0.999

# batches to test for
batches_list = np.logspace(
    0, np.log(X_train.shape[0] + 1), 7, base=np.exp(1), dtype=int
)

# schedulers to test for
schedulers = [Constant, Momentum, Adagrad, AdagradMomentum, RMS_prop, Adam]
#
# parameters for schedulers
constant_params = []
momentum_params = momentums
adagrad_params = []
adagrad_momentum_params = momentums
rms_params = [rho]
adam_params = [rho, rho2]

# list of scheduler parameters
params_list = [
    constant_params,
    momentum_params,
    adagrad_params,
    adagrad_momentum_params,
    rms_params,
    adam_params,
]

# results
optimal_params_list = []
optimal_eta = np.zeros(len(schedulers))
optimal_lambdas = np.zeros(len(schedulers))
optimal_batches = np.zeros(len(schedulers), dtype=int)

# gridsearch eta, lambda
for i in range(len(schedulers)):
    plt.subplot(321 + i)
    plt.suptitle("Test loss for eta, lambda grid", fontsize=22)
    optimal_params, optimal_lambda, loss_heatmap = neural.optimize_scheduler(
        X_train,
        z_train,
        X_test,
        z_test,
        schedulers[i],
        eta,
        lam,
        params_list[i],
        batches=X_train.shape[0],
        epochs=epochs // 2,
        folds=folds,
    )

    optimal_eta[i] = optimal_params[0]
    optimal_lambdas[i] = optimal_lambda
    optimal_params_list.append(optimal_params)

    # plot heatmap
    ax = sns.heatmap(loss_heatmap, xticklabels=lam, yticklabels=eta, annot=True)
    ax.add_patch(
        Rectangle(
            (np.where(lam == optimal_lambda)[0], np.where(eta == optimal_params[0])[0]),
            width=1,
            height=1,
            fill=False,
            edgecolor="crimson",
            lw=4,
            clip_on=False,
        )
    )
    plt.xlabel("lambda", fontsize=18)
    plt.ylabel("eta", fontsize=18)
    plt.title(f"{schedulers[i].__name__}", fontsize=22)
plt.show()

# search batch size
for i in range(len(schedulers)):
    plt.subplot(321 + i)
    plt.suptitle("MSE over epochs for different \n batch sizes", fontsize=22)
    optimal_batch, batches_list_search = neural.optimize_batch(
        X_train,
        z_train,
        X_test,
        z_test,
        schedulers[i],
        optimal_lambdas[i],
        *optimal_params_list[i],
        batches_list=batches_list,
        epochs=epochs,
    )

    for j in range(len(batches_list)):
        plt.plot(
            batches_list_search[j, :],
            label=f"batch size {X_train.shape[0]//batches_list[j]}",
        )
        plt.legend(loc=(1.04, 0))
    plt.xlabel("epochs", fontsize=18)
    plt.ylabel("MSE score", fontsize=18)
    plt.title(schedulers[i].__name__, fontsize=22)
plt.show()

# plot best run for each scheduler
for i in range(len(schedulers)):
    neural = FFNN(dims, seed=1337)
    scores = neural.cross_val(
        folds,
        X_train,
        z_train,
        schedulers[i],
        *optimal_params_list[i],
        batches=optimal_batch,
        epochs=epochs,
        lam=optimal_lambdas[i],
    )

    test_errors = scores["test_errors"]

    plt.plot(test_errors, label=f"{schedulers[i].__name__}")
    plt.legend(loc=(1.04, 0))

best_MSE_analytically = np.zeros(epochs)
best_MSE_analytically[:] = 0.003027

plt.plot(best_MSE_analytically)
plt.legend(loc=(1.04, 0))
plt.xlabel("Epochs", fontsize=18)
plt.ylabel("MSE", fontsize=18)
plt.title("MSE over Epochs for different schedulers", fontsize=22)
plt.show()

# # plot bias-variance trade-off
# for i in range(len(schedulers)):
#     plt.subplot(321 + i)
#     plt.suptitle("Bias-variance trade-off", fontsize=22)
#
#     plt.title(schedulers[i].__name__, fontsize=22)
#     plt.plot(test_errors[i], label="MSE score")
#     plt.plot(all_biases[i], label="bias")
#     plt.plot(all_variances[i], label="variance")
#     plt.legend(loc=(1.04, 0))
# plt.show()
