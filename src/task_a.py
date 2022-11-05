# Our own library of functions
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.patches import Rectangle
from utils import *

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

epochs = 10

# no hidden layers, no activation function
dims = (X.shape[1], 1)

eta = np.logspace(-5, -1, 5)
lam = np.logspace(-5, -1, 5)
lam[0] = 0
rho = 0.9
rho2 = 0.999

schedulers = [Constant, Momentum, Adagrad, RMS_prop, Adam]

constant_params = []
momentum_params = np.linspace(0, 1, 5)
adagrad_params = []
rms_params = [rho]
adam_params = [rho, rho2]

params_list = [constant_params, momentum_params, adagrad_params, rms_params, adam_params]
optimal_params_list = []

optimal_eta = np.zeros(len(schedulers))
optimal_lambdas = np.zeros(len(schedulers))
optimal_batches = np.zeros(len(schedulers), dtype=int)
minimal_errors = np.zeros(len(schedulers))

neural = FFNN(dims, checkpoint_file=f"test_task_a")
for i in range(len(schedulers)):
    plt.subplot(321 + i)
    plt.suptitle("Loss for eta, lambda grid", fontsize=22)
    (
        optimal_params,
        optimal_lambda,
        optimal_batch,
        minimal_error,
        plotting_data,
    ) = neural.optimize_scheduler(
        X_train, z_train, schedulers[i], eta, lam, params_list[i], epochs=epochs,
    )

    optimal_eta[i] = optimal_params[0]
    optimal_lambdas[i] = optimal_lambda
    optimal_batches[i] = optimal_batch
    minimal_errors[i] = minimal_error
    optimal_params_list.append(optimal_params)

    # plot heatmap
    ax = sns.heatmap(plotting_data[0], xticklabels=lam, yticklabels=eta, annot=True)
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

for i in range(len(schedulers)):
    print(
        f"{schedulers[i].__name__} \n {optimal_eta[i]=}\n {optimal_lambdas[i]=} \n {optimal_batches[i]=}\n{minimal_errors[i]=}"
    )
    # neural.read(f"comparison{i}")
    error_over_epochs, _ = neural.fit(
        X_train,
        z_train,
        schedulers[i],
        *optimal_params_list[i],
        batches=optimal_batches[i],
        epochs=epochs,
        lam=optimal_lambdas[i],
    )
    plt.plot(error_over_epochs, label=f"{schedulers[i].__name__}")
    plt.legend(loc=(1.04, 0))
plt.xlabel("Epochs", fontsize=18)
plt.ylabel("MSE", fontsize=18)
plt.title("MSE over Epochs for different schedulers", fontsize=22)
plt.show()
