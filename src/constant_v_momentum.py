from utils import *
from Schedulers import *
from FFNN import FFNN
import timeit



"""
The following script first performs a gridsearch for optimal parameters lambda and eta for use 
with regular full-batch gradient descent and momentum gradient descent. It the proceeds with the 
measurement of the runtime of each of said methods, before it finally prints said runtime and 
test error acieved by each technique for the sake of comparison of both schedulers against eachother.
"""
# ---------------------- Loading data ---------------------- 
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

# --------------------- Setting params --------------------- 
epochs = 200
folds = 5

full_batch = 1
eta = np.logspace(-5, -1, 5)
lam = np.logspace(-5, -1, 5)

schedulers = [Momentum, Constant]
constant_params = []
momentum_params = np.linspace(0, 0.1, 5)
params = [momentum_params, constant_params]

# ------------------ Linear Regression NN ------------------
dims = (X.shape[1], 1)
neural = FFNN(dims, seed=1337)

batches_list = np.logspace(
    0, np.log(X_train.shape[0] + 1), 7, base=np.exp(1), dtype=int
)

sns.set(font_scale=3)
# Gridsearching parameters learning rate and lambda for "Constant" and Momentum scheduler
for i in range(len(schedulers)):
    optimal_params, optimal_lambda, _ = neural.optimize_scheduler(
        X_train,
        z_train,
        schedulers[i],
        eta,
        lam,
        params[i],
        batches=full_batch,
        epochs=epochs,
        folds=folds,
    )

    # Measurement of average runtime of our FFNN model
    start = timeit.default_timer()
    constant_scores = neural.cross_val(
        folds,
        X_train,
        z_train,
        schedulers[i],
        *optimal_params,
        batches=full_batch,
        epochs=epochs,
        lam=optimal_lambda,
    )
    stop = timeit.default_timer()
    runtime = (stop - start) / folds
    print(f"Constant time to train {epochs} epochs: {runtime}")
    test_errors = constant_scores["test_errors"]
    plt.figure(1)
    if schedulers[i].__name__ == "Constant":
        plt.plot(test_errors, "b--", label=f"{schedulers[i].__name__} test", linewidth=5)
    else:
        plt.plot(test_errors, "r", label=f"{schedulers[i].__name__} test", linewidth=5)

    
    # force momentum to random value between 0 and 1, since optimal for full batch == 0
    if schedulers[i].__name__ == "Momentum":
        optimal_params = [optimal_params[0], 0.3]

    _, batches_list_search = neural.optimize_batch(
        X_train,
        z_train,
        X_test,
        z_test,
        schedulers[i],
        optimal_lambda,
        *optimal_params,
        batches_list=batches_list,
        epochs=epochs,
    )
    plt.figure(2)
    colors = ["r", "g", "y", "b", "k", "c", "m", "w"]
    for j in range(len(batches_list)):
        if schedulers[i].__name__ == "Constant":
            plt.plot(
                batches_list_search[j, :], f"{colors[j]}--", linewidth=5,
                label=f"{schedulers[i].__name__} batch size {X_train.shape[0]//batches_list[j]}",
            )
        else:
            plt.plot(
                batches_list_search[j, :], colors[j], linewidth=3,
                label=f"{schedulers[i].__name__} batch size {X_train.shape[0]//batches_list[j]}",
            )

# plot
plt.figure(1)
plt.legend(loc="upper right")
plt.xlabel("epochs", fontsize=22)
plt.ylabel("MSE", fontsize=22)
plt.title("Constant vs Momentum MSE over epochs", fontsize=32)

sns.set(font_scale=2)
plt.figure(2)
plt.legend(loc=(1.04, 0))
plt.xlabel("epochs", fontsize=22)
plt.ylabel("MSE", fontsize=22)
plt.title("Constant vs Momentum MSE over epochs \n for different batch sizes", fontsize=32)
plt.show()
