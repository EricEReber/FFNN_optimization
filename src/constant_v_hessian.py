from utils import *
from Schedulers import *
from FFNN import FFNN
import timeit
from tabulate import tabulate

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

# define params
epochs = 200
folds = 5

full_batch = 1
eta = np.logspace(-5, -1, 5)
lam = np.logspace(-5, -1, 5)

constant_scheduler = Constant
constant_params = []

# linear regression NN
dims = (X.shape[1], 1)
neural = FFNN(dims, seed=1337)

# optimize constant learning rate, regularization
optimal_params, optimal_lambda, _ = neural.optimize_scheduler(
    X_train,
    z_train,
    constant_scheduler,
    eta,
    lam,
    constant_params,
    batches=full_batch,
    epochs=epochs,
    folds=folds,
)

# take average runtime constant
time = np.zeros(2)
start = timeit.default_timer()
constant_scores = neural.cross_val(
    folds,
    X_train,
    z_train,
    constant_scheduler,
    *optimal_params,
    batches=full_batch,
    epochs=epochs,
    lam=optimal_lambda,
)
stop = timeit.default_timer()
time[0] = (stop - start) / folds
print(f"Constant time to train {epochs} epochs: {time[0]}")
constant_test_errors = constant_scores["test_errors"]

# take average runtime Newton's
start = timeit.default_timer()
hessian_scores, _ = hessian_cv(folds, X_train, z_train, epochs=epochs)
stop = timeit.default_timer()
time[1] = (stop - start) / folds
print(f"Hessian time to train {epochs} epochs: {time[1]}")
hessian_test_errors = hessian_scores["test_errors"]

# plot
plt.plot(constant_test_errors, label="Constant test")
plt.plot(hessian_test_errors, label="Hessian test")
plt.legend(loc="upper right")
plt.xlabel("epochs", fontsize=18)
plt.ylabel("MSE", fontsize=18)
plt.title("Constant vs Hessian MSE over epochs", fontsize=22)
plt.show()
