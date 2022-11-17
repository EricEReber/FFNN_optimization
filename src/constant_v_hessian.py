# Our own library
from utils import *
from Schedulers import *
from FFNN import FFNN
# Other libraries
import timeit
from tabulate import tabulate


"""
The following script first performs a gridsearch for optimal parameters lambda and eta for use 
with regular full-batch gradient descent. It the proceeds with the measurement of the runtime 
of our FFNN and Newton-Raphon's method, before it finally prints said runtime and test error 
acieved by each technique for the sake of comparison of both methods against eachother.
"""

np.random.seed(1337)

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

constant_scheduler = Constant
constant_params = []

# ------------------ Linear Regression NN ------------------
dims = (X.shape[1], 1)
neural = FFNN(dims, seed=1337)

# Gridsearching parameters learning rate and lambda for "Constant" scheduler, i.e full-batch gradient descent
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

# Measurement of average runtime of our FFNN model
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

# Measurement of average runtime of Newton-Raphson's method 
start = timeit.default_timer()
hessian_scores, _ = hessian_cv(folds, X_train, z_train, epochs=epochs)
stop = timeit.default_timer()
time[1] = (stop - start) / folds
print(f"Hessian time to train {epochs} epochs: {time[1]}")
hessian_test_errors = hessian_scores["test_errors"]

# ----------------------- Plots -----------------------
sns.set(font_scale=3)
plt.plot(constant_test_errors, label="Constant test", linewidth=5)
plt.plot(hessian_test_errors, label="Hessian test", linewidth=5)
plt.legend(loc="upper right")
plt.xlabel("epochs", fontsize=32)
plt.ylabel("MSE", fontsize=32)
plt.title("Constant vs Hessian MSE over epochs", fontsize=42)
plt.show()
