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

# epochs to run for
epochs = 50
folds = 5

# batches to test for
batches_list = np.logspace(
    0, np.log(X_train.shape[0] + 1), 7, base=np.exp(1), dtype=int
)

schedulers = [Constant, Momentum, Adagrad, AdagradMomentum, RMS_prop, Adam]
# optimal parameters from task_a.py
optimal_params = [[0.01], [0.01, 0.1], [0.1], [0.1, 0.075], [0.001, 0.9], [0.01, 0.9, 0.999]]
optimal_lambdas = [0.1, 0.1, 0.001, 0.001, 0.001, 0.01]

# no hidden layers, no activation function
dims = (X.shape[1], 1)
neural = FFNN(dims, seed=1337)

time = np.zeros(len(schedulers))
MSE_to_reach = 0.2

for i in range(len(schedulers)):
    optimal_batch, _ = neural.optimize_batch(
        X_train,
        z_train,
        X_test,
        z_test,
        schedulers[i],
        optimal_lambdas[i],
        *optimal_params[i],
        batches_list=batches_list,
        epochs=epochs,
    )
    start = timeit.default_timer()
    scores = neural.cross_val(
        folds,
        X_train,
        z_train,
        schedulers[i],
        *optimal_params[i],
        batches=optimal_batch,
        epochs=epochs,
        lam=optimal_lambdas[i],
    )
    index=0
    test_errors = scores["test_errors"]
    for j in range(epochs):
        if test_errors[j] <= MSE_to_reach:
            index = j
            break
    stop = timeit.default_timer()
    runtime = stop - start
    runtime /= epochs
    time[i] = runtime * index

time /= folds
table = [schedulers, time]

string = tabulate(table)
with open(f"time_comparison", "w") as file:
    file.write(string)
