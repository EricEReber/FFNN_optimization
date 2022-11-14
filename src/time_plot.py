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
epochs = 100
folds = 5

# batches to test for
batches_list = np.logspace(
    0, np.log(X_train.shape[0] + 1), 7, base=np.exp(1), dtype=int
)

schedulers = [Constant, Momentum, Adagrad, AdagradMomentum, RMS_prop, Adam]
# parameters dont matter as we just want to test runtime
optimal_params = [[0.01], [0.01, 0.1], [0.1], [0.1, 0.075], [0.001, 0.9], [0.01, 0.9, 0.999]]
optimal_lambda = 0.1

# no hidden layers, no activation function
dims = (X.shape[1], 1)
neural = FFNN(dims, seed=1337)

time = np.zeros(len(schedulers))

for i in range(len(schedulers)):
    start = timeit.default_timer()
    scores = neural.cross_val(
        folds,
        X_train,
        z_train,
        schedulers[i],
        *optimal_params[i],
        batches=batches_list[3],
        epochs=epochs,
        lam=optimal_lambda,
    )
    stop = timeit.default_timer()
    time[i] = stop - start

time /= folds
table = [schedulers, time]

string = tabulate(table)
with open(f"time_comparison", "w") as file:
    file.write(string)
