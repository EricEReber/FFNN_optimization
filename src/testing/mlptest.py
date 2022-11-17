"""
task b (and task g): plot terrain, approximate terrain with OLS (own implementation and scikit) and calculate MSE, R2 &
                     beta over model complexity for real data. Performs task_b, so no resampling.
"""
# Our own library of functions
from utils import *
from FFNN import FFNN
from Schedulers import *
from sklearn.neural_network import MLPRegressor

np.random.seed(42069)

# define model (should probably be sent in)
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

dims = (X_train.shape[1], 20, 20, 20, 1)

batches = 15
batch_size = X_train.shape[0] // batches

mlp = MLPRegressor(
    hidden_layer_sizes=dims[1:4],
    activation="logistic",
    solver="adam",
    max_iter=1000,
    # learning_rate="adaptive",
    tol=0,
    n_iter_no_change=10000000,
    batch_size=batch_size,
)

neural = FFNN(dims)


z_train = FrankeFunction(X_train[:, 1], X_train[:, 2])
z_train = z_train.reshape(z_train.shape[0], 1)

mlp.fit(X_train, np.ravel(z_train))
mlp_errors = mlp.loss_curve_

eta = 0.001
rho = 0.9
rho2 = 0.999
adam_args = [eta, rho, rho2]

train_errors, _ = neural.fit(X_train, z_train, Adam, *adam_args, batches=batches)

plt.plot(train_errors, label="own")
plt.plot(mlp_errors, label="scikit")
plt.legend()
plt.show()
