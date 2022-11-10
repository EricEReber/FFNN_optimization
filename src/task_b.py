from utils import *
from Schedulers import *
from FFNN import FFNN
from sklearn.neural_network import MLPRegressor

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

# ------------------------- Params -------------------------
eta = 0.00005
momentum = 0.5
rho = 0.9
rho2 = 0.99
sched = Adam
# sched = Momentum
params = [eta, rho, rho2]
opt_params = [rho, rho2]
# params = [eta, momentum]

dims = (2, 224, 112, 1)
train_epochs = 1000

eta = np.logspace(-5, -1, 5)
lams = np.logspace(-5, -1, 5)
batch_sizes = np.linspace(1, X.shape[0] // 2, 5, dtype=int)

# ------------------------- FFNN -------------------------
neural = FFNN(dims, checkpoint_file="weights", hidden_func=RELU)

optimal_params, optimal_lambda, _ = neural.optimize_scheduler(
    X_train[:, 1:3],
    z_train,
    X_test[:, 1:3],
    z_test,
    sched,
    eta,
    lams,
    opt_params,
    batches=25,
    epochs=1000,
)

params = [optimal_params[0], rho, rho2]

optimal_batch = neural.optimize_batch(
    X_train[:, 1:3],
    z_train,
    X_test[:, 1:3],
    z_test,
    sched,
    optimal_lambda,
    *params,
    batches_list=batch_sizes,
    epochs=30,
)

scores = neural.fit(
    X_train[:, 1:3],
    z_train,
    sched,
    *params,
    batches=optimal_batch[0],
    epochs=train_epochs,
    lam=optimal_lambda,
    X_test=X_test[:, 1:3],
    t_test=z_test,
)

# ------------------------- MLPRegressor -------------------------
mlp = MLPRegressor(
    hidden_layer_sizes=dims,
    activation="logistic",
    solver="adam",
    max_iter=train_epochs,
    tol=0.05,
    n_iter_no_change=train_epochs * 10,
    batch_size=X_train.shape[0] // 30,
)

mlp.fit(X_train[:, 1:3], np.ravel(z_train))
mlp_errors = mlp.loss_curve_

# ------------------------- MSE plotting -------------------------
plt.plot(scores["train_error"], label="Train_FNN")
plt.plot(scores["test_error"], label="Test_FNN")
plt.plot(mlp_errors, label="scikit_error")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("MSE over Epochs")
plt.show()

z_pred = neural.predict(X[:, 1:3])

pred_map = z_pred.reshape(z.shape)

plot_terrain(x, y, z, pred_map)
