from utils import *
from Schedulers import *
from FFNN import FFNN
from sklearn.neural_network import MLPRegressor

np.random.seed(1337)
(
    betas_to_plot,
    N,
    X,
    X_train,
    X_test,
    t,
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
eta = 0.01
rho = 0.9
rho2 = 0.999
sched = Adam
params = [eta, rho, rho2]

#dims = (2, 266, 133, 1) # second best
#dims = (2, 133, 133, 1)
#dims = (2, 320, 160, 1) # second best 
#dims = (2, 133, 133, 133, 1) # best so far 
#dims = (2, 133, 133, 133, 1)
dims = (2, 66,66,66, 1) 

train_epochs = 100


# ------------------------- FFNN -------------------------
neural = FFNN(dims, hidden_func=LRELU, seed=1337)
"""
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
"""
scores = neural.cross_val(
    5,
    X[:, 1:3],
    z.reshape(400, 1) ,
    sched,
    *params,
    batches=X_train.shape[0],
    epochs=train_epochs,
    lam=10e-6,
    #X_test=X_test[:, 1:3],
    #t_test=z_test,
)

print(f"Final test_error: {scores['final_test_error']}")

# ------------------------- MLPRegressor -------------------------
mlp = MLPRegressor(
    hidden_layer_sizes=dims,
    activation="logistic",
    solver="adam",
    max_iter=train_epochs,
    n_iter_no_change=train_epochs * 10,
    batch_size=X_train.shape[0] // 30,
)

mlp.fit(X_train[:, 1:3], np.ravel(z_train))
mlp_errors = mlp.loss_curve_

# ------------------------- MSE plotting -------------------------
plt.plot(scores["train_errors"], label="Train_FNN")
plt.plot(scores["test_errors"], label="Test_FNN")
plt.plot(mlp_errors, label="scikit_error")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("MSE over Epochs")
plt.show()

z_pred = neural.predict(X[:, 1:3])

pred_map = z_pred.reshape(z.shape)

plot_terrain(x, y, z, pred_map)
