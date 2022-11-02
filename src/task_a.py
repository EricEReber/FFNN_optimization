# Our own library of functions
import numpy as np
import seaborn as sns

from matplotlib.patches import Rectangle
from utils import *

np.random.seed(42069)

# implemented model under testing
# OLS_model = OLS
# scikit model under testing
# OLS_scikit = LinearRegression(fit_intercept=False)
#
# perform linear regression
# betas, z_preds_train, z_preds_test, z_preds = linreg_to_N(
#     X, X_train, X_test, z_train, z_test, N, centering=centering, model=OLS_model
# )

# perform linear regression with gradient descent
# n_iterations = 10000
# eta = 0.005
# initialize betas
# MSEs_gd = np.zeros((n_iterations))
#
# beta_gd = np.random.uniform(low=0, high=1, size=(X.shape[1]))
# for iteration in range(0, n_iterations):
#     _, z_pred_train_gd, z_pred_test_gd, z_pred_gd = gradient_descent_linreg(
#         CostOLS, X, X_train, X_test, beta_gd, z_train, eta,
#     )
#     MSEs_gd[iteration] = MSE(z_test, z_pred_test_gd)


# perform linear regression scikit
# _, z_preds_train_sk, z_preds_test_sk, _ = linreg_to_N(
#     X, X_train, X_test, z_train, z_test, N, centering=centering, model=OLS_scikit
# )

# Calculate OLS scores
# MSE_train, R2_train = scores(z_train, z_preds_train)
# MSE_test, R2_test = scores(z_test, z_preds_test)
#
# calculate OLS scikit scores without resampling
# MSE_train_sk, R2_train_sk = scores(z_train, z_preds_train_sk)
# MSE_test_sk, R2_test_sk = scores(z_test, z_preds_test_sk)
#
# approximation of terrain (2D plot)
# pred_map = z_preds[:, -1].reshape(z.shape)


# tests schedulers for a given model
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
z_train = z_train.reshape(z_train.shape[0], 1)

# no hidden layers, no activation function
dims = (X.shape[1], 1)

# hyperparameters to be gridsearched
eta = np.logspace(-4, -1, 6)
lam = np.logspace(-4, -1, 6)
lam[-1] = 0

# gridsearch eta, lambda
loss_heatmap = np.zeros((eta.shape[0], lam.shape[0]))
for i in range(eta.shape[0]):
    for j in range(lam.shape[0]):
        neural = FFNN(dims, checkpoint_file=f"write{i}{j}")
        # neural.read(f"write{i}{j}")
        error_over_epochs = neural.fit(
            X_train, z_train, Constant, eta[i], lam=lam[j], epochs=8000
        )
        loss_heatmap[i, j] = np.min(error_over_epochs)

# select optimal eta, lambda
i, j = (
    loss_heatmap.argmin() % loss_heatmap.shape[1],
    loss_heatmap.argmin() // loss_heatmap.shape[1],
)
min_eta = eta[j]
min_lam = lam[i]

# plot heatmap
ax = sns.heatmap(loss_heatmap, xticklabels=eta, yticklabels=lam, annot=True)
ax.add_patch(
    Rectangle(
        (i, j), width=1, height=1, fill=False, edgecolor="crimson", lw=4, clip_on=False
    )
)
plt.title("Loss for eta, lambda grid")
plt.xlabel("eta")
plt.ylabel("lambda")
plt.show()

# now on to scheduler specific parameters
batch_size = [X_train.shape[0] // i for i in np.linspace(1, X_train.shape[0], 5)]
momentum = [i for i in np.linspace(0, 1, 5)]
rho = 0.9
rho2 = 0.999

momentum_params = [min_eta, momentum]
adagrad_params = [min_eta, batch_size]
rms_params = [min_eta, batch_size, rho]
# adam_params = [min_eta, batch_size, rho, rho2]

params = [momentum_params, adagrad_params, rms_params]
schedulers = [
    Momentum,
    Adagrad,
    RMS_prop,
    # Adam,
]

# presume we can get error_over_epochs
for i in range(len(schedulers)):
    loss_search = np.zeros(len(params[i][1]))
    for k in range(len(params[i][1])):
        neural = FFNN(dims, checkpoint_file=f"loss_search{i}{k}")
        test_params = params[i][:]
        test_params[1] = params[i][1][k]
        error_over_epochs = neural.fit(
            X_train,
            z_train,
            schedulers[i],
            *test_params,
            batches=10,
            epochs=5000,
            lam=min_lam,
        )
        loss_search[k] = np.min(error_over_epochs)
    params[i][1] = params[i][1][np.argmin(loss_search)]

# presume we can get error_over_epochs
for i in range(len(schedulers)):
    neural = FFNN(dims, checkpoint_file=f"comparison{i}")
    error_over_epochs = neural.fit(
        X_train,
        z_train,
        schedulers[i],
        *params[i],
        batches=10,
        epochs=5000,
        lam=min_lam,
    )
    plt.plot(error_over_epochs, label=f"{schedulers[i]}")
    plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("MSE over Epochs for different schedulers")
plt.show()

z_pred = neural.predict(X)

pred_map = z_pred.reshape(z.shape)


# ------------ PLOTTING 3D -----------------------
fig = plt.figure(figsize=plt.figaspect(0.3))

# Subplot for terrain
ax = fig.add_subplot(121, projection="3d")
# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
ax.set_title("Scaled terrain", size=24)
# Add a color bar which maps values to colors.
# fig.colorbar(surf_real, shrink=0.5, aspect=5)

# Subplot for the prediction
# Plot the surface.
ax = fig.add_subplot(122, projection="3d")
# Plot the surface.
surf = ax.plot_surface(
    x,
    y,
    pred_map,
    cmap=cm.coolwarm,
    linewidth=0,
    antialiased=False,
)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
ax.set_title(f"Neural netbork *wuff* *wuff*", size=24)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
