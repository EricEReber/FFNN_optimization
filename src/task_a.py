"""
task b (and task g): plot terrain, approximate terrain with OLS (own implementation and scikit) and calculate MSE, R2 &
                     beta over model complexity for real data. Performs task_b, so no resampling.
"""
# Our own library of functions
from utils import *

np.random.seed(42069)

# get data
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

# implemented model under testing
OLS_model = OLS
# scikit model under testing
OLS_scikit = LinearRegression(fit_intercept=False)

# perform linear regression
betas, z_preds_train, z_preds_test, z_preds = linreg_to_N(
    X, X_train, X_test, z_train, z_test, N, centering=centering, model=OLS_model
)

# perform linear regression with gradient descent
n_iterations = 10000
eta = 0.005
# initialize betas
MSEs_gd = np.zeros((n_iterations))

beta_gd = np.random.uniform(low=0, high=1, size=(X.shape[1]))
for iteration in range(0, n_iterations):
    _, z_pred_train_gd, z_pred_test_gd, z_pred_gd = gradient_descent_linreg(
        CostOLS, X, X_train, X_test, beta_gd, z_train, eta,
    )
    MSEs_gd[iteration] = MSE(z_test, z_pred_test_gd)


# perform linear regression scikit
_, z_preds_train_sk, z_preds_test_sk, _ = linreg_to_N(
    X, X_train, X_test, z_train, z_test, N, centering=centering, model=OLS_scikit
)

# Calculate OLS scores
MSE_train, R2_train = scores(z_train, z_preds_train)
MSE_test, R2_test = scores(z_test, z_preds_test)

# calculate OLS scikit scores without resampling
MSE_train_sk, R2_train_sk = scores(z_train, z_preds_train_sk)
MSE_test_sk, R2_test_sk = scores(z_test, z_preds_test_sk)

# approximation of terrain (2D plot)
pred_map = z_preds[:, -1].reshape(z.shape)

# ------------ PLOTTING 3D -----------------------
fig = plt.figure(figsize=plt.figaspect(0.3))

# Subplot for terrain
ax = fig.add_subplot(221, projection="3d")
# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
ax.set_title("Scaled terrain", size=24)
# Add a color bar which maps values to colors.
# fig.colorbar(surf_real, shrink=0.5, aspect=5)

# Subplot for the prediction
# Plot the surface.
ax = fig.add_subplot(222, projection="3d")
# Plot the surface.
surf = ax.plot_surface(
    x,
    y,
    np.reshape(z_preds[:, N], z.shape),
    cmap=cm.coolwarm,
    linewidth=0,
    antialiased=False,
)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
ax.set_title(f"Polynomial fit of scaled terrain, N = {N}", size=24)
fig.colorbar(surf, shrink=0.5, aspect=5)

ax = fig.add_subplot(223, projection="3d")
# Plot the surface.
surf = ax.plot_surface(
    x,
    y,
    np.reshape(z_pred_gd, z.shape),
    cmap=cm.coolwarm,
    linewidth=0,
    antialiased=False,
)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
ax.set_title(f"Gradient descent fit of scaled terrain, N = {N}", size=24)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# ---------------- PLOTTING GRAPHS --------------
MSEs_matrix_inversion = np.zeros(MSEs_gd.shape)
MSEs_matrix_inversion[:] = MSE_test[-1]
plt.plot(range(n_iterations), MSEs_gd, label="test gradient descent")
plt.plot(
    range(n_iterations),
    MSEs_matrix_inversion,
    "r--",
    label="test matrix inversion",
)
plt.ylabel("MSE score", size=15)
plt.xlabel("Iterations of Gradient Descent", size=15)
plt.title(f"MSE scores over Gradient Descent iterations\n eta={eta}", size=18)
plt.legend()
plt.show()
