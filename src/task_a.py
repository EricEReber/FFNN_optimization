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
    np.reshape(z_preds[:, N], z.shape),
    cmap=cm.coolwarm,
    linewidth=0,
    antialiased=False,
)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
ax.set_title(f"Polynomial fit of scaled terrain, N = {N}", size=24)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# ---------------- PLOTTING GRAPHS --------------
plt.subplot(121)
plt.title("Terrain 1")
plt.imshow(z)
plt.colorbar()
plt.subplot(122)
plt.title("Predicted terrain 1")
plt.imshow(pred_map)
plt.colorbar()
plt.show()

for beta in range(betas_to_plot):
    data = betas[beta, :]
    data[data == 0] = np.nan
    plt.plot(data, label=f"beta{beta}", marker="o", markersize=3)
    plt.xlabel("Polynomial degree (N)", size=15)
    plt.ylabel("Beta value", size=15)
    plt.title("Beta progression", size=22)
    plt.legend()
plt.show()

print(f"Minimal MSE_test value = {np.min(MSE_test)} for N = {np.argmin(MSE_test)}")
print(f"Beta vector for final regression: {betas[:,N]}")
plt.plot(MSE_train, label="train implementation", marker="o", markersize=3)
plt.plot(MSE_test, label="test implementation", marker="o", markersize=3)
plt.plot(MSE_train_sk, "r--", label="train ScikitLearn", marker="o", markersize=3)
plt.plot(MSE_test_sk, "g--", label="test ScikitLearn", marker="o", markersize=3)
plt.ylabel("MSE score", size=15)
plt.xlabel("Polynomial degree (N)", size=15)
plt.title("MSE scores over model complexity", size=22)
plt.legend()
plt.show()

plt.plot(R2_train, label="train implementation", marker="o", markersize=3)
plt.plot(R2_test, label="test implementation", marker="o", markersize=3)
plt.plot(R2_train_sk, "r--", label="train ScikitLearn", marker="o", markersize=3)
plt.plot(R2_test_sk, "g--", label="test ScikitLearn", marker="o", markersize=3)
plt.ylabel("R2 score")
plt.xlabel("Polynomial degree (N)")
plt.title("R2 scores over model complexity")
plt.legend()
plt.show()
