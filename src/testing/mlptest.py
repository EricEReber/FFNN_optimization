"""
task b (and task g): plot terrain, approximate terrain with OLS (own implementation and scikit) and calculate MSE, R2 &
                     beta over model complexity for real data. Performs task_b, so no resampling.
"""
# Our own library of functions
from utils import *
from sklearn.neural_network import MLPRegressor

np.random.seed(42069)

# tests schedulers for a given model
def test_scheduler():
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

    dummy_x = np.array(
        [
            [0.134, 0.91827, 0.1982, 0.34654],
            [0.7246, 0.8887, 0.1513, 0.97716],
            [0.441, 0.123, 0.321, 0.71],
        ]
    )
    dims = (2, 20, 20, 20, 1)

    mlp = MLPRegressor(
        hidden_layer_sizes=dims,
        activation="tanh",
        solver="adam",
        max_iter=10000,
        learning_rate="adaptive",
        tol=0,
        n_iter_no_change=10000000,
    )

    z_train = FrankeFunction(X_train[:, 1], X_train[:, 2])
    z_train = z_train.reshape(z_train.shape[0], 1)

    mlp.fit(X_train, np.ravel(z_train))
    z_pred = mlp.predict(X)
    print(mlp.n_iter_)

    pred_map = z_pred.reshape(z.shape)

    return pred_map, x, y, z


pred_map, x, y, z = test_scheduler()
print(pred_map)

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
