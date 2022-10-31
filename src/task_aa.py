"""
task b (and task g): plot terrain, approximate terrain with OLS (own implementation and scikit) and calculate MSE, R2 &
                     beta over model complexity for real data. Performs task_b, so no resampling.
"""
# Our own library of functions
from utils import *

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

    dims = (2, 40, 40, 1)

    # dims = (4, 3, 2, 2)

    dummy_x = np.array(
        [
            [0.134, 0.91827, 0.1982, 0.34654],
            [0.7246, 0.8887, 0.1513, 0.97716],
            [0.441, 0.123, 0.321, 0.71],
        ]
    )
    dummy_t = np.array([[1, 2, 3], [1, 2, 3]]).T

    z_train = z_train.reshape(z_train.shape[0], 1)

    eta = 0.001
    batch_size = X_train.shape[0] // 10
    momentum = 0.5
    rho = 0.95
    rho2 = 0.9

    constant_params = [eta]
    momentum_params = [eta, momentum]
    adagrad_params = [eta, batch_size]
    rms_params = [eta, batch_size, rho]
    adam_params = [eta, batch_size, rho, rho2]

    params = [momentum_params, adagrad_params, rms_params, adam_params, constant_params]
    params = [rms_params]
    schedulers = [
        # Momentum,
        # Adagrad,
        RMS_prop,
        # Adam,
        # Constant,
    ]
    # presume we can get error_over_epochs
    for i in range(len(schedulers)):
        neural = FFNN(dims, checkpoint_file="weights")
        neural.read("weights")
        error_over_epochs = neural.fit(
            X_train[:, 1:3],
            z_train,
            schedulers[i],
            *params[i],
            batches=10,
            epochs=1000,
        )
        plt.plot(error_over_epochs, label=f"{schedulers[i]}")
        plt.legend()
    neural.write("weights")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("MSE over Epochs for different schedulers")
    plt.show()

    z_pred = neural.predict(X[:, 1:3])

    pred_map = z_pred.reshape(z.shape)

    return pred_map, x, y, z


pred_map, x, y, z = test_scheduler()

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
