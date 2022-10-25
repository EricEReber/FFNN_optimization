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

# approximation of terrain (2D plot)

dims = (2, 20, 20, 20, 1)

z_train = FrankeFunction(X_train[:, 1], X_train[:, 2])
z_train = z_train.reshape(z_train.shape[0], 1)
print(f"{X_train[:, 1:3]=}")

neural = FFNN(dims, epochs=1000)
neural.fit(X_train[:, 1:3], z_train, scheduler=Scheduler(0.3))
z_pred = neural.predict(X[:, 1:3])
print(z_pred)

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
ax.set_title(f"Polynomial fit of scaled terrain, N = {N}", size=24)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
