"""
task b (and task g): plot terrain, approximate terrain with OLS (own implementation and scikit) and calculate MSE, R2 &
                     beta over model complexity for real data. Performs task_b, so no resampling.
"""
# Our own library of functions
from utils import *

np.random.seed(42069)

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)

xs, ys = np.meshgrid(x, y)

X = np.dstack(np.meshgrid(x, y)).reshape(-1, 2)

zs = FrankeFunction(xs, ys)

z = FrankeFunction(X[:, 0], X[:, 1]).reshape(X.shape[0], 1)

dims = (2, 20, 1)
np.seterr(all="raise")

neural = FFNN(dims, epochs=1000)
neural.fit(X, z, scheduler=Scheduler(0.01))
z_pred = neural.predict(X)
print(z_pred)

pred_map = z_pred.reshape(zs.shape)

# ------------ PLOTTING 3D -----------------------
fig = plt.figure(figsize=plt.figaspect(0.3))

# Subplot for terrain
ax = fig.add_subplot(121, projection="3d")
# Plot the surface.
surf = ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
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
    xs,
    ys,
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
