from utils import *
from Schedulers import *
from FFNN import FFNN

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

eta = 0.1
momentum = 1
rho = 0.9
rho2 = 0.99
epochs = 1000

sched = Momentum
params = [eta, momentum]

# hessianscores, _ = hessian(
#     X_train,
#     z_train,
#     epochs=epochs,
#     X_test=X_test,
#     t_test=z_test,
# )

dims = (X_train.shape[1], 1)
neural = FFNN(dims, hidden_func=RELU)
scores = neural.fit(
    X_train,
    z_train,
    sched,
    *params,
    batches=320,
    epochs=epochs,
    lam=0.0,
    X_test=X_test,
    t_test=z_test,
)

plt.plot(scores["train_errors"], label="Train")
plt.plot(scores["test_errors"], label="Train")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("MSE over Epochs")
plt.show()

z_pred = neural.predict(X[:, 1:3])

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
