from Schedulers import *
from FFNN import FFNN
from utils import *

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

dims = (X.shape[1], 1)
eta = np.logspace(-5, -1, 5)
lam = np.logspace(-5, -1, 5)
lam[0] = 0

epochs = 1000
neural = FFNN(dims)

loss_heatmap = np.zeros((eta.shape[0], lam.shape[0]))
for y in range(eta.shape[0]):
    for x in range(lam.shape[0]):
        _, test_error, _, _ = neural.fit(
            X_train,
            z_train,
            Constant,
            eta[y],
            epochs=epochs,
            batches=1,
            lam=lam[x],
            X_test=X_test,
            t_test=z_test,
        )
        loss_heatmap[y, x] = test_error[-1]
        neural._reset_weights()

# select optimal eta, lambda
y, x = (
    loss_heatmap.argmin() // loss_heatmap.shape[1],
    loss_heatmap.argmin() % loss_heatmap.shape[1],
)

optimal_eta = eta[y]
optimal_lambda = lam[x]


train_error, test_error, _, _ = neural.fit(
    X_train,
    z_train,
    Constant,
    optimal_eta,
    epochs=epochs,
    batches=1,
    lam=optimal_lambda,
    X_test=X_test,
    t_test=z_test,
)

ax = sns.heatmap(loss_heatmap, xticklabels=lam, yticklabels=eta, annot=True)
ax.add_patch(
    Rectangle(
        (x, y),
        width=1,
        height=1,
        fill=False,
        edgecolor="crimson",
        lw=4,
        clip_on=False,
    )
)
plt.xlabel("lambda", fontsize=18)
plt.ylabel("eta", fontsize=18)
plt.title(f"Gridsearch lambda, eta for Constant scheduler", fontsize=22)
plt.show()

plt.plot(test_error, label=f"test Constant")
plt.plot(train_error, label=f"train Constant")
best_MSE_analytically = np.zeros(epochs)
best_MSE_analytically[:] = 0.003027
plt.plot(best_MSE_analytically, label="analytical solution MSE")
plt.legend(loc=(1.04, 0))
plt.xlabel("Epochs", fontsize=18)
plt.ylabel("MSE", fontsize=18)
plt.title("MSE over Epochs for Constant scheduler", fontsize=22)
plt.show()
