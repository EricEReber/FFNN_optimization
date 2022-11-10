from utils import *
from FFNN import FFNN
from Schedulers import *

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
z = z.ravel()
z = z.reshape(z.shape[0], 1)
dims = (X_train.shape[1], 1)
neural = FFNN(dims, seed=1337)

batches = X.shape[0] // 8
eta = 0.1
momentum = 1
lam = 0.0
sched = Constant
params = [eta]
print(batches)

epochs = 1000
K = 7
matrices = crossval(X, z, K)

test_errors, biases, variances = neural.bootstrap(
    20,
    X_train,
    z_train,
    sched,
    *params,
    X_test=X_test,
    t_test=z_test,
    batches=batches,
    epochs=epochs
)
# train_errors = np.zeros(epochs)
# test_errors = np.zeros(epochs)
#
# for i in range(len(matrices)):
#     scores = neural.fit(
#         matrices[i][0],
#         matrices[i][2],
#         sched,
#         *params,
#         X_test=matrices[i][1],
#         t_test=matrices[i][3],
#         lam=lam,
#         batches=batches,
#         epochs=epochs
#     )
#
#     train_errors += scores["train_error"]
#     test_errors += scores["test_error"]
#
# train_errors /= K
# test_errors /= K
#
# plt.plot(train_errors, "r--", label="Crossval train")
# plt.plot(test_errors, label="Crossval test")
# plt.legend()
# plt.show()

# B = 10
# matrices = bootstrap(X_train, z_train, B)
#
# test_errors = np.zeros(epochs)
# predictions = np.zeros((X_test.shape[0], B, epochs))
# biases = np.zeros(epochs)
# variances = np.zeros(epochs)
#
# for i in range(len(matrices)):
#     scores = neural.fit(
#         matrices[i][0],
#         matrices[i][1],
#         sched,
#         *params,
#         X_test=X_test,
#         t_test=z_test,
#         batches=batches,
#         epochs=epochs
#     )
#     neural.reset_weights()
#
#     predictions[:, i, :] = scores["test_pred"]
#
# for i in range(epochs):
#     if not i:
#         print(predictions[:, :, i])
#     error, bias, variance = bias_variance(z_test, predictions[:, :, i])
#     biases[i] = bias
#     variances[i] = variance
#     test_errors[i] = error


plt.plot(test_errors, label="Boooootstrap test >:0")
plt.plot(biases, label="biases")
plt.plot(variances, label="variances")
plt.legend()
plt.show()
