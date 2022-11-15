from utils import *
from Schedulers import *
from FFNN import FFNN
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
np.random.seed(1337)


# ------------------------- Params -------------------------
eta = 0.00005
momentum = 0.5
rho = 0.9
rho2 = 0.99
sched = Adam
# sched = Momentum
params = [eta, rho, rho2]
opt_params = [rho, rho2]
# params = [eta, momentum]

dims = (80)
train_epochs = 1000

eta = np.logspace(-5, -1, 5)
lams = np.logspace(-5, -1, 5)
# batch_sizes = np.linspace(1, X.shape[0] // 2, 5, dtype=int)

cancer = load_breast_cancer()

X = cancer.data
z = cancer.target
# z = z.reshape(z.shape[0], 1)

# ------------------------- MLPRegressor -------------------------
mlp = MLPClassifier(
    hidden_layer_sizes=dims,
    learning_rate='constant',
    learning_rate_init=0.0002, 
    activation="relu",
    solver="adam",
    max_iter=train_epochs,
    n_iter_no_change=train_epochs * 10,
    batch_size=20,
)

cv_results = cross_val_score(mlp, X, z, cv=5)


print(f'Fit score: {cv_results.mean()}')
# ------------------------- MSE plotting -------------------------
plt.plot(mlp_errors, label="scikit_error")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("MSE over Epochs")
plt.show()
pred = mlp.predict(X)
print(accuracy(z, pred))
