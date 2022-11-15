# Our own library of functions
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.patches import Rectangle
from utils import *
from Schedulers import *
from FFNN import FFNN
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import time

np.random.seed(1337)

# read in data
cancer = load_breast_cancer()

X = cancer.data
z = cancer.target
z = z.reshape(z.shape[0], 1)

# epochs to run for
epochs = 100
folds = 5


# parameters to test for
eta = np.logspace(-5, -1, 5)
lam = np.logspace(-5, -1, 5)
lam[0] = 0
momentums = np.linspace(0, 0.1, 5)
rho = 0.9
rho2 = 0.999

# batches to test for
batches_list = np.logspace(0, np.log(X.shape[0] + 1), 7, base=np.exp(1), dtype=int)
# schedulers to test for
sched = Adam
adam_params = [rho, rho2]

# adam_params = [0.005, rho, rho2]

X_train, X_test, t_train, t_test = train_test_split(X, z)

hessianscores, _ = hessian(
    X_train,
    t_train,
    epochs=epochs,
    X_test=X_test,
    t_test=t_test,
)

# dims = (30, 60, 1)
dims = (30, 60, 1)
neural = FFNN(
    dims, hidden_func=RELU, output_func=sigmoid, cost_func=CostLogReg, seed=1337
)
# optimal_params, optimal_lambda, loss_heatmap = neural.optimize_scheduler(
#     X,
#     z,
#     sched,
#     eta,
#     lam,
#     adam_params,
#     batches=7,
#     epochs=epochs // 2,
#     folds=folds,
# )

# optimal params found to be:
optimal_params = [0.01, 0.9, 0.999]
optimal_lambda = 0


start = time.time()

print(optimal_params)
print(optimal_lambda)

neuralscores = neural.cross_val(
    folds,
    X,
    z,
    Adam,
    *optimal_params,
    batches=7,
    epochs=epochs,
    lam=optimal_lambda,
    use_best_weights=True,
)
print(f"Time taken: {time.time() - start:.4f}")
plot_confusion(
    neuralscores["confusion"],
    title="Confusion matrix for cancer data using a neural network",
)


dims = (30, 1)
logreg = FFNN(dims, output_func=sigmoid, cost_func=CostLogReg, seed=1337)

# optimal_params, optimal_lambda, loss_heatmap = logreg.optimize_scheduler(
#     X,
#     z,
#     sched,
#     eta,
#     lam,
#     adam_params,
#     batches=7,
#     epochs=epochs // 2,
#     folds=folds,
# )

# optimal params found to be:
optimal_params = [0.1, 0.9, 0.999]
optimal_lambda = 0

start = time.time()
logregscores = logreg.cross_val(
    folds,
    X,
    z,
    Adam,
    *optimal_params,
    batches=7,
    epochs=epochs,
    lam=optimal_lambda,
    use_best_weights=True,
)
print(f"Time taken: {time.time() - start}")


plot_confusion(
    logregscores["confusion"],
    title="Confusion matrix for cancer data using logistic regression",
)


plt.title("Crossvalidated test accuracy per epoch")
plt.plot(neuralscores["test_accs"], label="Neural", lw=4)
plt.plot(logregscores["test_accs"], label="Logreg", lw=4)
plt.plot(hessianscores["test_accs"], label="Hessian", lw=4)
plt.xlabel("Epochs")
plt.ylabel("Test accuracy")
plt.legend()
plt.show()
