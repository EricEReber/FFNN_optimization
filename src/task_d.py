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

np.random.seed(1337)

# read in data
cancer = load_breast_cancer()

X = cancer.data
z = cancer.target

X_train, X_test, z_train, z_test = train_test_split(X, z)

z_train = z_train.reshape(z_train.shape[0], 1)
z_test = z_test.reshape(z_test.shape[0], 1)

# epochs to run for
epochs = 200
folds = 5

# no hidden layers, no activation function
dims = (X.shape[1], 1)
neural = FFNN(dims, seed=1337, hidden_func=)

# parameters to test for
eta = np.logspace(-5, -1, 5)
lam = np.logspace(-5, -1, 5)
momentums = np.linspace(0, 0.1, 5)
lam[0] = 0
rho = 0.9
rho2 = 0.999

# batches to test for
batches_list = np.logspace(
    0, np.log(X_train.shape[0] + 1), 7, base=np.exp(1), dtype=int
)

# schedulers to test for
sched = Adam
adam_params = [rho, rho2]

optimal_params, optimal_lambda, loss_heatmap = neural.optimize_scheduler(
    X_train,
    z_train,
    X_test,
    z_test,
    sched,
    eta,
    lam,
    adam_params,
    batches=7,
    epochs=epochs // 2,
    folds=folds,
)

print(optimal_params)
print(optimal_lambda)


