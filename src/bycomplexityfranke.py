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

# epochs to run for
epochs = 10
folds = 5
scheduler = Adam
args = [0.001, 0.9, 0.999]

funcs = [sigmoid, lambda x: x, CostOLS]

results = plot_arch(
    FFNN,
    400,
    funcs,
    X[:, 1:3],
    z,
    scheduler,
    *args,
    lam=0,
    batches=7,
    epochs=800,
    step_size=40,
    folds=3,
)


sns.set(font_scale=2)
plt.title("MSE by model complexity for the Franke function")
plt.xlabel("Total amount of hidden nodes")
plt.ylabel("MSE")
plt.plot(
    results["node_sizes"],
    results["one_hid_train"],
    label="One hidden layer: train",
    lw=4,
)
plt.plot(
    results["node_sizes"],
    results["one_hid_test"],
    "--",
    label="One hidden layer: test",
    lw=4,
)
plt.plot(
    results["node_sizes"],
    results["two_hid_train"],
    label="Two hidden layers: train",
    lw=4,
)
plt.plot(
    results["node_sizes"],
    results["two_hid_test"],
    "--",
    label="Two hidden layers: test",
    lw=4,
)
plt.plot(
    results["node_sizes"],
    results["three_hid_test"],
    "--",
    label="Three hidden layers: test",
    lw=4,
)
plt.plot(
    results["node_sizes"],
    results["three_hid_train"],
    label="Three hidden layers: train",
    lw=4,
)
# plt.legend(loc=(1.04, 0))
plt.legend()
plt.show()
