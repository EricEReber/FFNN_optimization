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
z = z.reshape(z.shape[0], 1)

# epochs to run for
epochs = 200
folds = 5
scheduler = Adam
args = [0.01, 0.9, 0.999]

funcs = [RELU, sigmoid, CostLogReg]

results = plot_arch(
    FFNN,
    400,
    funcs,
    X,
    z,
    scheduler,
    *args,
    lam=0,
    batches=7,
    epochs=50,
    classify=True,
    step_size=20
)


sns.set(font_scale=2)
plt.title("Accuracy by model complexity for cancer data")
plt.xlabel("Total amount of hidden nodes")
plt.ylabel("Accuracy")
plt.plot(
    results["node_sizes"],
    results["one_hid_train"],
    label="One hidden layer: train",
    lw=4,
)
plt.plot(
    results["node_sizes"], results["one_hid_test"], label="One hidden layer: test", lw=4
)
plt.plot(
    results["node_sizes"],
    results["two_hid_train"],
    "--",
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
plt.legend()
plt.show()
