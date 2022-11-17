# Our own library of functions
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# Other libraries and packages
from matplotlib.patches import Rectangle
from utils import *
from Schedulers import *
from FFNN import FFNN
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

np.random.seed(1337)
"""
When run, the following code generates a plot the accuracy achieved by a model
as function of its complexity, i.e the number og hidden layers and nodes 
in each of said layers. 
"""


# ------------------------ Loading data ------------------------
cancer = load_breast_cancer()

X = cancer.data
z = cancer.target
z = z.reshape(z.shape[0], 1)

# ----------------------- Setting params -----------------------
epochs = 200
folds = 5
scheduler = Adam
args = [0.01, 0.9, 0.999]

funcs = [RELU, sigmoid, CostLogReg]

# ------------------------ Plots ------------------------
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
    step_size=60
)

sns.set(font_scale=2)
plt.title(
    "Accuracy by model complexity for cancer data, \n using RELU as the activation function"
)
plt.xlabel("Total amount of hidden nodes")
plt.ylabel("Accuracy")
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
    results["three_hid_train"],
    label="Three hidden layers: train",
    lw=4,
)
plt.plot(
    results["node_sizes"],
    results["three_hid_test"],
    "--",
    label="Three hidden layers: test",
    lw=4,
)

plt.legend(loc=(1.04, 0))
plt.show()
