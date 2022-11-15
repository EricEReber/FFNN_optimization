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

one_hid_train, one_hid_test, two_hid_train, two_hid_test = plot_arch(
    FFNN,
    50,
    funcs,
    X,
    z,
    scheduler,
    *args,
    lam=0,
    batches=7,
    epochs=10,
    classify=True,
)

one_hid_train2 = list()

for item in one_hid_train:
    if not np.isnan(item):
        one_hid_train2.append(item)


print(f"{one_hid_train2=}")
print(f"{one_hid_test=}")
plt.plot(one_hid_train2)
plt.plot(one_hid_test)
plt.show()
