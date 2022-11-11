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

plot_arch(
    FFNN, 
    max_nodes,
    funcs,
    X,
    t,
    scheduler,
    *args,
    lam: float = 0,
    batches: int = 1,
    epochs: int = 1000,
    classify: bool = False,
):
