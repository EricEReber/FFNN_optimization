# Our own library of functions
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.patches import Rectangle
from utils import *
from Schedulers import *
from FFNN import FFNN
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

np.random.seed(1337)

# read in data
digits = load_digits()

X = digits.data
z = digits.target

print(X[0])
z = onehot(z)
# z = z.reshape(z.shape[0], 1)

# epochs to run for
epochs = 200
folds = 5

dims = (64, 66, 10)


# no hidden layers, no activation function
neural = FFNN(dims, hidden_func=LRELU, output_func=softmax, cost_func=CostCrossEntropy)

# parameters to test for
# eta = np.logspace(-5, -1, 5)
# lam = np.logspace(-5, -1, 5)
# momentums = np.linspace(0, 0.1, 5)
# lam[0] = 0
rho = 0.9
rho2 = 0.999

# batches to test for
batches_list = np.logspace(0, np.log(X.shape[0] + 1), 7, base=np.exp(1), dtype=int)

# schedulers to test for
sched = Adam
adam_params = [rho, rho2]
adam_params = [0.01, rho, rho2]

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

eta = 0.01
adam_params = [eta, rho, rho2]

scores = neural.cross_val(
    5,
    X,
    z,
    Adam,
    *adam_params,
    batches=X.shape[0],
    epochs=200,
    use_best_weights=True,
)

print(f"{scores['final_train_acc']=}")
print(f"{scores['final_test_acc']=}")
