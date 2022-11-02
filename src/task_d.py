"""
task b (and task g): plot terrain, approximate terrain with OLS (own implementation and scikit) and calculate MSE, R2 &
                     beta over model complexity for real data. Performs task_b, so no resampling.
"""
# Our own library of functions
from utils import *
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


np.random.seed(42069)

cancer = datasets.load_breast_cancer()

X = cancer.data
z = cancer.target

X_train, X_test, z_train, z_test = train_test_split(X, z)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)

dims = (30, 20, 1)

z_train = z_train.reshape(z_train.shape[0], 1)
z_test = z_test.reshape(z_test.shape[0], 1)
batches = 1
batch_size = X.shape[0] // batches

eta = 0.003

neural = FFNN(
    dims,
    output_func=sigmoid,
    cost_func=CostLogReg,  # checkpoint_file="cancerweights20"
)
# neural.read("cancerweights20")
train_errors, test_errors = neural.fit(
    X_train_sc,
    z_train,
    Constant,
    eta,
    batches=batches,
    # epochs=10000,
    epochs=100000,
    X_test=X_test_sc,
    t_test=z_test,
)
plt.plot(train_errors, label="train")
plt.plot(test_errors, label="test")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("LogLoss")
plt.title("LogLoss over Epochs")
plt.show()

z_pred = neural.predict(X_test_sc)
