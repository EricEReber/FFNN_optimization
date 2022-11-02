# Our own library of functions
from utils import *
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

np.random.seed(42069)

iris = datasets.load_iris()
cancer = datasets.load_breast_cancer()

X = iris.data[:100, :2]
t = iris.target[:100]

X = cancer.data
t = cancer.target


X_train, X_test, t_train, t_test = train_test_split(X, t)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)

dims = (30, 1)

t_train = t_train.reshape(t_train.shape[0], 1)
t_test = t_test.reshape(t_test.shape[0], 1)

eta = 0.01

neural = FFNN(dims, output_func=sigmoid, cost_func=CostLogReg)
test_errors, train_errors = neural.fit(
    X_train_sc,
    t_train,
    Constant,
    eta,
    batches=1,
    epochs=10000,
    X_test=X_test_sc,
    t_test=t_test,
)
plt.plot(train_errors, label=f"train")
plt.plot(test_errors, label=f"test")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("MSE over Epochs for different schedulers")
plt.show()
