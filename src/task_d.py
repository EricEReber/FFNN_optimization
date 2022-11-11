"""
task b (and task g): plot terrain, approximate terrain with OLS (own implementation and scikit) and calculate MSE, R2 &
                     beta over model complexity for real data. Performs task_b, so no resampling.
"""
# Our own library of functions
from utils import *
from Schedulers import *
from FFNN import FFNN
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import metrics


np.random.seed(1337)

cancer = datasets.load_breast_cancer()

X = cancer.data
z = cancer.target


X_train, X_test, z_train, z_test = train_test_split(X, z)

scaler = StandardScaler()
scaler.fit(X_train)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)


# parameters
neural_dims = (30, 60, 1)
logreg_dims = (30, 1)
eta = 0.001
#lam = np.logspace(-5, -1, 5)
rho = 0.90
rho2 = 0.999
z_train = z_train.reshape(z_train.shape[0], 1)
z_test = z_test.reshape(z_test.shape[0], 1)
z = z.reshape(z.shape[0], 1)
batches = 10

#neural = FFNN(neural_dims, hidden_func=RELU, output_func=sigmoid, cost_func=CostLogReg)
# neural = FFNN(logreg_dims, output_func=sigmoid, cost_func=CostLogReg)

momentum = 0.5

sched = RMS_prop
sched = Adam
# params = [eta, momentum]
# params = [eta]
opt_params = [rho, rho2]
params = [eta, rho, rho2]
# params = [eta]
# params = [eta, rho]
<<<<<<< HEAD
"""
batch_sizes = np.logspace(0, np.log(X_train.shape[0]+1), 7, base=np.exp(1), dtype=int)

optimal_params, optimal_lambda, _ = neural.optimize_scheduler(
    X_train_sc,
    z_train,
    X_test_sc,
    z_test,
    sched,
    eta,
    lam,
    opt_params,
    batches=batch_sizes[2],
    epochs=100,
    classify=True,
)

optimal_batch, _ = neural.optimize_batch(
    X_train_sc,
    z_train,
    X_test_sc,
    z_test,
    sched,
    optimal_lambda,
    *optimal_params,
    batches_list=batch_sizes,
    epochs=100,
    classify=True,
)

scores = neural.fit(
    X_train_sc,
    z_traint ,
    sched,
    *optimal_params,
    # batches=optimal_batch,
    batches=batch_sizes[2],
    epochs=1000,
    lam=optimal_lambda,
    X_test=X_test_sc,
    t_test=z_test,
) """

o1, o2, t1, t2 = optimize_arch(FFNN, 
              15, 
              [RELU, sigmoid, CostLogReg],
              X_train_sc,
              z_train,
              X_test_sc,
              z_test,
              sched,
              params,
              lams=0.05,
              batch=20,
              epochs=100,
              classify=True)



plt.plot(o1, label="o1")
plt.plot(o2, label="o2")
plt.show()

plt.plot(t1, label="t1")
plt.plot(t2, label="t2")
plt.legend()
plt.show()


prediction = neural.predict(X_test_sc)
plot_confusion(prediction, z_test)
