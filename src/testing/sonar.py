from utils import *

np.random.seed(42069)


inputs = 60
data = np.genfromtxt("../../data/sonar.csv", delimiter=",")

data = resample(data)
X = data[:, :inputs]
t = data[:, inputs]


X_train, X_test, t_train, t_test = train_test_split(X, t)

t_train = t_train.reshape(t_train.shape[0], 1)
t_test = t_test.reshape(t_test.shape[0], 1)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)

eta = 0.008
momentum = 0.5
rho = 0.9
rho2 = 0.99

sched = Adam
params = [eta, rho, rho2]

dims = (inputs, 30, 30, 1)
neural = FFNN(
    dims,
    output_func=sigmoid,
    cost_func=CostLogReg,
)
train_errors, test_errors = neural.fit(
    X_train,
    t_train,
    sched,
    *params,
    batches=10,
    epochs=10000,
    X_test=X_test,
    t_test=t_test,
)
plt.plot(train_errors, label="Train")
plt.plot(test_errors, label="Train")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("MSE over Epochs")
plt.show()
