from utils import *
from sklearn.datasets import load_digits

np.random.seed(42069)


inputs = 64
digits = load_digits()

# print(digits.target)


X = digits.data
t_int = digits.target

X_fives = np.hstack([X, t_int.reshape(t_int.shape[0], 1)])

X_fives = X_fives[X_fives[:, 64] == 2, :][:, :64]

t = np.zeros((t_int.size, 10))
t[np.arange(t_int.size), t_int] = 1

print(t)


X_train, X_test, t_train, t_test = train_test_split(X, t)

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

dims = (inputs, 10, 10)
neural = FFNN(
    dims,
    output_func=softmax,
    cost_func=CostCrossEntropy,
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


print(neural.predict(X_fives))

plt.plot(train_errors, label="Train")
plt.plot(test_errors, label="Train")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("MSE over Epochs")
plt.show()
