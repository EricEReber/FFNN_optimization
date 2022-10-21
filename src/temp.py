from utils import *

dims = (4, 2, 4, 2)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def nothing(x):
    return x


neural = FFNN(dims, hidden_func=lambda x: x, iterations=2)

x = np.array([[1, 1, 1, 1], [0, 1, 1, 1], [3, 5, 6, 7]])

neural.fit(x, None)

# x = np.array([2, 3, 4, 5])

# print(f"{neural.predict(x)=}")

# print(neural.hidden_outs)
