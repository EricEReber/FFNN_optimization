from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import autograd.numpy as np
from autograd import grad, elementwise_grad
from random import random, seed
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from typing import Tuple, Callable
from imageio import imread
import sys
import argparse
import re


def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


# debug function
def SkrankeFunction(x, y):
    return 0 + 1 * x + 2 * y + 3 * x**2 + 4 * x * y + 5 * y**2


def create_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x ** (i - k)) * (y**k)

    return X


def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model) ** 2) / n


def OLS(X_train: np.ndarray, z_train: np.ndarray):
    beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
    return beta


def ridge(X_train, z_train, lam):
    L = X_train.shape[1]
    beta = np.linalg.pinv(X_train.T @ X_train + lam * np.eye(L)) @ X_train.T @ z_train
    return beta


def bootstrap(
    X: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    z_train: np.ndarray,
    z_test: np.ndarray,
    bootstraps: int,
    *,
    centering: bool = False,
    model: Callable = OLS,
    lam: float = 0,
):
    z_preds_train = np.empty((z_train.shape[0], bootstraps))
    z_preds_test = np.empty((z_test.shape[0], bootstraps))

    # non resampled train
    _, z_pred_train, _, _ = evaluate_model(
        X, X_train, X_test, z_train, model, lam=lam, centering=centering
    )

    for i in range(bootstraps):
        X_, z_ = resample(X_train, z_train)
        _, _, z_pred_test, _ = evaluate_model(
            X, X_, X_test, z_, model, lam=lam, centering=centering
        )
        # z_preds_train[:, i] = z_pred_train
        z_preds_test[:, i] = z_pred_test

    return z_preds_test, z_pred_train


def crossval(
    X: np.ndarray,
    z: np.ndarray,
    K: int,
    *,
    centering: bool = False,
    model=OLS,
    lam: float = 0,
):
    chunksize = X.shape[0] // K

    errors = np.zeros(K)
    X, z = resample(X, z)

    for k in range(K):
        if k == K - 1:
            # if we are on the last, take all thats left
            X_test = X[k * chunksize :, :]
            z_test = z[k * chunksize :]
        else:
            X_test = X[k * chunksize : (k + 1) * chunksize, :]
            z_test = z[k * chunksize : (k + 1) * chunksize :]

        X_train = np.delete(
            X,
            [i for i in range(k * chunksize, k * chunksize + X_test.shape[0])],
            axis=0,
        )
        z_train = np.delete(
            z,
            [i for i in range(k * chunksize, k * chunksize + z_test.shape[0])],
            axis=0,
        )

        _, _, z_pred_test, _ = evaluate_model(
            X,
            X_train,
            X_test,
            z_train,
            model,
            lam=lam,
            centering=centering,
        )
        errors[k] = MSE(z_test, z_pred_test)

    return np.mean(errors)


def bias_variance(z_test: np.ndarray, z_preds_test: np.ndarray):
    MSEs, _ = scores(z_test, z_preds_test)
    error = np.mean(MSEs)
    bias = np.mean(
        (z_test - np.mean(z_preds_test, axis=1, keepdims=True).flatten()) ** 2
    )
    variance = np.mean(np.var(z_preds_test, axis=1, keepdims=True))

    return error, bias, variance


def preprocess(x: np.ndarray, y: np.ndarray, z: np.ndarray, N, test_size):
    X = create_X(x, y, N)

    zflat = np.ravel(z)
    X_train, X_test, z_train, z_test = train_test_split(X, zflat, test_size=test_size)

    return X, X_train, X_test, z_train, z_test


def evaluate_model(
    X,
    X_train,
    X_test,
    z_train,
    model,
    *,
    lam: float = 0,
    centering: bool = False,
):
    if isinstance(model, Callable):
        intercept = 0
        if centering:
            X_train = X_train[:, 1:]
            X_test = X_test[:, 1:]
            X = X[:, 1:]
            z_train_mean = np.mean(z_train, axis=0)
            X_train_mean = np.mean(X_train, axis=0)

            if model.__name__ == "OLS":
                beta = model((X_train - X_train_mean), (z_train - z_train_mean))

            elif model.__name__ == "ridge":
                beta = model((X_train - X_train_mean), (z_train - z_train_mean), lam)

            intercept = z_train_mean - X_train_mean @ beta

        else:
            if model.__name__ == "OLS":
                beta = model(X_train, z_train)

            elif model.__name__ == "ridge":
                beta = model(
                    X_train,
                    z_train,
                    lam,
                )
        # intercept is zero if no centering
        z_pred_train = X_train @ beta + intercept
        z_pred_test = X_test @ beta + intercept
        z_pred = X @ beta + intercept

    # presumed scikit model
    else:
        intercept = 0
        if centering:
            # if width is 1, simply return the intercept
            if X_train.shape[1] == 1:
                beta = np.zeros(1)
                intercept = np.mean(z_train, axis=0)
                z_pred_train = np.ones(X_train.shape[0]) * intercept
                z_pred_test = np.ones(X_test.shape[0]) * intercept
                z_pred = np.ones(X.shape[0]) * intercept

                return beta, z_pred_train, z_pred_test, z_pred

            X_train = X_train[:, 1:]
            X_test = X_test[:, 1:]
            X = X[:, 1:]
            z_train_mean = np.mean(z_train, axis=0)
            X_train_mean = np.mean(X_train, axis=0)

            model.fit((X_train - X_train_mean), (z_train - z_train_mean))
            beta = model.coef_
            intercept = np.mean(z_train_mean - X_train_mean @ beta)
        else:
            model.fit(X_train, z_train)

        beta = model.coef_
        z_pred = model.predict(X) + intercept
        z_pred_train = model.predict(X_train) + intercept
        z_pred_test = model.predict(X_test) + intercept

    return beta, z_pred_train, z_pred_test, z_pred


def minmax_dataset(X, X_train, X_test, z, z_train, z_test):
    x_scaler = MinMaxScaler()
    z_scaler = MinMaxScaler()

    x_scaler.fit(X_train)
    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)
    X = x_scaler.transform(X)

    z_shape = z.shape

    # make all zeds into 1 dimensional arrays for standardscaler
    z_train = z_train.reshape((z_train.shape[0], 1))
    z_test = z_test.reshape((z_test.shape[0], 1))
    z = z.ravel().reshape((z.ravel().shape[0], 1))

    z_scaler.fit(z_train)
    z_train = np.ravel(z_scaler.transform(z_train))
    z_test = np.ravel(z_scaler.transform(z_test))
    z = np.ravel(z_scaler.transform(z))
    z = z.reshape(z_shape)

    return X, X_train, X_test, z, z_train, z_test


def linreg_to_N(
    X: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    z_train: np.ndarray,
    z_test: np.ndarray,
    N: int,
    *,
    centering: bool = False,
    model: Callable = OLS,
    lam: float = 0,
):
    L = X_train.shape[1]

    betas = np.zeros((L, N + 1))
    z_preds_train = np.empty((z_train.shape[0], N + 1))
    z_preds_test = np.empty((z_test.shape[0], N + 1))
    z_preds = np.empty((X.shape[0], N + 1))

    for n in range(N + 1):
        print(n)
        l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
        beta, z_pred_train, z_pred_test, z_pred = evaluate_model(
            X[:, :l],
            X_train[:, :l],
            X_test[:, :l],
            z_train,
            model,
            lam=lam,
            centering=centering,
        )

        betas[0 : len(beta), n] = beta
        z_preds_test[:, n] = z_pred_test
        z_preds_train[:, n] = z_pred_train
        z_preds[:, n] = z_pred

    return betas, z_preds_train, z_preds_test, z_preds


def scores(z, z_preds):
    N = z_preds.shape[1]
    MSEs = np.zeros((N))
    R2s = np.zeros((N))

    for n in range(N):
        MSEs[n] = MSE(z, z_preds[:, n])
        R2s[n] = R2(z, z_preds[:, n])

    return MSEs, R2s


def find_best_lambda(X, z, model, lambdas, N, K):
    kfolds = KFold(n_splits=K, shuffle=True)
    model = GridSearchCV(
        estimator=model,
        param_grid={"alpha": list(lambdas)},
        scoring="neg_mean_squared_error",
        cv=kfolds,
    )
    best_polynomial = 0
    best_lambda = 0
    best_MSE = 10**10

    for n in range(N + 1):
        print(n)
        l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
        model.fit(X[:, :l], z)

        if -model.best_score_ < best_MSE:
            best_MSE = -model.best_score_
            best_lambda = model.best_params_["alpha"]
            best_polynomial = n

    return best_lambda, best_MSE, best_polynomial


def CostOLS(target):
    """Return a function valued only at X, so
    that it may be easily differentiated
    """

    def func(X):
        return (1.0 / target.shape[0]) * np.sum((target - X) ** 2)

    return func


def CostLogReg(target):
    """Return a function valued only at X, so
    that it may be easily differentiated
    """

    def func(X):
        return -(1.0 / target.shape[0]) * np.sum(
            (target * np.log(X)) + ((1 - X) * np.log(1 - X))
        )

    return func


# Activation functions
def sigmoid(x):
    try:
        return 1.0 / (1 + np.exp(-x))
    except FloatingPointError:
        return np.where(x > np.zeros(x.shape), np.ones(x.shape), np.zeros(x.shape))


def derivate(func):
    if func.__name__ == "sigmoid":

        def func(x):
            return sigmoid(x) * (1 - sigmoid(x))

        return func

    elif func.__name__ == "RELU":

        def func(x):
            return np.where(x > np.zeros(x.shape), np.ones(x.shape), np.zeros(x.shape))

        return func

    elif func.__name__ == "LRELU":

        def func(x):
            return np.where(
                x > np.zeros(x.shape), np.ones(x.shape), np.full((x.shape), delta)
            )

        return func

    else:
        return elementwise_grad(func)


def RELU(x: np.ndarray):
    return np.where(x > np.zeros(x.shape), x, np.zeros(x.shape))


def LRELU(x: np.ndarray, delta: float):
    return np.where(x > np.zeros(x.shape), x, delta * x)


def accuracy(prediction: np.ndarray, target: np.ndarray):
    return np.average((target == prediction))


# ------------------- Gradient Descent Optimizing Methods -------------------#

# abstract class for schedulers
class Scheduler:
    def __init__(self, eta):
        self.eta = eta

    # should be overwritten
    def update_change(self, gradient):
        raise NotImplementedError


class Constant(Scheduler):
    def __init__(self, eta):
        super().__init__(eta)

    def update_change(self, gradient):
        return self.eta * gradient


class Momentum(Scheduler):
    def __init__(self, eta: float, momentum: float):
        super().__init__(eta)
        self.momentum = momentum
        self.change = 0

    def update_change(self, gradient):
        self.change = self.eta * gradient + self.momentum * self.change
        return self.change


class Adagrad(Scheduler):
    def __init__(self, eta, batch_size):
        super().__init__(eta)
        self.G_t = None
        self.batch_size = batch_size
        self.change = 0

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero

        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        gradient = gradient / self.batch_size

        self.G_t += gradient @ gradient.T

        G_t_inverse = 1 / (delta + np.sqrt(np.reshape(np.diagonal(self.G_t), (self.G_t.shape[0], 1))))
        self.change = self.eta * gradient * G_t_inverse
        return self.change

    def reset(self):
        self.G_t = None


class RMS_prop2(Scheduler):
    def __init__(self, eta, batch_size, rho):
        super().__init__(eta)
        self.batch_size = batch_size
        self.rho = rho
        self.giter = None
        self.change = 0

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero

        if not isinstance(self.giter, np.ndarray):
            self.giter = np.zeros((gradient.shape[0], gradient.shape[0]))

        gradient = (1 / self.batch_size) * gradient
        self.prev_giter = self.giter
        self.giter += gradient @ gradient.T

        gnew = self.rho * self.prev_giter + (1 - self.rho) * self.giter
        ginverse = np.c_[self.eta / (delta + np.sqrt(np.diagonal(gnew)))]
        self.change = np.multiply(ginverse, gradient)
        return self.change

    def reset(self):
        self.giter = None


class RMS_prop(Scheduler):
    def __init__(self, eta, batch_size, rho):
        super().__init__(eta)
        self.batch_size = batch_size
        self.rho = rho
        self.second = 0.0

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero
        gradient = gradient / self.batch_size
        self.second = self.rho * self.second + (1 - self.rho) * gradient * gradient
        self.change = self.eta * gradient / (np.sqrt(self.second + delta))
        return self.change

    def reset(self):
        self.second = 0.0


class Adam(Scheduler):
    def __init__(self, eta, batch_size, rho, rho2):
        super().__init__(eta)
        self.rho = rho
        self.rho2 = rho2

        self.batch_size = batch_size

        self.rho_t = rho
        self.rho2_t = rho2

        self.moment = 0
        self.second = 0

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero

        gradient = gradient / self.batch_size

        self.moment = self.rho * self.moment + (1 - self.rho) * gradient
        self.second = self.rho2 * self.second + (1 - self.rho2) * gradient * gradient

        self.rho_t *= self.rho_t
        self.rho2_t *= self.rho2_t

        self.moment = self.moment / (1 - self.rho_t)
        self.second = self.second / (1 - self.rho2_t)

        self.change = self.eta * self.moment / (np.sqrt(self.second + delta))

        return self.change

    def reset():
        self.rho_t = self.rho
        self.rho2_t = self.rho2
        self.moment = 0
        self.second = 0


class FFNN:
    """
    Feed Forward Neural Network

    Attributes:
        dimensions (list[int]): A list of positive integers, which defines our layers. The first number
        is the input layer, and how many nodes it has. The last number is our output layer. The numbers
        in between define how many hidden layers we have, and how many nodes they have.

        hidden_func (Callable): The activation function for the hidden layers

        output_func (Callable): The activation function for the output layer

        cost_func (Callable): Our cost function

        checkpoint_file (string): A file path where our weights will be saved

        weights (list): A list of numpy arrays, containing our weights
    """

    def __init__(
        self,
        dimensions: tuple[int],
        hidden_func: Callable = sigmoid,
        output_func: Callable = lambda x: x,
        cost_func: Callable = CostOLS,
        checkpoint_file: str = None,
    ):
        self.weights = list()
        self.schedulers_weight = list()
        self.schedulers_bias = list()
        self.a_matrices = list()
        self.dimensions = dimensions
        self.hidden_func = hidden_func
        self.output_func = output_func
        self.cost_func = cost_func
        self.z_matrices = list()
        self.checkpoint_file = checkpoint_file

        for i in range(len(dimensions) - 1):
            # weight_array = np.ones((dimensions[i] + 1, dimensions[i + 1])) * 2
            weight_array = np.random.randn(dimensions[i] + 1, dimensions[i + 1])
            weight_array[0, :] = np.random.randn(dimensions[i + 1]) * 0.01

            # weight_array[0, :] = np.ones(dimensions[i + 1])
            self.weights.append(weight_array)

    def write(self, path: str):
        """Write weights and biases to file
        Parameters:
            path (str): The path to the file to be written to
        """
        print(f'Writing weights to file "{path}"')
        np.set_printoptions(threshold=np.inf)
        with open(path, "w") as file:
            text = str(self.dimensions) + "\n"
            for weight in self.weights:
                text += str(weight.shape) + "\n"
                array_str = np.array2string(weight, max_line_width=1e8, separator=",")
                text += array_str + "\n"
            file.write(text)
        # default value
        np.set_printoptions(threshold=1000)

    def read(self, path: str):
        """Read weights and biases from file. This overwrites the weights and biases for the calling instance,
        and may well change the dimensions completely if the saved dimensions are not the same as the current
        ones.
        Parameters:
            path (str): The path to the file to be read from
        """
        print(f'Reading weights to file "{path}"')
        self.weights = list()
        with open(path, "r") as file:
            self.dimensions = eval(file.readline())
            while True:
                shape_string = file.readline()
                if not shape_string:
                    # we have reached EOF
                    break
                shape = eval(shape_string)
                string = ""
                for i in range(shape[0]):
                    string += file.readline()
                python_array = eval(string)
                numpy_array = np.array(python_array, dtype="float64")
                self.weights.append(numpy_array)

    def feedforward(self, X: np.ndarray):
        """
        Return a prediction vector for each row in X

        Parameters:
            X (np.ndarray): The design matrix, with n rows of p features each

            Returns:
            z (np.ndarray): A prediction vector (row) for each row in our design matrix
        """

        # reset matrices
        self.a_matrices = list()
        self.z_matrices = list()

        # if X is just a vector, make it into a design matrix
        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))

        # put a coloumn of ones as the first coloumn of the design matrix, so that
        # we have a bias term
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # a^0, the nodes in the input layer (one a^0 for each row in X)
        a = X
        self.a_matrices.append(a)
        self.z_matrices.append(a)

        # the feed forward part
        for i in range(len(self.weights)):
            if i < len(self.weights) - 1:
                z = a @ self.weights[i]
                self.z_matrices.append(z)
                a = self.hidden_func(z)
                a = np.hstack([np.ones((a.shape[0], 1)), a])
                self.a_matrices.append(a)
            else:
                # a^L, the nodes in our output layer
                z = a @ self.weights[i]
                a = self.output_func(z)
                self.a_matrices.append(a)
                self.z_matrices.append(z)

        # this will be a^L
        return a

    def predict(self, X: np.ndarray, *, raw=False, threshold=0.5):
        """
        Return a prediction vector for each row in X

        Parameters:
            X (np.ndarray): The design matrix, with n rows of p features each

        Returns:
            z (np.ndarray): A prediction vector (row) for each row in our design matrix
        """

        # if self.output_func.__name__ == "sigmoid":
        #   return np.where(self.feedforward(X) > 0.5, 1, 0)
        # else:
        predict = self.feedforward(X)
        if raw:
            return predict
        elif self.cost_func.__name__ == "CostLogReg":
            return np.where(
                predict > np.ones(predict.shape) * threshold,
                np.ones(predict.shape),
                np.zeros(predict.shape),
            )
        else:
            return predict

    def fit(
        self,
        X: np.ndarray,
        t: np.ndarray,
        scheduler_class,
        *args,  # arguments for the scheduler
        batches: int = 1,
        epochs: int = 1000,
        lam: float = 0,
        X_test: np.ndarray = None,
        t_test: np.ndarray = None,
    ):
        train_errors = np.zeros(epochs)
        test_errors = np.zeros(epochs)
        chunksize = X.shape[0] // batches
        X, t = resample(X, t)

        checkpoint_length = epochs // 10
        checkpoint_num = 0

        self.schedulers_weight = list()
        self.schedulers_bias = list()

        # this function returns a function valued only at X
        cost_function_train = self.cost_func(t)
        if X_test is not None and t_test is not None:
            cost_function_test = self.cost_func(t_test)

        for i in range(len(self.weights)):
            self.schedulers_weight.append(scheduler_class(*args))
            self.schedulers_bias.append(scheduler_class(*args))

        print(scheduler_class.__name__)
        try:
            for e in range(epochs):
                for i in range(batches):
                    # print(f"Batch: {i}")
                    if i == batches - 1:
                        # if we are on the last, take all thats left
                        X_batch = X[i * chunksize :, :]
                        t_batch = t[i * chunksize :, :]
                    else:
                        X_batch = X[i * chunksize : (i + 1) * chunksize, :]
                        t_batch = t[i * chunksize : (i + 1) * chunksize :, :]

                    self.feedforward(X_batch)
                    self.backpropagate(X_batch, t_batch, lam)

                    if isinstance(self.schedulers_weight[0], RMS_prop) or isinstance(
                        self.schedulers_weight[0], Adagrad
                    ):
                        for scheduler in self.schedulers_weight:
                            scheduler.reset()

                        for scheduler in self.schedulers_bias:
                            scheduler.reset()
                train_error = cost_function_train(self.predict(X, raw=True))
                if X_test is not None and t_test is not None:
                    test_error = cost_function_test(self.predict(X_test, raw=True))
                else:
                    test_error = 0

                train_acc = None
                test_acc = None
                if self.cost_func.__name__ == "CostLogReg":
                    train_acc = accuracy(self.predict(X, raw=False), t)
                    if X_test is not None and t_test is not None:
                        test_acc = accuracy(self.predict(X_test, raw=False), t_test)

                train_errors[e] = train_error
                test_errors[e] = test_error
                progression = e / epochs

                length = self._progress_bar(
                    progression,
                    train_error=train_error,
                    test_error=test_error,
                    train_acc=train_acc,
                    test_acc=test_acc,
                )

                if (e % checkpoint_length == 0 and self.checkpoint_file and e) or (
                    e == epochs - 1 and self.checkpoint_file
                ):
                    checkpoint_num += 1
                    print()
                    print(" " * length, end="\r")
                    print(f"{checkpoint_num}/10: Checkpoint reached")
                    self.write(self.checkpoint_file)

        except KeyboardInterrupt:
            pass

        print(" " * length, end="\r")
        self._progress_bar(
            1,
            train_error=train_error,
            test_error=test_error,
            train_acc=train_acc,
            test_acc=test_acc,
        )
        print()

        return train_errors, test_errors

    def update_w_and_b(self, update_list):
        """Updates weights and biases using a list of arrays that matches
        self.weights
        """
        for i in range(len(self.weights)):
            self.weights[i] -= update_list[i]

    # def scale_X(

    def backpropagate(self, X, t, lam):
        out_derivative = derivate(self.output_func)
        hidden_derivative = derivate(self.hidden_func)
        update_list = list()

        for i in range(len(self.weights) - 1, -1, -1):

            # creating the delta terms
            if i == len(self.weights) - 1:
                cost_func_derivative = grad(self.cost_func(t))
                delta_matrix = out_derivative(
                    self.z_matrices[i + 1]
                ) * cost_func_derivative(self.a_matrices[i + 1])

            else:
                delta_matrix = (
                    self.weights[i + 1][1:, :] @ delta_matrix.T
                ).T * hidden_derivative(self.a_matrices[i + 1][:, 1:])

            gradient_weights_matrix = np.zeros(
                (
                    self.a_matrices[i][:, 1:].shape[0],
                    self.a_matrices[i][:, 1:].shape[1],
                    delta_matrix.shape[1],
                )
            )

            for j in range(len(delta_matrix)):
                gradient_weights_matrix[j, :, :] = np.outer(
                    self.a_matrices[i][j, 1:], delta_matrix[j, :]
                )

            gradient_weights = np.sum(gradient_weights_matrix, axis=0)
            delta_accumulated = np.sum(delta_matrix, axis=0)

            gradient_weights = self.a_matrices[i][:, 1:].T @ delta_matrix
            gradient_weights += self.weights[i][1:, :] * lam

            update_matrix = np.vstack(
                [
                    self.schedulers_bias[i].update_change(
                        delta_accumulated.reshape(1, delta_accumulated.shape[0])
                    ),
                    self.schedulers_weight[i].update_change(gradient_weights),
                ]
            )
            # print(f"{update_matrix=}")
            update_list.insert(0, update_matrix)

        self.update_w_and_b(update_list)

    def _progress_bar(self, progression, **kwargs):
        length = 40
        num_equals = int(progression * length)
        num_not = length - num_equals
        arrow = ">" if num_equals > 0 else ""
        bar = "[" + "=" * (num_equals - 1) + arrow + "-" * num_not + "]"
        perc_print = fmt(progression * 100, N=5)
        line = f"  {bar} {perc_print}% "

        for key in kwargs:
            if kwargs[key]:
                value = fmt(kwargs[key], N=4)
                line += f"| {key}: {value} "
        print(line, end="\r")
        return len(line)


def fmt(value, N=4):
    import math

    if value > 0:
        v = value
    elif value < 0:
        v = -10 * value
    else:
        v = 1
    n = 1 + math.floor(math.log10(v))
    if n >= N - 1:
        return str(round(value))
        # or overflow
        # return '!'*N
    return f"{value:.{N-n-1}f}"


# todo: update this function
def gradient_descent_linreg(
    cost_func,
    X,
    X_train,
    X_test,
    beta,
    target,
    *args,
    scheduler_class=Scheduler,
):
    scheduler = scheduler_class(args)

    ols_grad = grad(cost_func, 1)

    eta = scheduler.update_eta()
    beta -= eta * ols_grad(X_train, beta, target)

    z_pred = X @ beta
    z_pred_train = X_train @ beta
    z_pred_test = X_test @ beta
    return beta, z_pred_train, z_pred_test, z_pred


# ---------------------------------------------------------------------------------- OTHER METHODS

# ---------------------------------------------------------------------------------- OTHER METHODS
def read_from_cmdline():
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Read in arguments for tasks")

    group = parser.add_mutually_exclusive_group()

    # with debug or file, we cannot have noise. We cannot have debug and file
    # either
    group.add_argument("-f", "--file", help="Terrain data file name")
    group.add_argument(
        "-d",
        "--debug",
        help="Use debug function for testing. Default false",
        action="store_true",
    )
    group.add_argument(
        "-no",
        "--noise",
        help="Amount of noise to have. Recommended range [0-0.1]. Default 0.05",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "-st",
        "--step",
        help="Step size for linspace function. Range [0.01-0.4]. Default 0.05",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "-b", "--betas", help="Betas to plot, when applicable. Default 10", type=int
    )
    parser.add_argument("-n", help="Polynomial degree. Default 9", type=int, default=9)
    parser.add_argument(
        "-nsc",
        "--noscale",
        help="Do not use scaling (centering for synthetic case or MinMaxScaling for organic case)",
        action="store_true",
    )

    # parse arguments and call run_filter
    args = parser.parse_args()

    # error checking
    if args.noise < 0 or args.noise > 1:
        raise ValueError(f"Noise value out of range [0,1]: {args.noise}")

    if args.step < 0.01 or args.step > 0.4:
        raise ValueError(f"Step value out of range [0,1]: {args.noise}")

    if args.n <= 0:
        raise ValueError(f"Polynomial degree must be positive: {args.N}")

    num_betas = int((args.n + 1) * (args.n + 2) / 2)  # Number of elements in beta
    if args.betas:
        if args.betas > num_betas:
            raise ValueError(
                f"More betas than exist in the design matrix: {args.betas}"
            )
        betas_to_plot = args.betas
    else:
        betas_to_plot = min(10, num_betas)

    if args.file:
        # Load the terrain
        z = np.asarray(imread(args.file), dtype="float64")
        x = np.arange(z.shape[0])
        y = np.arange(z.shape[1])
        x, y = np.meshgrid(x, y, indexing="ij")

        # split data into test and train
        X, X_train, X_test, z_train, z_test = preprocess(x, y, z, args.n, 0.2)

        # normalize data
        centering = False
        if not args.noscale:
            X, X_train, X_test, z, z_train, z_test = minmax_dataset(
                X, X_train, X_test, z, z_train, z_test
            )
    else:
        # create synthetic data
        x = np.arange(0, 1, args.step)
        y = np.arange(0, 1, args.step)
        x, y = np.meshgrid(x, y)
        if args.debug:
            z = SkrankeFunction(x, y)
        else:
            z = FrankeFunction(x, y)
            # add noise
            z += args.noise * np.random.standard_normal(z.shape)
        centering = not args.noscale

        X, X_train, X_test, z_train, z_test = preprocess(x, y, z, args.n, 0.2)

    return (
        betas_to_plot,
        args.n,
        X,
        X_train,
        X_test,
        z,
        z_train,
        z_test,
        centering,
        x,
        y,
        z,
    )
