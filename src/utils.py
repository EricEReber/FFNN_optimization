from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
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
import seaborn as sns
import sys
import argparse


def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


# debug function
def SkrankeFunction(x, y):
    return 3 * x + 8 * y + 4 * x**2 - 4 * x * y - 5 * y**2


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


def scores(z, z_preds):
    N = z_preds.shape[1]
    MSEs = np.zeros((N))
    R2s = np.zeros((N))

    for n in range(N):
        MSEs[n] = MSE(z, z_preds[:, n])
        R2s[n] = R2(z, z_preds[:, n])

    return MSEs, R2s


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
            (target * np.log(X + 10e-10)) + ((1 - target) * np.log(1 - X + 10e-10))
        )

    return func


def CostCrossEntropy(target):
    def func(X):
        return -(1.0 / target.size) * np.sum(target * np.log(X + 10e-10))

    return func


# Activation functions
def sigmoid(x):
    try:
        return 1.0 / (1 + np.exp(-x))
    except FloatingPointError:
        return np.where(x > np.zeros(x.shape), np.ones(x.shape), np.zeros(x.shape))


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / (np.sum(np.exp(x), axis=-1, keepdims=True) + 10e-10)


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
            delta = 10e-4
            return np.where(
                x > np.zeros(x.shape), np.ones(x.shape), np.full((x.shape), delta)
            )

        return func

    else:
        return elementwise_grad(func)


def RELU(x: np.ndarray):
    return np.where(x > np.zeros(x.shape), x, np.zeros(x.shape))


def LRELU(x: np.ndarray):
    delta = 10e-4
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
    # take in batch size for unity but dont use it
    def __init__(self, eta, batch_size):
        super().__init__(eta)

    def update_change(self, gradient):
        return self.eta * gradient


class Momentum(Scheduler):
    # take in batch size for unity but dont use it
    def __init__(self, eta: float, momentum: float, batch_size):
        super().__init__(eta)
        self.momentum = momentum
        self.change = 0

    def update_change(self, gradient):
        self.change = self.momentum * self.change + self.eta * gradient
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

        G_t_inverse = 1 / (
            delta + np.sqrt(np.reshape(np.diagonal(self.G_t), (self.G_t.shape[0], 1)))
        )
        self.change = self.eta * gradient * G_t_inverse
        return self.change

    def reset(self):
        self.G_t = None


class RMS_prop(Scheduler):
    def __init__(self, eta, rho, batch_size):
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
    def __init__(self, eta, rho, rho2, batch_size):
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

    def reset(self):
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

        for i in range(len(self.dimensions) - 1):
            # weight_array = np.ones((dimensions[i] + 1, dimensions[i + 1])) * 2
            weight_array = np.random.randn(self.dimensions[i] + 1, self.dimensions[i + 1])
            weight_array[0, :] = np.random.randn(self.dimensions[i + 1]) * 0.01

            # weight_array[0, :] = np.ones(dimensions[i + 1])
            self.weights.append(weight_array)

    def reset_weights(self):
        self.weights = list()
        for i in range(len(self.dimensions) - 1):
            # weight_array = np.ones((dimensions[i] + 1, dimensions[i + 1])) * 2
            weight_array = np.random.randn(self.dimensions[i] + 1, self.dimensions[i + 1])
            weight_array[0, :] = np.random.randn(self.dimensions[i + 1]) * 0.01

            # weight_array[0, :] = np.ones(dimensions[i + 1])
            self.weights.append(weight_array)

    def optimize_scheduler(
        # todo change to test_score instead of train
        # todo select minimal error at the end of training, not min all time as this leads to strange results
        self,
        X,
        t,
        X_test,
        t_test,
        scheduler,
        eta,
        lam,
        *args,
        batches=1,
        epochs=1000,
    ):

        if scheduler is not Momentum:
            loss_heatmap = np.zeros((eta.shape[0], lam.shape[0]))
            for y in range(eta.shape[0]):
                for x in range(lam.shape[0]):
                    params = [eta[y]] + [*args][0]
                    _, test_error = self.fit(
                        X,
                        t,
                        scheduler,
                        *params,
                        batches=batches,
                        epochs=epochs,
                        lam=lam[x],
                        X_test=X_test,
                        t_test=t_test,
                    )
                    loss_heatmap[y, x] = test_error[-1]
                    self.reset_weights()

            # select optimal eta, lambda
            y, x = (
                loss_heatmap.argmin() // loss_heatmap.shape[1],
                loss_heatmap.argmin() % loss_heatmap.shape[1],
            )
            optimal_eta = eta[y]
            optimal_lambda = lam[x]

            optimal_params = [optimal_eta] + [*args][0]
            batch_sizes = np.linspace(1, X.shape[0] // 2, 5, dtype=int)

            optimal_batch = 0
            batch_size_search = np.zeros(len(batch_sizes))
            for i in range(len(batch_sizes)):
                # neural.read(f"batch_size_search{i}")
                _, test_error = self.fit(
                    X,
                    t,
                    scheduler,
                    *optimal_params,
                    batches=batch_sizes[i],
                    epochs=epochs,
                    lam=optimal_lambda,
                    X_test=X_test,
                    t_test=t_test,
                )
                self.reset_weights()
                # todo would be interesting to see how much time / how fast it happens
                batch_size_search[i] = test_error[-1]
            minimal_error = np.min(batch_size_search)
            optimal_batch = batch_sizes[np.argmin(batch_size_search)]

            plotting_data = [loss_heatmap, batch_size_search]

            # if plot_batch:
            #     plt.plot(batch_size_search)
            #     plt.title("batch size vs error", fontsize=22)
            #     plt.xlabel("batch size", fontsize=18)
            #     plt.ylabel("error", fontsize=18)
            #     plt.show()

            return (
                optimal_params,
                optimal_lambda,
                optimal_batch,
                minimal_error,
                plotting_data,
            )
        else:
            return self.optimize_momentum(
                X, t, X_test, t_test, eta, lam, *args, batches=batches, epochs=epochs
            )

    def optimize_momentum(
        self, X, t, X_test, t_test, eta, lam, momentums, batches=1, epochs=1000
    ):
        # todo change to test_score instead of train
        # todo select minimal error at the end of training, not min all time as this leads to strange results
        loss_heatmap = np.zeros((eta.shape[0], lam.shape[0], len(momentums)))
        for y in range(eta.shape[0]):
            for x in range(lam.shape[0]):
                for z in range(len(momentums)):
                    params = [eta[y], momentums[z]]
                    _, test_error = self.fit(
                        X,
                        t,
                        Momentum,
                        *params,
                        batches=batches,
                        epochs=epochs,
                        lam=lam[x],
                        X_test=X_test,
                        t_test=t_test,
                    )
                    loss_heatmap[y, x, z] = test_error[-1]
                    self.reset_weights()

        y, x, z = np.unravel_index(loss_heatmap.argmin(), loss_heatmap.shape)
        optimal_eta = eta[y]
        optimal_lambda = lam[x]
        optimal_momentum = momentums[z]

        optimal_params = [optimal_eta, optimal_momentum]
        batch_sizes = np.linspace(1, X.shape[0] // 2, 5, dtype=int)

        optimal_batch = 0
        batch_size_search = np.zeros(len(batch_sizes))
        for i in range(len(batch_sizes)):
            # neural.read(f"batch_size_search{i}")
            _, test_error = self.fit(
                X,
                t,
                Momentum,
                *optimal_params,
                batches=batch_sizes[i],
                epochs=epochs,
                lam=optimal_lambda,
                X_test=X_test,
                t_test=t_test,
            )
            self.reset_weights()

            # todo would be interesting to see how much time / how fast it happens
            batch_size_search[i] = test_error[-1]
        minimal_error = np.min(batch_size_search)
        optimal_batch = batch_sizes[np.argmin(batch_size_search)]

        plotting_data = [loss_heatmap[:, :, z], batch_size_search]

        # if plot_batch:
        #     plt.plot(batch_size_search)
        #     plt.title("batch size vs error", fontsize=22)
        #     plt.xlabel("batch size", fontsize=18)
        #     plt.ylabel("error", fontsize=18)
        #     plt.show()

        return (
            optimal_params,
            optimal_lambda,
            optimal_batch,
            minimal_error,
            plotting_data,
        )

    def write(self, path: str):
        """Write weights and biases to file
        Parameters:
            path (str): The path to the file to be written to
        """
        # print(f'Writing weights to file "{path}"')
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
        elif (
            self.cost_func.__name__ == "CostLogReg"
            or self.cost_func.__name__ == "CostCrossEntropy"
        ):
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
        *args,  # arguments for the scheduler (sans batchsize)
        batches: int = 1,
        epochs: int = 1000,
        lam: float = 0,
        X_test: np.ndarray = None,
        t_test: np.ndarray = None,
    ):
        train_errors = np.empty(epochs)
        train_errors.fill(np.nan)  # makes for better plots if we cancel early
        test_errors = np.empty(epochs)
        test_errors.fill(np.nan)
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
            self.schedulers_weight.append(scheduler_class(*args, chunksize))
            self.schedulers_bias.append(scheduler_class(*args, chunksize))

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

                    if (
                        isinstance(self.schedulers_weight[0], RMS_prop)
                        or isinstance(self.schedulers_weight[0], Adam)
                        or isinstance(self.schedulers_weight[0], Adagrad)
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
                if (
                    self.cost_func.__name__ == "CostLogReg"
                    or self.cost_func.__name__ == "CostCrossEntropy"
                ):
                    train_acc = accuracy(self.predict(X, raw=False), t)
                    # print(self.predict(X, raw=False))
                    # print(self.predict(X))
                    # print(train_acc)
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
            # allows for stopping training at any point and seeing the result
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
                if self.output_func.__name__ == "softmax":
                    delta_matrix = self.a_matrices[i + 1] - t
                else:
                    cost_func_derivative = grad(self.cost_func(t))
                    delta_matrix = out_derivative(
                        self.z_matrices[i + 1]
                    ) * cost_func_derivative(self.a_matrices[i + 1])

            else:
                delta_matrix = (
                    self.weights[i + 1][1:, :] @ delta_matrix.T
                ).T * hidden_derivative(self.z_matrices[i + 1])

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


def plot_decision_boundary(X, t, classifier):
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    plt.figure(figsize=(8, 6))

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap="Paired")
    plt.scatter(X[:, 0], X[:, 1], c=t, s=20.0, cmap="Paired")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision regions")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.show()
