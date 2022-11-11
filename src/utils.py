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
    assert prediction.size == target.size
    return np.average((target == prediction))


def crossval(
    X: np.ndarray,
    z: np.ndarray,
    K: int,
):
    batch_size = X.shape[0] // K
    np.random.seed(1337)
    X, z = resample(X, z)
    tuples = list()

    for k in range(K):
        if k == K - 1:
            # if we are on the last, take all thats left
            X_left_out = X[k * batch_size :, :]
            z_left_out = z[k * batch_size :, :]
        else:
            X_left_out = X[k * batch_size : (k + 1) * batch_size, :]
            z_left_out = z[k * batch_size : (k + 1) * batch_size, :]

        X_train = np.delete(
            X,
            [i for i in range(k * batch_size, k * batch_size + X_left_out.shape[0])],
            axis=0,
        )
        z_train = np.delete(
            z,
            [i for i in range(k * batch_size, k * batch_size + z_left_out.shape[0])],
            axis=0,
        )

        tuples.append((X_train, z_train, X_left_out, z_left_out))

    return tuples


def bias_variance(z_test: np.ndarray, z_preds_test: np.ndarray):
    MSEs, _ = scores(z_test, z_preds_test)
    error = np.mean(MSEs)
    bias = np.mean(
        (z_test - np.mean(z_preds_test, axis=1, keepdims=True).flatten()) ** 2
    )
    variance = np.mean(np.var(z_preds_test, axis=1, keepdims=True))

    return error, bias, variance


def bootstrap(
    X_train: np.ndarray,
    z_train: np.ndarray,
    bootstraps: int,
):
    tuples = list()

    for i in range(bootstraps):
        X_, z_ = resample(X_train, z_train)
        tuples.append((X_, z_))

    return tuples


def confusion(prediction: np.ndarray, target: np.ndarray):
    # expects that both are vector of zero and one
    # print(np.hstack([target,prediction]))
    target = np.where(target, True, False)
    pred = np.where(prediction, True, False)
    not_target = np.bitwise_not(target)
    not_pred = np.bitwise_not(pred)

    true_pos = np.sum(pred * target)
    true_neg = np.sum(not_pred * not_target)

    false_pos = np.sum(pred * not_target)
    false_neg = np.sum(not_pred * target)

    false_neg_perc = false_neg / (false_neg + true_neg)
    true_neg_perc = true_neg / (false_neg + true_neg)

    true_pos_perc = true_pos / (false_pos + true_pos)
    false_pos_perc = false_pos / (false_pos + true_pos)

    return np.array([[true_neg_perc, false_neg_perc], [false_pos_perc, true_pos_perc]])


def plot_confusion(confusion_matrix: np.ndarray):
    fontsize = 40

    sns.set(font_scale=4)
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=".2%",
        cmap="Blues",
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.show()


# formatting for prints stolen from stack overflow.
# makes decimal values have the same number of digits
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
    np.random.seed(1337)
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
