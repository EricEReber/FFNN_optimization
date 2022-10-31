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

# tests schedulers for a given model
def test_scheduler():
    # define model (should probably be sent in)
    (
        betas_to_plot,
        N,
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
    ) = read_from_cmdline()

    iris = datasets.load_iris()

    X = iris.data[:100, :2]
    z = iris.target[:100]

    print(z)

    x_train, x_test, z_train, z_test = train_test_split(X, z)

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train_sc = scaler.transform(x_train)
    x_test_sc = scaler.transform(x_test)

    dims = (2, 1)

    # dims = (4, 3, 2, 2)

    dummy_x = np.array(
        [
            [0.134, 0.91827, 0.1982, 0.34654],
            [0.7246, 0.8887, 0.1513, 0.97716],
            [0.441, 0.123, 0.321, 0.71],
        ]
    )
    dummy_t = np.array([[1], [1], [0]])

    z_train = z_train.reshape(z_train.shape[0], 1)

    eta = 0.01
    # batch_size = X_train.shape[0] // 10
    # momentum = 0.5
    # rho = 0.1
    # rho2 = 0.4

    constant_params = [eta]
    # momentum_params = [eta, momentum]
    # adagrad_params = [eta, batch_size]
    # rms_params = [eta, batch_size, rho]
    # adam_params = [eta, batch_size, rho, rho2]

    # params = [momentum_params, adagrad_params, rms_params, adam_params, constant_params]
    params = [constant_params]
    # params = [rms_params]
    schedulers = [
        # Momentum,
        # Adagrad,
        # RMS_prop,
        # Adam,
        Constant,
    ]
    # presume we can get error_over_epochs
    for i in range(len(schedulers)):
        neural = FFNN(dims, output_func=sigmoid, cost_func=CostLogReg)
        error_over_epochs = neural.test_fit(
            x_train_sc,  # X_train[:, 1:3],
            z_train,  # z_train,
            schedulers[i],
            *params[i],
            batches=1,
            epochs=10000,
        )
        plt.plot(error_over_epochs, label=f"{schedulers[i]}")
        plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("MSE over Epochs for different schedulers")
    plt.show()

    z_pred = neural.predict(x_test_sc)
    print(z_pred.T)
    print(z_test)

    # pred_map = z_pred.reshape(z.shape)

    return x, y, z


test_scheduler()
