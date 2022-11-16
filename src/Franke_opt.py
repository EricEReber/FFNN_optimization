from utils import *
from Schedulers import *
from FFNN import FFNN
from sklearn.neural_network import MLPRegressor

np.random.seed(1337)
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

z_train = z_train.reshape(z_train.shape[0], 1)
z_test = z_test.reshape(z_test.shape[0], 1)

# ------------------------- Params -------------------------
rho = 0.9
rho2 = 0.99

dims = (2, 66,66,66, 1)
train_epochs = 1000

eta = np.logspace(-4, -1, 4)
lams = np.logspace(-4, -1, 4)
lams[0] = 0.0

# batch_sizes = np.linspace(1, X.shape[0] // 2, 5, dtype=int)

schedulers = [Adagrad]
adagrad_params = []
adam_params = [rho, rho2]

params_list = [ 
        adagrad_params,
        ]

optimal_params_list = []
optimal_eta = np.zeros(len(schedulers))
optimal_lambdas = np.zeros(len(schedulers))
# ------------------------- FFNN -------------------------
neural = FFNN(dims, hidden_func=RELU, seed=1337)

# gridsearch params eta and lambda for schedulers
for i in range(len(schedulers)):
    plt.subplot(321 + i)
    plt.suptitle("Test loss for eta, lambda grid", fontsize=22)
    
    optimal_params, optimal_lambda, loss_heatmap = neural.optimize_scheduler(
        X_train[:, 1:3],
        z_train,
        schedulers[i],
        eta,
        lams,
        params_list[i],
        batches=25,
        epochs=200,
        folds=5,
    )

    optimal_eta[i] = optimal_params[0]
    optimal_lambdas[i] = optimal_lambda
    optimal_params_list.append(optimal_params)

    # plot heatmap
    ax = sns.heatmap(loss_heatmap, xticklabels=lams, yticklabels=eta, annot=True)
    ax.add_patch(
        Rectangle(
            (np.where(lams == optimal_lambda)[0], np.where(eta == optimal_params[0])[0]),
            width=1,
            height=1,
            fill=False,
            edgecolor="crimson",
            lw=4,
            clip_on=False,
        )
    )
    plt.xlabel("lambda", fontsize=18)
    plt.ylabel("eta", fontsize=18)
    plt.title(f"{schedulers[i].__name__}", fontsize=22)
plt.show()
