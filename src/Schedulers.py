from utils import *


class Scheduler:
    """
    Abstract class for Schedulers
    """

    def __init__(self, eta):
        self.eta = eta

    # should be overwritten
    def update_change(self, gradient):
        raise NotImplementedError

    # overwritten if needed
    def reset(self):
        pass


class Constant(Scheduler):
    # take in batch size for unity but does not use it
    def __init__(self, eta, batch_size):
        super().__init__(eta)

    def update_change(self, gradient):
        return self.eta * gradient


class Momentum(Scheduler):
    # take in batch size for unity but does not use it
    def __init__(self, eta: float, momentum: float, batch_size):
        super().__init__(eta)
        self.momentum = momentum
        self.change = 0

    def update_change(self, gradient):
        self.change = self.momentum * self.change + self.eta * gradient
        return self.change

    def reset(self):
        self.change = 0


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


class AdagradMomentum(Scheduler):
    def __init__(self, eta, momentum, batch_size):
        super().__init__(eta)
        self.G_t = None
        self.batch_size = batch_size
        self.momentum = momentum
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
        self.change = self.change * self.momentum + self.eta * gradient * G_t_inverse
        return self.change

    def reset(self):
        self.G_t = None
        self.change = 0


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
