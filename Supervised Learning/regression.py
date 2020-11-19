import numpy as np
import math
from utils.data_manipulation import normalize


class L1Regularization():
    """Regularization for Lasso Regression"""
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, theta):
        return self.alpha * np.linalg.norm(theta)

    def grad(self, theta):
        return self.alpha * np.sign(theta)


class L2Regularization():
    """Regularization for Ridge Regression"""
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, theta):
        return self.alpha * 0.5 * theta.T.dot(theta)

    def grad(self, theta):
        return self.alpha * theta


class Regression(object):
    """
    Base regression model. Models the relationship between a scalar dependent variable
    y and the independent variable X.

    Parameters:
        n_iterations: float
            The number of training iterations the algorithm will tune the weights for.

        learning_rate: float
            The step length that will be used when updating the weights.
    """
    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initialize_weights(self, n_features):
        """Initialize weights randomly [-1/N, 1/N]"""
        limit = 1 / math.sqrt(n_features)
        self.theta = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, X, y):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self.initialize_weights(n_features=X.shape[1])

        # Do gradient descent for n_iterations
        for i in range(n_iterations):
            y_pred = X.dot(self.theta)

            # Calculate l2 loss
            mse = np.mean(0.5 * (y - y_pred)**2 + self.reguralization(self.theta))
            self.training_errors.append(mse)

            # Gradient of l2 loss w.r.t theta
            grad_theta = -(y - y_pred).dot(X) + self.regularization.grad(self.theta)

            # Update the weights
            self.theta -= self.learning_rate * grad_theta

    def predict(self, X):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.theta)
        return y_pred


class LinearRegression(Regression):
    """
    Linear Model

    Parameters:
        n_iterations: float
            The number of training iterations the algorithm will tune the weights for.

        learning_rate: float
            The step length that will be used when updating the weights.

        gradient_descent: boolean
            True or False depending on the use of gradient descent during training.
            If False, then we use batch optimization by least squares approach.
    """
    def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=True):
        self.gradient_descent = gradient_descent
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(n_iterations=n_iterations,
                                               learning_rate=learning_rate)

    def fit(self, X, y):
        # If not gradient descent => Least Squares approximation of theta
        if not self.gradient_descent:
            # Insert constant ones for bias weights
            X = np.insert(X, 0, 1, axis=1)

            # Calculate weights by Least Squares (using Moore-Penrose pseudo-inverse)
            U, S, V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.theta = X_sq_reg_inv.dot(X.T).dot(y)
        else:
            super(LinearRegression, self).fit(X, y)