import numpy as np
from mlfromscratch.utils import mean_squared_error


def initialize_parameters(lenw):
    w = np.random.randn(1, lenw)
    b = 0
    return w, b


def forward_prop(X, w, b):
    z = np.dot(w, X) + b
    return z


def cost_function(z, y):
    m = y.shape[1]
    J = (1/(2*m)) * np.sum(np.square(z-y))
    return J


def back_prop(X, y, z):
    m = y.shape[1]
    dz = (1/m)*(z-y)
    dw = np.dot(dz, X.T)
    db = np.sum(dz)
    return dw, db


def gradient_descent_update(w, b, dw, db, learning_rate):
    w = w - learning_rate*dw
    b = b - learning_rate*db
    return w, b


def linear_regression_model(X_train, y_train, X_test, y_test, learning_rate=0.001, epochs=100):

    lenw = X_train.shape[0]
    w, b = initialize_parameters(lenw)
    costs_train = []

    for i in range(1, epochs+1):
        z_train = forward_prop(X_train, w, b)
        cost_train = cost_function(z_train, y_train)
        dw, db = back_prop(X_train, y_train, z_train)
        w, b = gradient_descent_update(w, b, dw, db, learning_rate)

        # store training cost in a list
        if i % 10 == 0:
            costs_train.append(cost_train)

        # MSE Train
        mse_train = mean_squared_error(y_train, z_train)
        print(mse_train, w)
        
        # MSE Test
        z_test = forward_prop(X_test, w, b)
        mse_test = mean_squared_error(y_test, z_test)

        print('Epochs', i, '/', epochs, ': ')
        print('MSE Train', mse_train, 'MSE Test', mse_test)