import numpy as np
from mlfromscratch.utils import mean_squared_error


def initialize_parameters(lenx, leny):
    w = 0.01 * np.random.randn(lenx, leny)
    b = np.zeros((1, 1))
    return w, b


def forward_prop(X, w, b):
    z = np.dot(X, w) + b
    return z


def cost_function(z, y):
    m = len(y)
    J = (1/m) * np.sum(np.square(y-z))
    return J


def back_prop(X, y, z):
    m = len(y)
    dw = -(2/m) * np.sum((y-z) * X.T)
    db = -(2/m) * np.sum((y-z) * 1)
    return dw, db


def gradient_descent_update(w, b, dw, db, learning_rate):
<<<<<<< HEAD
    w += -learning_rate * dw
=======
    w += -learning_rate * dW
>>>>>>> 93fe9c4aa5da666405c8a2be8e145d3ea778a90a
    b += -learning_rate * db
    return w, b


def linear_regression_model(X_train, y_train, X_test, y_test, learning_rate=0.001, epochs=100):

    lenx = X_train.shape[1]
    leny = y_train.shape[1]
    w, b = initialize_parameters(lenx, leny)

    for i in range(epochs):
        z_train = forward_prop(X_train, w, b)
        cost_train = cost_function(z_train, y_train)
        # show progress
        if i%10 == 0:
            print (f"Epoch: {i}, loss: {cost_train:.3f}")
        dw, db = back_prop(X_train, y_train, z_train)
        w, b = gradient_descent_update(w, b, dw, db, learning_rate)

    
    z_train = w*X_train + b
    z_test = w*X_test + b
    
    # MSE Train
    mse_train = mean_squared_error(y_train, z_train)
    print("Train MSE:", mse_train, "Weight: ",w)
        
    # MSE Test
    mse_test = mean_squared_error(y_test, z_test)
    print("Test MSE:", mse_test, "Weight: ",w)