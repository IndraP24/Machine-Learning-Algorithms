import numpy as np
import pandas as pd
import mlfromscratch.utils


def reg_least_sqr(X, y):
    """ Calculating regression coefficients using the Least Squares method """
    X_mean, y_mean = np.mean(X), np.mean(y)
    
    num = np.sum((X - X_mean) * (y - y_mean))
    den = np.sum((X - X_mean)**2)
    
    m = num/den
    c = y_mean - m*X_mean
    
    return m, c


def reg_grad_desc(X, y, L = 0.001, epochs = 1000):
    """ Calculating regression coefficients using the Gradient Descent method """
    m = 0
    c = 0
    
    n = float(len(X))
    
    for i in range(epochs):
        y_pred = m*X + c
        D_m = (-2/n) * np.sum(X * (y - y_pred))
        D_c = (-2/n) * np.sum(y - y_pred)
        
        m = m - L * D_m
        c = c - L * D_c
        
    return m, c


def reg_predict_score(X_train, X_test, y_train, y_test, grad_desc=True, least_sqr=False):
    """ Predicting the target variable using suitable method and calculating the mean squared error between the actual and predicted target variable"""
    
    if grad_desc==True and least_sqr==False:
        m, c = reg_grad_desc(X_train, y_train)
    elif grad_desc==False and least_sqr==True:
        m, c = reg_least_sqr(X_train, y_train)
    
    y_pred = m*X_test + c
    score = utils.mean_squared_error(y_test, y_pred)
    return  y_pred, score