import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


def initialize_weights(INPUT_DIM, NUM_CLASSES):
  """Randomly initialize the model's weights W and bias b."""
  W = 0.01 * np.random.randn(INPUT_DIM, NUM_CLASSES)
  b = np.zeros((1, NUM_CLASSES))
  return W, b


def forward_prop(X_train, W, b):
  """Feed inputs X into the model to receive the logits (z = X . W). Apply the softmax operation on the logits to get the class probabilies y_hat in one-hot encoded form. For example, if there are three classes, the predicted class probabilities could look like [0.3, 0.3, 0.4].
  y_hat = softmax(z) = softmax(X.W) = e^(X.W)_y / sum_j e^(X.W) """
  logits = np.dot(X_train, W) + b
  return logits


def soft_norm(logits):
  """Normalization via softmax to obtain class probabilities"""
  exp_logits = np.exp(logits)
  y_hat = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
  return y_hat


def cost(y_hat, y_train):
  """Compare the predictions y_hat (ex. [0.3, 0.3, 0.4]]) with the actual target values y (ex. class 2 would look like [0, 0, 1]) with the objective (cost) function to determine loss J. A common objective function for logistics regression is cross-entropy loss.
J(theta) = - sum_i ln(y_hat_i}) = - sum_i ln(e^(X_i.W_y) / sum_j e^(X_i.W)"""
  correct_class_logprobs = -np.log(y_hat[range(len(y_hat)), y_train])
  loss = np.sum(correct_class_logprobs) / len(y_train)

  
 def back_prop(X_train, y_train, y_hat):
  """Calculate the gradient of loss J(theta) w.r.t the model weights assuming that our classes are mutually exclusive."""
  dscores = y_hat
  dscores[range(len(y_hat)), y_train] -+ 1
  dscores /= len(y_train)
  dW = np.dot(X_train.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)
  return dW, db

              
def update(W, b, dW, db, LEARNING_RATE):
  """Update the weights W using a small learning rate alpha."""
  LEARNING_RATE = 1e-1
  W += -LEARNING_RATE * dW
  b += -LEARNING_RATE * db
  return W, b


