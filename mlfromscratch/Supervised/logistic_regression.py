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
  """Feed inputs X into the model to receive the logits (z = X . W). Apply the softmax operation on the logits to get the class probabilies $\hat{y}$ in one-hot encoded form. For example, if there are three classes, the predicted class probabilities could look like [0.3, 0.3, 0.4].
$ \hat{y} = softmax(z) = softmax(XW) = \frac{e^{XW_y}}{\sum_j e^{XW}} $"""
  logits = np.dot(X_train, W) + b
  return logits


def soft_norm(logits):
  """Normalization via softmax to obtain class probabilities"""
  exp_logits = np.exp(logits)
  y_hat = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
  return y_hat
