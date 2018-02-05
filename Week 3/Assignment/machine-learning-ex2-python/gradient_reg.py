import numpy as np
from sigmoid import sigmoid

def compute_gradient_reg(theta, X, y, lamda):
    m = len(y)
    hypothesis = sigmoid(np.dot(X, theta))
    gradient = (1 / m) * np.dot(X.T, (hypothesis - y))
    gradient[1:] = gradient[1:] + lamda * theta[1:] / m
    return gradient
