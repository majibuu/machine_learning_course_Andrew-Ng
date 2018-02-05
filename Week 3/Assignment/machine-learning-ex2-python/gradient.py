import numpy as np
from sigmoid import sigmoid

def compute_gradient(theta, X, y):
    m = len(y)
    hypothesis = sigmoid(np.dot(X, theta))
    gradient = (1 / m) * np.dot(X.T, (hypothesis - y))
    return gradient