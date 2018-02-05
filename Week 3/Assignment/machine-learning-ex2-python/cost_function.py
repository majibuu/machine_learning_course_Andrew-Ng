import numpy as np
from sigmoid import sigmoid


def cost_function(theta, X, y, return_grad=False):
    J = 0
    m = len(y)
    grad = np.zeros(theta.shape)
    J = np.dot(np.transpose(y),np.log(sigmoid(np.dot(X,theta)))) + np.dot(np.transpose(1-y),np.log(1-sigmoid(np.dot(X,theta))))
    J = -J/m
    grad = np.dot(np.transpose(X),(sigmoid(np.dot(X,theta)) - y))/(m)
    if return_grad == True:
        return np.asscalar(J), np.asarray(grad)
    elif return_grad == False:
        return np.asscalar(J)