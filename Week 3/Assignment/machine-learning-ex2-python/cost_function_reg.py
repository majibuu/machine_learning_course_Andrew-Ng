import numpy as np
from sigmoid import sigmoid


def cost_function_reg(theta, X, y, lamda, return_grad=False):
    J = 0
    m = len(y)
    grad = np.zeros(theta.shape)
    hypothesis = sigmoid(np.dot(X,theta))
    J = np.dot(np.transpose(y),np.log(hypothesis)) + np.dot(np.transpose(1-y),np.log(1-hypothesis))
    J = -J/m + lamda * sum(np.power(theta[1:],2))
    grad = np.dot(np.transpose(X),(sigmoid(np.dot(X,theta)) - y))/(m)
    grad[1:] = grad[1:] + lamda * theta[1:] / m
    if return_grad == True:
        return np.asscalar(J), np.asarray(grad)
    elif return_grad == False:
        return np.asscalar(J)