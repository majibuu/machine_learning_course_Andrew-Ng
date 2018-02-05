import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize
from sigmoid import sigmoid
from cost_function import cost_function
from gradient import compute_gradient


# Load data from file and split to input and output
data = np.array(pd.read_csv('ex2data2.txt', sep=",", header=None), dtype=float)
m = data.shape[0]  # number examples
n = data.shape[1] - 1  # number features
X = data[:, 0:n]  # training data
y = data[:, n]  # test data
iterations = 4000000
alpha = 0.001

# Plot data
plt.figure(1)
positive_X = X[np.nonzero(y == 1)[0], :]
negative_X = X[np.nonzero(y == 0)[0], :]
plt.plot(positive_X[:, 0], positive_X[:, 1], 'b+')
plt.plot(negative_X[:, 0], negative_X[:, 1], 'yo')
plt.legend(['Admitted', 'Not admitted'], loc=5, borderaxespad=0.)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show(block=False)
#

# Add bias
X = np.hstack((np.ones((m, 1)), X))
#

# Train model by advanced optimize
initial_theta = np.zeros((n+1),dtype=float)
cost, grad = cost_function(initial_theta,X,y,True)
print('Cost at initial theta (zeros): {}'.format(cost))
print('Gradient at initial theta (zeros): {}'.format(grad))
#myargs=(X, y, True)
#theta = sp.optimize.fmin_tnc(cost_function, x0=initial_theta, args=myargs)
myargs=(X, y)
theta = sp.optimize.fmin_ncg(cost_function, x0=initial_theta, fprime=compute_gradient,args=myargs)
print(theta)

# Train model by gradient descent
# theta = np.zeros((n + 1), dtype=float)
# grad_old = np.zeros((n + 1), dtype=float)
# for i in range(0, iterations):
#     cost, grad = cost_function(theta, X, y, True)
#     theta = theta - 0.9*alpha*grad_old - alpha*grad
#     grad_old = grad
#     if(i % 50 == 0):
#         print("Training error in iteration {} : {}".format(
#             i, cost_function(theta, X, y, False)))
#         print("-------")
# print(theta)

# Plot decision boundary
# x_1 = np.min(X[:,1])
# x_2 = np.max(X[:,1])
# y_1 = -(theta[0] + x_1*theta[1])/theta[2]
# y_2 = -(theta[0] + x_2*theta[2])/theta[1]
# plt.plot([x_1,x_2],[y_1,y_2])
# plt.legend(['Admitted', 'Not admitted', 'Decision boundary'], loc=5, borderaxespad=0.)
# plt.show()
#