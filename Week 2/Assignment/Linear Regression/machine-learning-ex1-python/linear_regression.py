import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Compute Cost function
def compute_cost(X,y,theta):
	m = X.shape[0]
	J = np.dot(np.transpose(np.dot(X,theta) - y),np.dot(X,theta) - y)/(2*m)
	return np.asscalar(J)
#

# Gradient descent
def gradient_descent(X,y,theta,alpha):
	m = X.shape[0]
	hypothesis = np.dot(X,theta)
	theta = theta - alpha*np.dot(np.transpose(X),(hypothesis - y))/m
	return theta
#

# Normalize features
def feature_normalize(X):
	n = X.shape[1]
	mean = np.zeros((n))
	sigma = np.zeros((n))
	for i in range(0,n):
		mean[i] = X[:,i].mean()
		sigma[i] = np.std(X[:,i])
		X[:,i] = (X[:,i]-mean[i])/sigma[i]
	return X,mean,sigma
#

#Load data from file and split to input and output
data = pd.read_csv('ex1data1.txt',sep=",", header = None)
data = np.array(data,dtype=float)
m = data.shape[0] # number example
n = data.shape[1] - 1 # number feature
X = data[:,0:n].reshape(m,n)
y = data[:,n].reshape(m,1)
#

# Plot data
#plt.figure(1)
#plt.plot(X,y,'r^')
#plt.show()
#

# Intial parameter
theta = np.zeros((n + 1,1))
iterations = 100
alpha = 0.1
J_history = np.zeros((iterations,1))
#

# Training
X,mean,sigma = feature_normalize(X) # nomarlize features
X = np.hstack((np.ones((m,1)),X))
for i in range(0,iterations):
	theta = gradient_descent(X,y,theta,alpha)
	J = compute_cost(X,y,theta)
	J_history[i] = J
	if i%10 == 0:
		print('Loss of iterator {}: {}'.format(i,J))
#

# Plot function cost per iter
plt.figure(2)
plt.plot(np.arange(1,J_history.shape[0]+1,1),J_history)
plt.show(block=False)
#

# Plot Contuor
plt.figure(3)
list_1 = np.arange(-20,20,0.05)
list_2 = np.arange(-20,20,0.05)
theta_1, theta_2 = np.meshgrid(list_1,list_2)
J = np.zeros(theta_1.shape)
for i in range(0,theta_1.shape[0]):
	for j in range(0,theta_1.shape[1]):
		J[i,j] = compute_cost(X,y,np.array([[theta_1[i,j]],[theta_2[i,j]]]))
plt.contour(theta_1,theta_2,J)
plt.plot(theta[0],theta[1],'r+')
plt.show(block=False)
#

# Plot Surface
fig = plt.figure(4)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(theta_1, theta_2, J, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.plot(theta[0],theta[1],'r+')
plt.show()
#

# #Test
# print("Theta: {}".format(theta))
# # X = np.array(([[1,(1650-mean[0])/sigma[0],(3-mean[1])/sigma[1]]]),dtype=float)
# # print (np.asscalar(np.dot(X,theta)))
# #