import numpy as np

def map_feature(X1, X2):
	degree = 6
	out = np.ones((X1.shape[0],1))
	for i in range(1,degree+1,1):
		for j in range(0,i+1,1):
			out = np.hstack((out,(np.power(X1,i-j)*np.power(X2,j)).reshape(X1.shape[0],1)))
	return out