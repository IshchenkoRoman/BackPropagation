import os
import os.path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.optimize import minimize
from PIL import Image

class BP():
    
    def __init__(self, path_data, path_weight):

    	try:
    		self._df = loadmat(path_data)
    	except IOError:
    		print("Data file doesn't exist")
    		raise
    	try:
    		self._w = loadmat(path_weight)
    	except IOError:
    		print("Weight doesn't exist")
    		raise

    	self.X = self._df['X']
    	self.X = np.c_[np.ones((self.X.shape[0], 1)), self.X]
    	self.y = self._df['y']
    	self.theta1 = self._w['Theta1']
    	self.theta2 = self._w['Theta2']
    	self.params = np.r_[self.theta1.ravel(), self.theta2.ravel()] # Row presentation of theta1 and theta2 (just like in concepts)
    	self.X_shape = self.X.shape
    	self.y_shape = self.y.shape
    	self.t1_shape = self.theta1.shape
    	self.t2_shape = self.theta2.shape

    def sigmoid(self, data):

    	return (1 / (1 + np.exp(-data)))

    def sigmoidGradient(self, data):

    	return (self.sigmoid(data) * (1 - self.sigmoid(data)))

    def randInitializeWeights(self, L_out, L_in):

    	epsilon = (6 ** (0.5)) / ((L_out + L_in) ** (0.5))

    def plotRandomDigits(self):

    	fig = plt.figure()

    	for i in range (1, 11):
    		sample = np.random.choice(self.X_shape[0], 10)
    		fig.add_subplot(10, 1, i)
    		plt.imshow(self.X[sample, 1:].reshape(-1, 20).T, cmap=None)
    		plt.axis('off')
    	plt.show()

    def forward_propagation(self, theta1, theta2, X):

    	z_2 = np.dot(X, theta1.T)
    	a_2 = self.sigmoid(z_2)
    	a_2 = np.c_[np.ones((a_2.shape[0], 1)), a_2]
    	z_3 = np.dot(a_2, theta2.T)
    	a_3 = self.sigmoid(z_3)

    	return (z_2, a_2, z_3, a_3)


    def costFunction(self, params, input_layer_size, hidden_layer_size, num_labels, X, y, lambd):

    	theta1 = params[0:(hidden_layer_size*(input_layer_size + 1))].reshape(hidden_layer_size, (input_layer_size + 1))
    	theta2 = params[hidden_layer_size*(input_layer_size + 1):].reshape(num_labels, (hidden_layer_size + 1))
    	z_2, a_2, z_3, a_3 = self.forward_propagation(theta1, theta2, X)
    	log1 = np.log(a_3)
    	log2 = np.log(1 - a_3)
    	y_k = pd.get_dummies(y.ravel()).as_matrix()
    	# y_k = np.identity(10)
    	# y_k = np.repeat(y_k, 500, axis = 0)
    	# y_k = np.r_[y_k[-500:], y_k[:4500]]
    	cost = y_k * log1 + ((1 - y_k) * log2)

    	regularization = (np.sum(theta1[:,1:] ** 2) + np.sum(theta2[:,1:] ** 2)) * (lambd / (2 * len(y)))
    	res = (-np.sum(cost) / self.y_shape[0]) + regularization

    	d_3 = (a_3 - y_k)
    	d_2 = np.dot(d_3, theta2)[:,1:] * self.sigmoidGradient(z_2)


    	delta1 = np.dot(d_2.T, X) / self.y_shape[0]
    	delta2 = np.dot(d_3.T, a_2) / self.y_shape[0]

    	t1 = np.c_[np.ones((theta1.shape[0], 1)), theta1[:,1:]]
    	t2 = np.c_[np.ones((theta2.shape[0], 1)), theta2[:,1:]]

    	theta1_ = np.c_[np.ones((theta1.shape[0], 1)), theta1[:,1:]]
    	theta2_ = np.c_[np.ones((theta2.shape[0], 1)), theta2[:,1:]]

    	theta1_grad = delta1 / self.y_shape[0] + (theta1_*lambd) / self.y_shape[0]
    	theta2_grad = delta2 / self.y_shape[0] + (theta2_*lambd) / self.y_shape[0]

    	grad = np.concatenate((np.ravel(theta1_grad), np.ravel(theta2)))

    	return (res, grad)

    def optimize(self, params, input_layer_size, hidden_layer_size, num_labels, X, y, lambd):

    	fmin = minimize(fun=self.costFunction, x0=params, args=(input_layer_size, hidden_layer_size, num_labels, X, y, lambd),
    					method='TNC', jac=True, options={'maxiter': 250})
    	return (fmin)

    def prediction(self, params, input_layer_size, hidden_layer_size, num_labels, X, y, lambd):

    	fmin = self.optimize(params, input_layer_size, hidden_layer_size, num_labels, X, y, lambd)
    	theta1 = fmin.x[0:(hidden_layer_size*(input_layer_size + 1))].reshape(hidden_layer_size, (input_layer_size + 1))
    	theta2 = fmin.x[hidden_layer_size*(input_layer_size + 1):].reshape(num_labels, (hidden_layer_size + 1)) 

    	h = self.forward_propagation(theta1, theta2, X)[3]

    	y_prediction = np.array(np.argmax(h, axis=1) + 1)

    	print('Accuracy = {0}'.format(np.mean(y_prediction == y.ravel()) * 100))

def main():
    
    path_data = os.getcwd() + '/ex4data1.mat'
    path_weight = os.getcwd() + '/ex4weights.mat'
    bp = BP(path_data, path_weight)
    print(bp.X_shape, bp.y_shape, bp.t1_shape, bp.t2_shape)
    # bp.plotRandomDigits()
    # gradient in dimension (1, (theta1.shape[0] * theta1.shape[1] + theta2.shape[0] * theta2.shape[0]))
    cost, gradinet = bp.costFunction(bp.params, 400, 25, 10, bp.X, bp.y, 1)
    # bp.optimize(bp.params, 400, 25, 10, bp.X, bp.y, 1)
    bp.prediction(bp.params, 400, 25, 10, bp.X, bp.y, 1)

if __name__ == '__main__':
    main()