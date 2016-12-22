# -*- coding: utf-8 -*-
# @Author: krocki
# @Date:   2016-12-21 10:21:06
# @Last Modified by:   krocki
# @Last Modified time: 2016-12-21 18:53:35

# a simple implementation of a multilayer NN
import numpy as np

class NN:

	layers = [] # stack of layers

	def __init__(self, layers):

		print "NN init"
		self.layers = layers

	def forward(self):

		print "NN forward"
		for l in self.layers:
			print l
			l.forward()

	def backward(self):

		print "NN backward"

class Layer:

	x = [] # inputs
	y = [] # outputs

	dx = [] # input grads
	dy = [] # output grads

	def __init__(self, inputs, outputs, batchsize):

		print "layer init"
		self.x = np.zeros((inputs, batchsize), dtype=np.float)
		self.y = np.zeros((outputs, batchsize), dtype=np.float)
		self.dx = np.zeros_like(self.x, dtype=np.float)
		self.dy = np.zeros_like(self.y, dtype=np.float)

	def forward(self):
		raise NotImplementedError()

	def backward(self):
		raise NotImplementedError()

	def resetgrads(self):
		pass

	def applygrads(self, alpha):
		pass

class Linear(Layer):

	W = [] # weights x-y
	b = [] # biases

	dW = [] # W grads
	db = [] # b grads

	def __init__(self, inputs, outputs, batchsize):

		Layer.__init__(self, inputs, outputs, batchsize)
		self.W = 0.1 * np.random.randn(outputs, inputs)
		self.b = np.zeros((outputs, 1), dtype=np.float)
		self.resetgrads();

		print "linear init"

	def forward(self):

		self.y = np.dot(self.W, self.x) + self.b;
		print "linear forward"

	def backward(self):

		self.dW = np.dot(self.dy, self.x.T)
		self.db = np.sum(self.dy, 2)
		self.dx = np.dot(self.W.T, self.dy)

	def resetgrads(self):
		
		self.dW = np.zeros_like(self.W, dtype=np.float)
		self.db = np.zeros_like(self.db, dtype=np.float)

	def applygrads(self, alpha):

		self.b += alpha * self.db;
		self.W += alpha * self.dW;

class Softmax(Layer):

	def __init__(self, inputs, outputs, batchsize):

		Layer.__init__(self, inputs, outputs, batchsize)
		print "softmax init"

	def forward(self):

		self.y = softmax(self.x);
		print "softmax forward"

	def backward(self):

		self.dx = self.dy - self.y;


class ReLU(Layer):

	def __init__(self, inputs, outputs, batchsize):

		Layer.__init__(self, inputs, outputs, batchsize)
		print "ReLU init"

	def forward(self):

		self.y = rectify(self.x);
		print "ReLU forward"

	def backward(self):

		self.dx = drectify(self.y) * self.dy;


#helper functions
def rectify(x):

	return np.maximum(x, 0);

def drectify(x):

	return x > 0;

def softmax(x):

	#probs(class) = exp(x, class)/sum(exp(x, class))

	e = np.exp(x);

	sums = np.sum(e, axis=0);

	return e / sums;

def xentropy(predictions, targets):

	return -np.sum (np.log(np.sum(targets * predictions, axis=0)))
