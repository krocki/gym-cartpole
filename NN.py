# -*- coding: utf-8 -*-
# @Author: krocki
# @Date:   2016-12-21 10:21:06
# @Last Modified by:   krocki
# @Last Modified time: 2016-12-21 19:15:42
#
# a simple implementation of a feedforward NN

import numpy as np

class NN:

	layers = [] # stack of layers

	def __init__(self, layers):
		self.layers = layers

	def forward(self):
		for l in self.layers:
			l.forward()

	def backward(self):
		for l in reversed(self.layers):
			l.backward()

class Layer:

	# inputs, outputs, input grads, output grads
	x, y, dx, dy = [], [], [], []

	def __init__(self, inputs, outputs, batchsize):
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

	# weights x-y, biases, w grads, b grads
	W, b, dW, db = [], [], [], []

	def __init__(self, inputs, outputs, batchsize):
		Layer.__init__(self, inputs, outputs, batchsize)
		self.W = 0.1 * np.random.randn(outputs, inputs)
		self.b = np.zeros((outputs, 1), dtype=np.float)
		self.resetgrads();

	def forward(self):
		print "linear forward"
		self.y = np.dot(self.W, self.x) + self.b;

	def backward(self):
		print "linear back"
		self.dW = np.dot(self.dy, self.x.T)
		self.db = np.sum(self.dy, axis=1)
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

	def forward(self):
		print "softmax forward"
		self.y = softmax(self.x);

	def backward(self):
		print "softmax back"
		self.dx = self.dy - self.y;

class ReLU(Layer):

	def __init__(self, inputs, outputs, batchsize):
		Layer.__init__(self, inputs, outputs, batchsize)

	def forward(self):
		print "relu forward"
		self.y = rectify(self.x);

	def backward(self):
		print "relu back"
		self.dx = drectify(self.y) * self.dy;

#helper functions
def rectify(x):
	return np.maximum(x, 0);

def drectify(x):
	return x > 0;

#probs(class) = exp(x, class)/sum(exp(x, class))
def softmax(x):
	e = np.exp(x);
	sums = np.sum(e, axis=0);
	return e / sums;

#cross-entropy
def crossentropy(predictions, targets):
	return -np.sum (np.log(np.sum(targets * predictions, axis=0)))
