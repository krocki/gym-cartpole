# -*- coding: utf-8 -*-
# @Author: krocki
# @Date:   2016-12-23 17:56:26
# @Last Modified by:   krocki
# @Last Modified time: 2017-01-02 12:37:38

import numpy as np
from NN import *


# class Policy:

#     def __init__(self):
#         pass

#     def forward(self, inputs):
#         raise NotImplementedError()

#     def backward(self, eph, epdlogp):
#     	raise NotImplementedError()

#     def adapt(self, alpha):
#     	raise NotImplementedError()

class NeuralPolicy():

	nn = []
	batchsize = 1

	def __init__(self):
		# Policy.__init__(self)

		layers = [
			Linear(4, 64, self.batchsize),
			ReLU(64, 64, self.batchsize),
			Linear(64, 1, self.batchsize),
			Sigmoid(1, 1, self.batchsize)
			# Softmax(2, 1, self.batchsize)
		]

		self.nn = NN(layers)
		self.reset()

	def reset(self):
		for i in reversed(range(self.nn.num_layers)):
			self.nn.layers[i].resetgrads()
			self.nn.layers[i].backward()
			self.nn.layers[i].resetgrads()

	def backward(self, epx, eph, epdlogp):

		self.nn.layers[2].dW += np.dot(eph.T, epdlogp).ravel()
		dh = np.outer(epdlogp, self.nn.layers[2].W)
		dh[eph <= 0] = 0 # backpro prelu
		self.nn.layers[0].dW += np.dot(dh.T, epx)

	def adapt(self, alpha):
		self.nn.adapt(alpha)
		self.reset()

	def forward(self, inputs):
		action = self.nn.forward(inputs)
		return sample(action), action, self.nn.layers[1].y

# class RandomPolicy(Policy):

#     def __init__(self):
#         Policy.__init__(self)

#     def forward(self, inputs):
#         return np.random.randint(2)


# class ReactiveCartPolicy(Policy):

#     def __init__(self):
#         Policy.__init__(self)

#     def forward(self, inputs):
#         if inputs[3] < 0:
#             action = 0
#         if inputs[3] > 0:
#             action = 1
#         return action


def sample(probs):

	r = np.random.rand(1)
	idx = 0

	if probs.shape[0] == 1:

		if r < probs[0]:
			idx = 0
		else:
			idx = 1

	else:

		cdf = np.cumsum(probs)

		print 'cdf, before sampling'
		print cdf

		for i in range(probs.shape[0]):
			if (cdf[i] >= r[0]):
				idx = i
				break

	return idx
