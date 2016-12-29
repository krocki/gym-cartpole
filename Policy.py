# -*- coding: utf-8 -*-
# @Author: krocki
# @Date:   2016-12-23 17:56:26
# @Last Modified by:   krocki
# @Last Modified time: 2016-12-28 19:14:05

import numpy as np
from NN import *

class Policy:

	def __init__(self):
		pass

	def forward(self, inputs):
		raise NotImplementedError()

class NeuralPolicy(Policy):

	nn = []
	batchsize = 1

	def __init__(self):
		Policy.__init__(self)

		layers = [ 
			Linear(4, 10, self.batchsize),
			ReLU(10, 10, self.batchsize), 
			Linear(10, 2, self.batchsize), 
			Softmax(2, 1, self.batchsize)
		]

		self.nn = NN(layers)

	def forward(self, inputs):
		action = self.nn.forward(inputs)
		print 'actions \n' + str(action)
		return sample(action)

class RandomPolicy(Policy):

	def __init__(self):
		Policy.__init__(self)

	def forward(self, inputs):
		return np.random.randint(2)

class ReactiveCartPolicy(Policy):

	def __init__(self):
		Policy.__init__(self)

	def forward(self, inputs):
		if inputs[3] < 0:
			action = 0
		if inputs[3] > 0:
			action = 1
		return action

def sample(probs):

	print probs
	cdf = np.cumsum(probs)
	r = np.random.rand(1)

	idx = 0

	for i in range(probs.shape[0]):
		if (cdf[i] >= r[0]):
			idx = i
			break
	print idx
	return idx