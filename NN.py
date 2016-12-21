# -*- coding: utf-8 -*-
# @Author: krocki
# @Date:   2016-12-21 10:21:06
# @Last Modified by:   krocki
# @Last Modified time: 2016-12-21 10:46:36

# a simple implementation of a multilayer NN

class NN:

	layers = []

	def __init__(self):
		print "NN init"

	def forward(self):
		print "NN forward"

	def backward(self):
		print "NN backward"

class Layer:

	x = []
	y = []

	dx = []
	dy = []

	def __init__(self):
		print "layer init"

	def forward(self):
		raise NotImplementedError()

class Linear(Layer):

	W = []
	b = []

	dW = []
	db = []

	def __init__(self):
		Layer.__init__(self)
		print "linear init"

	def forward(self):
		print "linear forward"

class ReLU(Layer):

	def __init__(self):
		Layer.__init__(self)
		print "ReLU init"

	def forward(self):
		print "ReLU forward"