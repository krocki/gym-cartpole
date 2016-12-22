# -*- coding: utf-8 -*-
# @Author: krocki
# @Date:   2016-12-21 10:22:24
# @Last Modified by:   krocki
# @Last Modified time: 2016-12-21 18:47:57

from NN import *
# import Linear

batchsize = 16

layers = [ 
			Linear(784, 100, batchsize), 
			ReLU(100, 100, batchsize), 
			Linear(100, 10, batchsize), 
			Softmax(10, 10, batchsize)
		 ]

nn = NN(layers)

nn.forward()