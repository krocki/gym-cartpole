# -*- coding: utf-8 -*-
# @Author: krocki
# @Date:   2016-12-23 17:56:26
# @Last Modified by:   krocki
# @Last Modified time: 2016-12-23 18:18:40

import numpy as np

class Policy:

	def __init__(self):
		pass

	def forward(self, inputs):
		raise NotImplementedError()

class ReactiveCartPolicy:

	def __init__(self):
		pass

	def forward(self, inputs):
		if inputs[3] < 0:
			action = 0
		if inputs[3] > 0:
			action = 1
		return action
