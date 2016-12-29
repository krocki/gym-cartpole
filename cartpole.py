# -*- coding: utf-8 -*-
# @Author: kmrocki
# @Date:   2016-12-21 09:39:30
# @Last Modified by:   krocki
# @Last Modified time: 2016-12-28 19:12:44

# based on tutorial from https://gym.openai.com/docs

import gym
from Policy import *
import numpy as np

env = gym.make('CartPole-v0')

policy = NeuralPolicy()

for i_episode in range(20):
	observation = env.reset()
	for t in range(100):

		env.render()
		print 'observations \n' + str(observation)

		action = policy.forward(np.expand_dims(observation, axis=1))

		observation, reward, done, info = env.step(action)

		# if done:
		#     print("Episode finished after {} timesteps".format(t+1))
		#     break