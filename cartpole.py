# -*- coding: utf-8 -*-
# @Author: kmrocki
# @Date:   2016-12-21 09:39:30
# @Last Modified by:   krocki
# @Last Modified time: 2016-12-21 10:25:24

# based on tutorial from https://gym.openai.com/docs

import gym

env = gym.make('CartPole-v0')

for i_episode in range(20):
	observation = env.reset()
	for t in range(100):
		env.render()
		print(observation)

		# plug something in here:
		action = env.action_space.sample()
		#

		observation, reward, done, info = env.step(action)

		# if done:
		    # print("Episode finished after {} timesteps".format(t+1))
		    # break