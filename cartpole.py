# -*- coding: utf-8 -*-
# @Author: kmrocki
# @Date:   2016-12-21 09:39:30
# @Last Modified by:   krocki
# @Last Modified time: 2017-01-02 14:33:10

# based on tutorial from
# https://gym.openai.com/docs

import gym
from Policy import *
import numpy as np

env = gym.make('CartPole-v0')

policy = NeuralPolicy()
episode_number = 0
learningrate = 1e-2
gamma = 0.99 # discount factor for reward
performance = 0

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
	running_add = running_add * gamma + r[t]
	discounted_r[t] = running_add
  return discounted_r

for i_episode in range(1000):

	x = env.reset()
	xs, hs, dlogps, drs = [], [], [], []
	reward_sum = 0

	for t in range(200):

		env.render()
		# print 'observations \n' + str(x)

		action, aprob, h = policy.forward(
			np.expand_dims(x, axis=1))

		# print '-- action \n' + str(action)
		# print '-- prob \n' + str(aprob)
		# print '-- hiddens'
		# print h

		xs.append(x)  # observation
		hs.append(h.T)  # hidden state

		y = action
		# grad that encourages the action that
		# was taken to be taken (see
		# http://cs231n.github.io/neural-networks-2/#losses
		# if confused)
		dlogps.append(y - aprob)

		x, reward, done, info = env.step(action)
		reward_sum += reward

		drs.append(reward)

		# print '--- reward \n' + str(reward)
		# print '--- reward sum\n' + str(reward_sum)
		# print '--- info \n' + str(info)

		if done:

			episode_number += 1
			epx = np.vstack(xs)
			eph = np.vstack(hs)
			epdlogp = np.vstack(dlogps)
			epr = np.vstack(drs)
			xs,hs,dlogps,drs = [],[],[],[] # reset array memory

			sum_epr = discount_rewards(epr)
			sum_epr -= np.mean(sum_epr)
			sum_epr /= np.std(sum_epr)

			epdlogp *= sum_epr # modulate the gradient with advantage (PG magic happens right here.)
			policy.backward(epx, eph, epdlogp)
			policy.adapt(learningrate)

			performance = performance * 0.9 + 0.1 * (t + 1)
			print("Episode finished after {} timesteps".format(t + 1))
			print str(performance)
			break

		if t == 199:
			print("episode {} - solved".format(i_episode + 1))