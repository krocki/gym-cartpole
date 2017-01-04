# -*- coding: utf-8 -*-
# @Author: krocki
# @Date:   2017-01-03 12:42:45
# @Last Modified by:   krocki
# @Last Modified time: 2017-01-03 19:06:14

# based on Andrej Karpathy's PG code
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

import numpy as np
import cPickle as pickle
import gym
from gym import wrappers

# hyperparameters
H = 256 # number of hidden layer neurons
batch_size = 4 # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = True
record = False
episodes = 100000

history = []

env = gym.make('MountainCar-v0')
name = '/tmp/mountaincarv0'

if record: env = wrappers.Monitor(env, name)

observation = env.reset()
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
t = 0

D = env.observation_space.shape[0] # input dimensionality
A = env.action_space.n # output dimensionality (discrete)
I = np.eye(A)

if resume:
  model = pickle.load(open('car.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D)
  model['W2'] = np.random.randn(A,H) / np.sqrt(H)
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() }
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() }

def softmax(x):
	e = np.exp(x);
	sums = np.sum(e, axis=0);
	return e / sums;

def discount_rewards(r):
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
	running_add = running_add * gamma + r[t]
	discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0
  logp = np.dot(model['W2'], h)
  p = softmax(logp)
  return p, h

def policy_backward(eph, epdlogp):

  dW2 = np.dot(eph.T, epdlogp).T
  dh = np.dot(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

def sample(probs):

	r = np.random.rand(1)
	idx = 0

	if probs.shape[0] == 1:
		if r < probs[0]: idx = 0
		else: idx = 1

	else:
		cdf = np.cumsum(probs)
		for i in range(probs.shape[0]):
			if (cdf[i] >= r[0]):
				idx = i
				break

	return idx

t = 0

while episode_number < episodes:

  t = t + 1
  if render and episode_number % 1000 == 0: env.render()

  x = observation

  aprob, h = policy_forward(x)
  action = sample(aprob)
  xs.append(x)
  hs.append(h)
  y = I[action]

  dlogps.append(y - aprob)

  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward)

  if done:
	t = 0
	episode_number += 1

	epx = np.vstack(xs)
	eph = np.vstack(hs)
	epdlogp = np.vstack(dlogps)
	epr = np.vstack(drs)
	xs,hs,dlogps,drs = [],[],[],[] # reset array memory

	discounted_epr = discount_rewards(epr)
	discounted_epr -= np.mean(discounted_epr)
	discounted_epr /= np.std(discounted_epr)

	epdlogp *= discounted_epr
	grad = policy_backward(eph, epdlogp)
	for k in model: 
		grad_buffer[k] += grad[k]

	if episode_number % batch_size == 0:
	  for k,v in model.iteritems():
		g = grad_buffer[k]
		rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
		model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
		grad_buffer[k] = np.zeros_like(v)

	running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
	print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
	if episode_number % 10 == 0: pickle.dump(model, open('car.p', 'wb'))
	history.append(running_reward)
	reward_sum = 0
	observation = env.reset()
