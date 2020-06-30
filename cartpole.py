# -*- coding: utf-8 -*-
# @Author: krocki
# @Date:   2017-01-03 12:42:45
# @Last Modified by:   krocki
# @Last Modified time: 2017-01-03 19:06:23

# based on Andrej Karpathy's PG code
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

import numpy as np
import pickle
import gym
from gym import wrappers

# hyperparameters
H = 256 # number of hidden layer neurons
batch_size = 4 # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = True
record = False
episodes = 1000

history = []

env = gym.make('CartPole-v1')
name = '/tmp/cartpolev1'


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
  model = pickle.load(open('cartpole.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(A,H) / np.sqrt(H)
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def softmax(x):
  e = np.exp(x);
  sums = np.sum(e, axis=0);
  return e / sums;

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
  # if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = softmax(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """

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
  if render and episode_number % 100 == 0: env.render()

  x = observation

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  action = sample(aprob)
  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = I[action]

  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    t = 0
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(eph, epdlogp)
    for k in model: 
      grad_buffer[k] += grad[k] # accumulate grad over batch

    if episode_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print('resetting env. episode reward total was {:}. running mean: {:}'.format(reward_sum, running_reward))
    if episode_number % 10 == 0: pickle.dump(model, open('cartpole.p', 'wb'))
    history.append(running_reward)
    reward_sum = 0
    observation = env.reset() # reset env
