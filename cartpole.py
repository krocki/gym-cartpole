# -*- coding: utf-8 -*-
# @Author: kmrocki
# @Date:   2016-12-21 09:39:30
# @Last Modified by:   kmrocki
# @Last Modified time: 2016-12-21 09:40:17

# based on tutorial from https://gym.openai.com/docs

import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action