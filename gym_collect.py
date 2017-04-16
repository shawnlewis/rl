#!/bin/python

import sys
import gym
import h5py
import numpy as np
import scipy
import scipy.misc

def preprocess_image(im):
    im = np.average(im, axis=2)
    return scipy.misc.imresize(im, (110, 84))

def main(argv):
    env_name = argv[1]
    f = h5py.File("steps_%s.hdf5" % env_name, "w")
    env = gym.make(env_name)
    env.reset()
    obs = []
    acts = []
    rews = []
    dones = []
    for i in xrange(100000):
        if i % 1000 == 0:
            print i
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            env.reset()
        obs.append(preprocess_image(observation))
        acts.append(action)
        rews.append(reward)
        dones.append(done)
    f['obs'] = np.array(obs)
    f['acts'] = np.array(acts)
    f['rews'] = np.array(rews)
    f['dones'] = np.array(dones)

if __name__ == '__main__':
    main(sys.argv)
