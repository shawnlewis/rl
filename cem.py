import numpy as np
import gym
from gym.spaces import Discrete, Box

# ================================================================
# Policies
# ================================================================

class DeterministicDiscreteActionNNPolicy(object):

    def __init__(self, theta, hidden_dim, ob_space, ac_space):
        """
        dim_ob: dimension of observations
        n_actions: number of actions
        theta: flat vector of parameters
        """
        dim_ob = ob_space.shape[0]
        n_actions = ac_space.n
        assert len(theta) == (dim_ob + 1) * hidden_dim + (hidden_dim + 1) * n_actions
        w1end = dim_ob * hidden_dim
        self.W1 = theta[0 : w1end].reshape(dim_ob, hidden_dim)
        b1end = w1end+hidden_dim
        self.b1 = theta[w1end : b1end].reshape(1, hidden_dim)
        w2end = b1end + hidden_dim * n_actions
        self.W2 = theta[b1end : w2end].reshape(hidden_dim, n_actions)
        self.b2 = theta[w2end:].reshape(1, n_actions)

    def act(self, ob):
        """
        """
        hidden = ob.dot(self.W1) + self.b1
        hidden[hidden < 0] = 0
        y = hidden.dot(self.W2) + self.b2
        a = y.argmax()
        return a

class DeterministicDiscreteActionLinearPolicy(object):

    def __init__(self, theta, ob_space, ac_space):
        """
        dim_ob: dimension of observations
        n_actions: number of actions
        theta: flat vector of parameters
        """
        dim_ob = ob_space.shape[0]
        n_actions = ac_space.n
        assert len(theta) == (dim_ob + 1) * n_actions
        self.W = theta[0 : dim_ob * n_actions].reshape(dim_ob, n_actions)
        self.b = theta[dim_ob * n_actions : None].reshape(1, n_actions)

    def act(self, ob):
        """
        """
        y = ob.dot(self.W) + self.b
        a = y.argmax()
        return a

class DeterministicContinuousActionNNPolicy(object):

    def __init__(self, theta, hidden_dim, ob_space, ac_space):
        """
        dim_ob: dimension of observations
        n_actions: number of actions
        theta: flat vector of parameters
        """
        self.ac_space = ac_space
        dim_ob = ob_space.shape[0]
        dim_ac = ac_space.shape[0]
        assert len(theta) == (dim_ob + 1) * hidden_dim + (hidden_dim + 1) * dim_ac
        w1end = dim_ob * hidden_dim
        self.W1 = theta[0 : w1end].reshape(dim_ob, hidden_dim)
        b1end = w1end+hidden_dim
        self.b1 = theta[w1end : b1end]
        w2end = b1end + hidden_dim * dim_ac
        self.W2 = theta[b1end : w2end].reshape(hidden_dim, dim_ac)
        self.b2 = theta[w2end:]

    def act(self, ob):
        """
        """
        hidden = ob.dot(self.W1) + self.b1
        hidden[hidden < 0] = 0
        y = hidden.dot(self.W2) + self.b2
        a = np.clip(y, self.ac_space.low, self.ac_space.high)
        return a

class DeterministicContinuousActionLinearPolicy(object):

    def __init__(self, theta, ob_space, ac_space):
        """
        dim_ob: dimension of observations
        dim_ac: dimension of action vector
        theta: flat vector of parameters
        """
        self.ac_space = ac_space
        dim_ob = ob_space.shape[0]
        dim_ac = ac_space.shape[0]
        assert len(theta) == (dim_ob + 1) * dim_ac
        self.W = theta[0 : dim_ob * dim_ac].reshape(dim_ob, dim_ac)
        self.b = theta[dim_ob * dim_ac : None]

    def act(self, ob):
        a = np.clip(ob.dot(self.W) + self.b, self.ac_space.low, self.ac_space.high)
        return a

def do_episode(policy, env, num_steps, render=False):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = policy.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if render and t%3==0: env.render()
        if done: break
    return total_rew

env = None
def noisy_evaluation(theta):
    policy = make_policy(theta)
    rew = do_episode(policy, env, num_steps)
    return rew


# Task settings:
env = gym.make('Pendulum-v0') # Change as needed
num_steps = 500 # maximum length of episode
# Alg settings:
n_iter = 10000 # number of iterations of CEM
batch_size = 50 # number of samples per batch
elite_frac = 0.2 # fraction of samples used as elite set
hidden_dim = 1000

def make_policy(theta):
    if isinstance(env.action_space, Discrete):
        return DeterministicDiscreteActionNNPolicy(theta, hidden_dim,
            env.observation_space, env.action_space)
        #return DeterministicDiscreteActionLinearPolicy(theta,
        #    env.observation_space, env.action_space)
    elif isinstance(env.action_space, Box):
        return DeterministicContinuousActionNNPolicy(theta, hidden_dim,
            env.observation_space, env.action_space)
        return DeterministicContinuousActionLinearPolicy(theta,
            env.observation_space, env.action_space)
    else:
        raise NotImplementedError


print 'Spaces', env.observation_space, env.action_space
if isinstance(env.action_space, Discrete):
    dim_theta = ((env.observation_space.shape[0]+1) * hidden_dim
        + (hidden_dim + 1) * env.action_space.n)
    #dim_theta = (env.observation_space.shape[0]+1) * env.action_space.n
elif isinstance(env.action_space, Box):
    dim_theta = ((env.observation_space.shape[0]+1) * hidden_dim
        + (hidden_dim + 1) * env.action_space.shape[0])
    #dim_theta = (env.observation_space.shape[0]+1) * env.action_space.shape[0]
else:
    raise NotImplementedError

# Initialize mean and standard deviation
theta_mean = np.zeros(dim_theta)
theta_std = np.ones(dim_theta)

# Now, for the algorithm
for iteration in xrange(n_iter):
    # Sample parameter vectors
    thetas = np.array([theta_mean + dth for dth
        in  theta_std[None,:]*np.random.randn(batch_size, theta_mean.size)])
    #thetas = np.random.multivariate_normal(
    #    theta_mean, np.diag(np.square(theta_std)), batch_size)
    rewards = [noisy_evaluation(theta) for theta in thetas]
    # Get elite parameters
    n_elite = int(batch_size * elite_frac)
    elite_inds = np.argsort(rewards)[batch_size - n_elite:batch_size]
    elite_thetas = np.array([thetas[i] for i in elite_inds])
    print elite_thetas.shape
    # Update theta_mean, theta_std
    theta_mean = np.mean(elite_thetas, axis=0)
    theta_std = np.std(elite_thetas, axis=0)
    print "iteration %i. mean f: %8.3g. max f: %8.3g"%(iteration, np.mean(rewards), np.max(rewards))
    render = (iteration % 10) == 0
    do_episode(make_policy(theta_mean), env, num_steps, render=render)
