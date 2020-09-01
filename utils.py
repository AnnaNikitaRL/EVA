import gym
import numpy as np
from atari_wrappers import wrap_atari, FrameBuffer

def epsilon(t, config):
    '''
    Calculates exploration rate (epsilon) as a function of global step.
    Epsilon decays exponentially with initial value config.max_eps, 
    final value config.min_eps, and decay rate config.eps_decay
    '''
    return config.min_eps + (config.max_eps - config.min_eps)* np.exp(- (config.eps_decay) * t )


def make_env(config, clip_rewards=True, seed=None):
    '''
    Creates gym environment wrapped in some atari wrappers.
    '''
    env = gym.make(config.envname)
    if seed is not None:
        env.seed(seed)
    env = wrap_atari(env, clip_rewards)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    return env

