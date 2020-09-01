import os
import numpy as np
import gym
import argparse
from qnetwork import Qnet
from atari_wrappers import wrap_atari, FrameBuffer
import torch
from tcp import trajectory_central_planning
from value_memory import ValueBuffer
from replay_buffer import ReplayBuffer
import logging

env = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config(argparse.Namespace):
    def __init__(self):
        super().__init__()
        self.envname = "BreakoutNoFrameskip-v4"
        self.embedding_size = 256
        self.lr = 1e-4
        self.tcp_frequency = 20
        self.path_length = 50
        self.gamma = 0.99
        self.lambd = 0.5
        self.t_max = 100000
        self.t_update = 50
        self.min_eps = 0.1
        self.max_eps = 1.0
        self.eps_decay = 2e-6
        self.replay_buffer_size = 400000
        self.value_buffer_size  = 2000
        self.n_episodes = 80000
        self.batch_size = 48
        self.num_tcp_paths = 10
        self.n_neigbors_value_buffer = 5
        self.save_freq = 1000
        self.test_freq = 1000
        self.save_dir = 'results'
        
        
config = Config()

def parse_arguments():
    parser = argparse.ArgumentParser(prog="python experiment.py")
    parser.add_argument('--envname', default=config.envname, help='Name of gym atari environment')
    parser.add_argument('--embedding_size', default=config.embedding_size, type=int, help='Size of the embedding vector')
    parser.add_argument('--lr', default=config.lr, type=float, help='Learning rate for Adam optimizer')
    parser.add_argument('--tcp_frequency', default=config.tcp_frequency, type=int, help='frequency of trajectory central planning per training episodes')
    parser.add_argument('--path_length', default=config.path_length, type=int, help='length of trajectory for trajectory central plannig')
    parser.add_argument('--gamma', default=config.gamma, type=float, help='discount factor of rewards')
    parser.add_argument('--lambd', default=config.lambd, type=float, help='weight of non-parametric action-value')
    parser.add_argument('--t_max', default=confg.t_max, type=int, help='maximum number of steps per episode')
    parser.add_argument('--t_update', default=config.t_update, type=int, help='target network update frequency')
    parser.add_argument('--min_eps', default=config.min_eps, type=float, help='floor of exploration rate')
    parser.add_argument('--max_eps', default=config.max_eps, type=float, help='cap of exploration rate')
    parser.add_argument('--eps_decay', default=config.eps_decay, type=float,  help='epsilon decay rate')
    parser.add_argument('--replay_buffer_size', default=config.replay_buffer_size, type=int, help='replay buffer size')
    parser.add_argument('--value_buffer_size', default=config.value_buffer_size, type=int, help='value buffer size')
    parser.add_argument('--n_episodes', default=config.n_episodes, type=int, help='number of episodes')
    parser.add_argument('--batch_size', default=config.batch_size, type=int, help='batch size for training')
    parser.add_argument('--num_tcp_paths', default=config.num_tcp_paths, type=int, help='number of trajectories for trajectory central planning')
    parser.add_argument('--n_neigbors_value_buffer', default=config.n_neigbors_value_buffer, type=int, help='number of nearest neigbors to average to obtain non-parameteric action-value')

    parser.add_argument('--test_freq', default=config.save_freq, type=int, help='frequency, in episodes, of test runs (eps=0)')
    parser.add_argument('--save_freq', default=config.save_freq, type=int, help='frequency, in episodes, of model saving')
    parser.add_argument('--save_dir', default=config.save_dir, help='directory to store results')
    parser.parse_args(namespace=config)

def make_env(clip_rewards=True, seed=None):
    env = gym.make(config.envname)
    if seed is not None:
        env.seed(seed)
    env = wrap_atari(env, clip_rewards)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    return env










def main():
    global env
    env = make_env()
    n_actions = env.action_space.n
    qnet = Qnet(n_actions, embedding_size=config.embedding_size).to(device)
    target_net = Qnet(n_actions, embedding_size=config.embedding_size).to(device)
    target_net.load_state_dict(qnet.state_dict())
    target_net.eval()
    os.makedirs(config.save_dir, exist_ok=True)
    replay = ReplayBuffer(config.replay_buffer_size, config.embedding_size, config.path_length)
    value_buffer= ValueBuffer(config.value_buffer_capacity)
    






    print(config)






    env.reset()
    


if __name__ == "__main__":
    parse_arguments()
    #parser = argparse.ArgumentParser(prog="python experiment.py")
    #parser.add_argument('--envname', help='Name of gym environment')
    #args = parser.parse_args()
    #print (vars(args))
    #print(args.envname)
    main()

