import os
import numpy as np
import gym
import argparse
from qnetwork import Qnet
import torch
from tcp import trajectory_central_planning
from value_buffer import ValueBuffer
from replay_buffer import ReplayBuffer
import logging
from config import Config, parse_arguments
from train_test import train, test
import utils

env = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config()











def main():
    global env
    env = utils.make_env(config)
    n_actions = env.action_space.n
    qnet = Qnet(n_actions, embedding_size=config.embedding_size).to(device)
    target_net = Qnet(n_actions, embedding_size=config.embedding_size).to(device)
    target_net.load_state_dict(qnet.state_dict())
    target_net.eval()
    os.makedirs(config.save_dir, exist_ok=True)
    replay_buffer = ReplayBuffer(config.replay_buffer_size, config.embedding_size, config.path_length)
    value_buffer = ValueBuffer(config.value_buffer_capacity)
    train()


    






    print(config)






    env.reset()
    


if __name__ == "__main__":
    parse_arguments(config)
    #parser = argparse.ArgumentParser(prog="python experiment.py")
    #parser.add_argument('--envname', help='Name of gym environment')
    #args = parser.parse_args()
    #print (vars(args))
    #print(args.envname)
    main()

