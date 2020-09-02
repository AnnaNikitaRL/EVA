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
from train_test import train
import utils

env = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config()
logging.basicConfig(level=0)

def main():
    global env
    logging.info("Started...")
    env = utils.make_env(config)
    n_actions = env.action_space.n
    qnet = Qnet(n_actions, embedding_size=config.embedding_size).to(device)
    target_net = Qnet(n_actions, embedding_size=config.embedding_size).to(device)
    target_net.load_state_dict(qnet.state_dict())
    target_net.eval()
    replay_buffer = ReplayBuffer(config.replay_buffer_size, config.embedding_size, config.path_length)
    optimizer = torch.optim.Adam(qnet.parameters(), lr=config.lr)
    value_buffer = ValueBuffer(config.value_buffer_size)
    train(env, qnet, target_net, optimizer, replay_buffer, value_buffer, config, device)


if __name__ == "__main__":
    parse_arguments(config)
    main()

