import gym
import argparse
from qnetwork import Qnet
from atari_wrappers import wrap_atari, FrameBuffer
import torch

env = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config(argparse.Namespace):
    def __init__(self):
        super().__init__()
        self.envname = "BreakoutNoFrameskip-v4"
        self.embedding_size = 256
        self.lr = 1e-4
        
config = Config()

def parse_arguments():
    parser = argparse.ArgumentParser(prog="python experiment.py")
    parser.add_argument('--envname', default=config.envname, help='Name of gym atari environment')
    parser.add_argument('--embedding_size', default=config.embedding_size, type=int, help='Size of the embedding vector')
    parser.add_argument('--lr', default=config.lr, type=float, help='Learning rate for Adam optimizer')
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

