import gym
import argparse
from atari_wrappers import wrap_atari, FrameBuffer

class Config(argparse.Namespace):
    def __init__(self):
        super().__init__()
        self.envname="BreakoutNoFrameskip-v4"

config = Config()
env = None

def parse_arguments():
    parser = argparse.ArgumentParser(prog="python experiment.py")
    parser.add_argument('--envname', default=config.envname, help='Name of gym environment')
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
    env.reset()
    


if __name__ == "__main__":
    parse_arguments()
    #parser = argparse.ArgumentParser(prog="python experiment.py")
    #parser.add_argument('--envname', help='Name of gym environment')
    #args = parser.parse_args()
    #print (vars(args))
    #print(args.envname)
    main()

