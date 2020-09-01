import argparse

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

