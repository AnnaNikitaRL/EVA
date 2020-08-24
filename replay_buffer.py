import numpy as np
import random


class ReplayMemory(object):

    def __init__(self, capacity, flann, emb_dimension=DIMENSION, path_length=PATH_LENGTH, rebuild_freq=REBUILD_FREQ):
        self.capacity = capacity
        self.memory = []
        self.embed = np.zeros((capacity, emb_dimension),dtype='float32')
        self.position = 0
        self.engine = flann
        self.path_length = path_length
        self.rebuild_freq=REBUILD_FREQ
        self.rebuild_counter=0
        self.last_rebuild_position = 0

    #use next_state = None if terminal
    def push(self, state, action, reward, next_state, embed):
        """Saves a transition."""
        self.rebuild_counter+=1
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        else:
            if self.rebuild_counter >= REBUILD_FREQ:
                self.rebuild_counter = 0
                self.engine.build_index(self.embed)
                self.last_rebuild_position = self.position

        self.memory[self.position] = Transition(state, action, reward, next_state)
        self.embed[self.position] = embed


        self.position = (self.position + 1) % self.capacity


    def is_idx_permitted(self, idx):
        if (self.position >= self.last_rebuild_position):
            if(idx <=self.position and idx >= self.last_rebuild_position):
                return False
        else:
            if (idx <=self.position or idx >= self.last_rebuild_position):
                return False

        if (self.position >= self.path_length):
            if(idx <=self.position and idx >= self.position - self.path_length):
                return False
        else:
            if (idx <=self.position or idx >= (self.position - self.path_length) % self.capacity):
                return False
        return True

    def neighbours(self, query, num_neighbors=M, return_embed=False):
        idxs, dist = self.engine.nn_index(query[np.newaxis], num_neighbors=num_neighbors+50)
        idxs, dist = idxs[0], dist[0]


        idxs = [idx for idx in idxs if self.is_idx_permitted(idx)][:num_neighbors]

        if return_embed:
            return idxs, [self.embed[idx] for idx in idxs]
        else:
            return idxs

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


    def __len__(self):
        return len(self.memory)


