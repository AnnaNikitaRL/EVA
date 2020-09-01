import numpy as np
import pyflann
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class ReplayBuffer(object):
    """ Cyclic buffer that stores transitions and embeddings observed. 
    Additionally supports  functionality to find and return fixed number of neigbors for each index. 
    Search for nearest neigbors is done with help of FLANN library for fast approximate nearest neighbors search """
    
    def __init__(self, capacity, emb_dimension, path_length, rebuild_freq=500):
        self.capacity              = capacity
        self.memory                = []
        self.embeddings            = np.zeros((capacity, emb_dimension), dtype='float32')
        self.position              = 0
        self.engine                = pyflann.FLANN()
        self.path_length           = path_length
        self.rebuild_freq          = rebuild_freq
        self.rebuild_counter       = 0
        self.last_rebuild_position = 0

    def push(self, state, action, reward, next_state, embedding):
        """ Saves a transition and embedding """
        
        self.rebuild_counter+=1
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        else:
            if self.rebuild_counter >= self.rebuild_freq:
                self.rebuild_counter = 0
                self.engine.build_index(self.embedding)
                self.last_rebuild_position = self.position

        self.memory[self.position]      = Transition(state, action, reward, next_state)
        self.embeddings[self.position]  = embedding
        self.position                   = (self.position + 1) % self.capacity


    def _is_idx_permitted(self, idx):
        """ Checks if index is permitted """
           
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

    def neighbors(self, query, num_neighbors=M, return_embeddings=False):
        """ For fixed index calculates M number of neighbors and returns their indecies and embeddings if requested """
         
        idxs, dist = self.engine.nn_index(query[np.newaxis], num_neighbors=num_neighbors+50)
        idxs, dist = idxs[0], dist[0]
        
        idxs       = [idx for idx in idxs if self._is_idx_permitted(idx)][:num_neighbors]
        if return_embeddings:
            return idxs, [self.embedding[idx] for idx in idxs]
        else:
            return idxs

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

