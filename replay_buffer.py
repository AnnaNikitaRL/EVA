import random
from collections import namedtuple
import numpy as np
import pyflann

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class ReplayBuffer():
    """ 
    Cyclic buffer that stores transitions and embeddings observed. 
    Additionally supports  functionality to find and return fixed number of neigbors for each index. 
    Search for nearest neigbors is done with help of FLANN library 
    for fast approximate nearest neighbors search 
    """
    
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
        """ Saves a transition (state, action, reward, next_state) and
            an embedding in cyclic arrays """
        
        self.rebuild_counter+=1
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        else:
            if self.rebuild_counter >= self.rebuild_freq:
                self.rebuild_counter = 0
                self.engine.build_index(self.embeddings)
                self.last_rebuild_position = self.position

        self.memory[self.position]      = Transition(state, action, reward, next_state)
        self.embeddings[self.position]  = embedding
        self.position                   = (self.position + 1) % self.capacity

    def _is_idx_permitted(self, idx):
        """ 
        Checks if index return by engine.nn_index is permitted. 
        It's not if either:
        1. The corresponding entry in self.memory was rewritten recently,
           but the engine wasn't rebuilt since than, so it found the idx
           based on an outdated embedding.
        2. We cannot build a trajectory starting from the idx, because we don't
           have enough steps of experience yet, for the case it is very recent idx,
           yet the engine was already rebuilt with its embedding.
        """

        # Not permitted if we have already rewritten an old entry in experience replay
        # but haven't rebuilt the nn_index yet, so it still contains an old embedding.
        if self.position >= self.last_rebuild_position:
            if (idx <=self.position) and (idx >= self.last_rebuild_position):
                return False

        # Same as above, but for the case that we cycled through the memory recently
        # so the current position is in the beginning of the array,
        # while last_rebuild_position points to somewhere near its end.
        else:
            if (idx <=self.position) or (idx >= self.last_rebuild_position):
                return False
        
        # Not permitted if we cannot build full trajectory from this point,
        # because it is the most recent experience and we haven't played it for
        # self.path_length ahead yet.
        if self.position >= self.path_length:
            if (idx <= self.position) and (idx >= (self.position - self.path_length)):
                return False

        # Same as above, but for the case that we cycled through the memory recently
        else:
            if (idx <=self.position or idx >= (self.position - self.path_length) % self.capacity):
                return False
        return True

    def neighbors(self, query, num_neighbors, return_embeddings=False):
        """ 
        Finds num_neighbors embedings that are nearest neighbors to query
        and returns their indices in self.memory list 
        along with the list of embeddings if requested. 
        """
        
        # Return more neighbors than requested from FLANN engine, as we will filter out some idxs
        # FLANN's KD-tree implementation doesn't increase much with increase in num_neighbors
        idxs, dist = self.engine.nn_index(query[np.newaxis], num_neighbors=num_neighbors+50)
        idxs, dist = idxs[0], dist[0]
        
        # Filter out indices from which we cannot build the corresponding trajectory
        # And return the nearest num_neighbors from the rest
        idxs       = [idx for idx in idxs if self._is_idx_permitted(idx)][:num_neighbors]
        if return_embeddings:
            return idxs, [self.embeddings[idx] for idx in idxs]
        return idxs

    def sample(self, batch_size):
        """ Sample batch of transitions (state, action, reward, next_state) """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """ Number of entries in memory. Cannot be greater than capacity """
        return len(self.memory)

