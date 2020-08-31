import numpy as np
from sklearn import neighbors

class ValueMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.idx_to_keys = []
        self.tree = None
        
    def push(self, embed, value):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.idx_to_keys.append(None)
                
        self.memory[self.position] = value
        self.idx_to_keys[self.position] = embed
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def build_tree(self):
        self.tree = neighbors.KDTree(np.array(self.idx_to_keys))
    
    def nn_Q(self, query, num_neighbors=5):
        return torch.mean( torch.stack([self.memory[q] for q in self.tree.query(query[np.newaxis], k=num_neighbors)[1][0]]), axis=0)

    def __len__(self):
        return len(self.memory)
