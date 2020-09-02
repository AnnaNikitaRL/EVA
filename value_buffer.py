import numpy as np
from sklearn import neighbors
import torch

class ValueBuffer():
    """ Cyclic buffer which stores the value estimates resulting from planning process """

    def __init__(self, capacity):
        self.capacity    = capacity
        self.values      = []
        self.position    = 0
        self.embeddings  = []
        self.tree        = None

    def push(self, embedding, value):
        """ Saves value and embedding """
        if len(self.values) < self.capacity:
            self.values.append(None)
            self.embeddings.append(None)

        self.values[self.position]      = value
        self.embeddings[self.position]  = embedding
        self.position = (self.position + 1) % self.capacity

    def build_tree(self):
        """ Build tree of neighbors on space of embeddings using default Euclidean metric """
        self.tree = neighbors.KDTree(np.array(self.embeddings))
    
    def nn_qnp_mean(self, query, n_neighbors=5):
        """ Calculates and returns action-value averaged 
            across fixed number of neighbors """
        return torch.mean( torch.stack([self.values[q] for q in self.tree.query(query[np.newaxis], k=n_neighbors)[1][0]]), axis=0)

    def __len__(self):
        """ Returns size of stored values """
        return len(self.values)
