import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class BinaryPerceptronRepeated:
    def __init__(self, n: int , P: int, seed = None):
        '''
        Initializations
        '''
        self.n = n
        self.P = P
        self.alpha = P/n
        self.seed = seed
        self.weights = np.zeros(n)
        self.iterations_to_solution = np.nan
        self.time_to_solution = np.nan


    def init_config(self, seed):
        '''
        Initial configuration of the objective matrix
        '''
        np.random.seed(seed)
        self.targets = np.random.choice([-1,1], size=self.P)
        self.X = np.random.choice([-1,1], size = (self.P, self.n))
        self.weights = np.random.choice([-1,1], size=self.n)


    def compute_cost(self):
        '''
        Define the cost function and computation
        '''
        self.pred = self.forward()
        cost = np.sum( (self.pred >= 0) == (self.targets == 1) )
        self.cost = cost
        new_c = cost
        return new_c

  
    def compute_delta_cost(self, action):
        '''
        Compute delta cost of a given action efficiently
        '''
        # current pred
        current_pred = self.pred
        current_cost = self.cost

        # delta predictions mathematically correct
        delta_pred = -2 * self.X[:, action] * self.weights[action]
        # derive new pred from the delta
        new_pred = current_pred + delta_pred.flatten()
        new_cost = np.sum( (new_pred >= 0) == (self.targets == 1) )
        
        # compute delta
        delta = new_cost - current_cost

        return delta


    def accept_action(self, action):
        '''
        Update the internal states given the taken action
        '''
        # update predictions
        delta_pred = (-2 * self.X[:, action] * self.weights[action]).flatten()
        self.pred = self.pred + delta_pred

        # update weights
        self.weights[action] = - self.weights[action]

        # update cost
        # new_errors = (self.pred * self.targets) < 0
        # self.cost = np.sum(new_errors)
        self.cost = np.sum( (self.pred >= 0) == (self.targets == 1) )


    def propose_action(self):
        '''
        Propose a move based on some criteria
        '''
        if self.seed is not None:
            np.random.seed(self.seed)
        index = np.random.choice(range(self.n), size=1)
        return index

    def compute_distance(self, reference_weights):
        ''' 
        Function that computes the distance between the given replica to the reference
        '''
        d = np.sum((self.weights - reference_weights)**2)/2 ## 
        return d

    def copy(self):
        '''
        Copy the whole problem
        '''
        return deepcopy(self)


    def display(self):
        '''
        Display the current state
        '''
    
    def forward(self):
        '''
        Function that outputs the prediction in the current state
        '''
        intermediate = self.X @ self.weights
        return intermediate
