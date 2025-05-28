import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class BinaryPerceptron:
    def __init__(self, n: int , P: int, seed: int):
        '''
        Initializations
        '''
        self.n = n
        self.P = P
        self.alpha = P/n
        self.seed = seed
        self.weights = np.zeros(n)


    def init_config(self):
        '''
        Initial configuration of the objective matrix
        '''
        self.targets = np.random.choice([-1,1], size=self.P)
        self.X = np.random.choice([-1,1], size = (self.P, self.n))
        self.weights = np.random.choice([-1,1], size=self.n)


    def compute_cost(self):
        '''
        Define the cost function and computation
        '''
        self.pred = self.forward()
        cost = np.sum( (self.pred >= 0) == (self.targets == 1) )
        return cost

  
    def compute_delta_cost(self, action):
        '''
        Compute delta cost of a given action efficiently
        '''
        # current pred
        current_pred = self.pred
        # delta predictions mathematically correct
        delta_pred = -2 * self.X[:, action] * self.weights[action]
        # derive new pred from the delta
        new_pred = current_pred + delta_pred.flatten()
        #current cost
        current_errors = (current_pred * self.targets) < 0
        current_cost = np.sum(current_errors)
        # new cost
        new_cost = np.sum( (new_pred >= 0) == (self.targets == 1) )
        # compute delta
        delta = new_cost - current_cost
        
        # VERIFICATION CORRECTNESS
        # temp_problem = self.copy()
        # temp_problem.accept_action(action)
        # verification_cost = temp_problem.compute_cost()
        # original_cost = self.compute_cost()
        # verification_delta = verification_cost - original_cost
        # assert delta == verification_delta
        
        return delta


    def accept_action(self, action):
        '''
        Update the internal states given the taken action
        '''
        delta_pred = (-2 * self.X[:, action] * self.weights[action]).flatten()
        self.pred = self.pred + delta_pred
        self.weights[action] = - self.weights[action]


    def propose_action(self):
        '''
        Propose a move based on some criteria
        '''
        index = np.random.choice(range(self.n), size=1)
        return index


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
