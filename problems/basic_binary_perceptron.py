import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class BasicBinaryPerceptron:
    def __init__(self, n: int , P: int, seed: int):
        '''
        Initializations
        '''
        self.n = n
        self.P = P
        self.alpha = P/n
        self.seed = seed
        self.targets = np.random.choice([-1,1], size=P)
        self.weights = np.zeros(n)


    def init_config(self):
        '''
        Initial configuration of the objective matrix
        '''
        self.X = np.random.normal(loc = 0, scale = 1, size = (self.P, self.n))
        self.weights = np.random.choice([True, False], size=self.n)


    def compute_cost(self):
        '''
        Define the cost function and computation
        '''
        self.frwd = self.forward()
        wrong_bool = (self.frwd * self.targets) < 0
        cost = wrong_bool.sum()
        return cost


    def compute_delta_cost(self, action):
        '''
        Compute delta cost of a given action
        '''
        starting_cost = self.compute_cost()
        temp_problem = self.copy()
        temp_problem.accept_action(action)
        new_cost = temp_problem.compute_cost()
        delta = (new_cost - starting_cost)

        return delta



    def accept_action(self, action):
        '''
        Update the internal states given the taken action
        '''
        self.weights[action] = not self.weights[action]


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
        Display the internal state of the problem
        '''
    def forward(self):
        '''
        Function that outputs the prediction in the current state
        '''
        intermediate = self.X @ self.weights
        return intermediate
