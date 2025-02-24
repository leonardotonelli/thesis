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

        common_addend = self.X @ self.weights - (self.X[:, action] * self.weights[action])


        # Current cost
        starting_addend = self.X[:, action] * self.weights[action]
        starting_pred = (common_addend + starting_addend) 
        starting_wrong_bool = (starting_pred * self.targets) < 0
        starting_cost = starting_wrong_bool.sum()

        # Copying and making the change
        temp_problem = self.copy()
        temp_problem.accept_action(action)

        # New Cost
        new_addend = temp_problem.X[:, action] * temp_problem.weights[action]
        new_pred = (common_addend + new_addend)
        new_wrong_bool = (new_pred * temp_problem.targets) < 0
        new_cost = new_wrong_bool.sum()

        delta_new = (new_cost - starting_cost)

        print(f"New= {delta_new}, Correct = {delta}")
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
        Display the current state
        '''
        
    
    def forward(self):
        '''
        Function that outputs the prediction in the current state
        '''
        intermediate = self.X @ self.weights
        return intermediate
