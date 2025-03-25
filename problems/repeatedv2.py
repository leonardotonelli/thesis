import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class BinaryPerceptronRepeated:
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
        self.weights = np.random.choice([-1,1], size=self.n)


    def compute_cost(self, gamma, distance):
        '''
        Define the cost function and computation
        '''
        self.frwd = self.forward()
        wrong_bool = (self.frwd * self.targets) < 0
        cost = wrong_bool.sum()
        return cost + gamma*distance

  
    def compute_delta_cost(self, action, gamma, reference_weights):
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
        new_errors = (new_pred * self.targets) < 0
        new_cost = np.sum(new_errors)
        # compute delta
        delta = new_cost - current_cost
        
        final_delta = delta + gamma * (2*self.weights[action]*reference_weights[action])


        # VERIFICATION CORRECTNESS
        # distance1 = self.compute_distance(reference_weights)

        # # current pred
        # starting_cost = self.compute_cost(gamma, distance1)
        # temp_problem = self.copy()
        # temp_problem.accept_action(action)

        # distance2 = temp_problem.compute_distance(reference_weights)
        # new_cost = temp_problem.compute_cost(gamma, distance2)
        # delta_real = (new_cost - starting_cost)

        # assert delta_real == final_delta
        
        return final_delta


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

    def compute_distance(self, reference_weights):
        ''' 
        Function that computes the distance between the given replica to the reference
        '''
        d = np.sum((self.weights - reference_weights)**2)/2
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
