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
        self.X = np.random.choice([-1,1], size = (self.P, self.n)) # is it correct?
        self.weights = np.random.choice([-1,1], size=self.n)
        self.pred = self.forward()


    def compute_cost(self):
        '''
        Define the cost function and computation
        '''
        self.frwd = self.forward()
        x = -(self.frwd * self.targets)
        theta = x>0
        cost = (x+1)/2 * theta
        # print(f"X = {self.X}")
        # print(f"weights = {self.weights}")
        # print(f"targets = {self.targets}")
        # print(f"cost = {cost}")

        return cost.sum()

    
    def min_switches(self, pred):
        '''
        Compute minimum switches cost giving a prediction and the self targets
        '''
        x = -(pred * self.targets)
        theta = x>0
        cost = ( (x+1)/2 ) * theta
        return np.sum(cost)


    def compute_delta_cost(self, action):
        '''
        Compute delta cost of a given action efficiently
        '''
        
        # current_predictions = self.pred * self.targets
        
        # mask1 = (self.weights[action] == self.X[:, action]).flatten() # if contribution to prediction is positive or negative
        # mask2 = current_predictions > 2 # if the current prediction is confidently correct or wrong
        # mask3 = current_predictions > 0 # if current prediction is correct overall

        # condition1 = mask1*mask2    # if contribution positive and prediction confidently correct
        # condition2 = mask1*~mask2   # if contribution positive and prediction either wrong (<1) or not confidently correct (=1, or =2)
        # condition3 = ~mask1*~mask3  # if contribution negative and prediction wrong
        # condition4 = ~mask1*mask3   # if contribution negative and prediction correct
        
        # # initialize the deltas vector
        # deltas = np.zeros(self.P)  

        # deltas[condition1] = 0      # given condition, changing the weight the prediction stays correct (0 cost, 0 delta)
        # deltas[condition2] = 1      # given condition, changing the weight will increase the cost by 1, additional weight to switch
        # deltas[condition3] = -1     # given condition, changing the weight will improve the prediction, then decrease the cost by 1
        # deltas[condition4] = 0      # given condition, changing the weight will make no change to correct prediction and then no delta cost

        # delta1 = np.sum(deltas)


        # MORE CLEAR PROPOSAL

        current_predictions = self.pred * self.targets
        
        delta = (-2 * self.X[:, action] * self.weights[action]).flatten()
        delta_mask = delta > 0 # if contribution to prediction is positive or negative
        pred_mask1 = current_predictions <= 0
        pred_mask2 = (current_predictions == 1) | (current_predictions == 2)
        pred_mask3 = current_predictions > 2

        condition1 = delta_mask*pred_mask1    # if contribution positive and prediction confidently correct
        condition2 = delta_mask*pred_mask2*pred_mask3   # if contribution positive and prediction either wrong (<1) or not confidently correct (=1, or =2)
        condition3 = ~delta_mask*pred_mask1*pred_mask2   # if contribution negative and prediction correct
        condition4 = ~delta_mask*pred_mask3

        # initialize the deltas vector
        deltas = np.zeros(self.P)  

        deltas[condition1] = -1      # given condition, changing the weight the prediction stays correct (0 cost, 0 delta)
        deltas[condition2] = 0      # given condition, changing the weight will increase the cost by 1, additional weight to switch
        deltas[condition3] = +1     # given condition, changing the weight will improve the prediction, then decrease the cost by 1
        deltas[condition4] = 0      # given condition, changing the weight will make no change to correct prediction and then no delta cost


        delta2 = np.sum(deltas)


        # VERIFICATION CORRECTNESS
        temp_problem = self.copy()
        temp_problem.accept_action(action)
        verification_cost = temp_problem.compute_cost()
        original_cost = self.compute_cost()
        verification_delta = verification_cost - original_cost
        # assert delta1 == verification_delta
        # assert delta2 == verification_delta
        print(f"delta2 = {delta2}, real = {verification_delta}")
        return delta2


    def accept_action(self, action):
        '''
        Update the internal states given the taken action
        '''
        self.pred = self.pred - (2 * self.X[:, action] * self.weights[action]).flatten()
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
