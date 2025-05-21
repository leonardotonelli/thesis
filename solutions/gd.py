import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class BinaryPerceptronGD:
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


    def compute_cost(self):
        '''
        Define the cost function and computation
        '''
        self.pred = self.forward()
        wrong_bool = (self.pred * self.targets) < 0
        cost = wrong_bool.sum()
        return cost

    def calculate_grad(self, index):
        '''
        compute gradient from loss computed for just the batch. It returns an array of gradients for each weight
        '''
        grad = np.zeros(self.n)
        for i, weight in enumerate(self.weights):
            pred = self.pred[index]
            new_pred = pred - 2*self.X[index, i]*weight
            current_loss = (pred * self.targets[index]) < 0
            new_loss = (new_pred * self.targets[index]) < 0
            grad[i] = new_loss - current_loss

        self.grad = grad

    def step(self, lr):
        '''
        make a continuous step towards the gradient direction
        '''
        self.weights = self.weights - lr * self.grad

    def discretize(self):
        self.weights = np.sign(self.weights)
        self.weights[self.weights == 0] = 1

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
        new_errors = (new_pred * self.targets) < 0
        new_cost = np.sum(new_errors)
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

    def shuffle(self):
        # Shuffle indices
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        # Apply permutation to both X and targets
        self.X = self.X[indices]
        self.targets = self.targets[indices]

# batch_size=1
def gd(probl, lr: float, max_epochs: int, batch_size: int):

    probl.init_config()
    stop = False
    epochs = 0

    while stop is not True:

        for i in range(probl.X.shape[0]):
            # calculate the gradients based on the discrete internal weights
            probl.calculate_grad(i)

            # make updates to the weights of the replica
            probl.step(lr)

            # discretize the new weights
            probl.discretize()

        epochs += 1
            
        # stopping criterion, either solved the problem or max amount of epochs reached
        if probl.cost == 0:
            print("Problem solved.")
            stop = True
        elif epochs>max_epochs:
            print(f"Maximum amount of epochs reached. Best cost reached= {probl.best_cost()}")
            stop = True

    return probl


# # generalized for all batch sizes
# def gd(probl, lr: float, max_epochs: int, batch_size: int):

#     probl.init_config()
#     stop = False
#     epochs = 0

#     while stop is not True:
#         probl.shuffle() 

#         for i in range(0, probl.X.shape[0], batch_size):
#             # calculate the gradients based on the discrete internal weights
#             probl.calculate_grad(batch_size)

#             # make updates to the weights of the replica
#             probl.step(lr)

#             # discretize the new weights
#             probl.discretize()

#         epochs += 1
            
#         # stopping criterion, either solved the problem or max amount of epochs reached
#         if probl.cost == 0:
#             print("Problem solved.")
#             stop = True
#         elif epochs>max_epochs:
#             print(f"Maximum amount of epochs reached. Best cost reached= {probl.best_cost()}")
#             stop = True

#     return probl