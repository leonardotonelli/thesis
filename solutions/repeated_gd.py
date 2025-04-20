import numpy as np
from copy import deepcopy

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


class RepeatedGD:
    def __init__(self):
        ...

    def init_config(self):
        ''' Method that initializes the replicas and the problems'''

    def compute_reference(self):
        ''' Method that computes the reference replica'''

    def random_replica(self):
        ''' Method that returns a random replica index to propose actions'''

    def calculate_grad(self):
        ''' Method that calculate the gradient needed for the update'''

    def step(self):
        ''' Method that modifies the specific replica weights according to the update rule'''

    def discretize(self):
        ''' Method that discretize the weights after the step'''

    def epoch_passed(self):
        ''' Method that check whether a full epoch as passed, if yes it reinitializes the record'''

    def best_cost(self):
        ''' method to get the best cost across all the replicas results'''

    def get_best_replica(self):
        ''' method to get the best performing replica and return it'''



def replicated_gd(probl, lr: float, lr_at: float, max_epochs: int, batch_size: int):

    probl.init_config()
    stop = False
    epochs = 0

    while stop is not True:
        
        # compute the reference weights (just as the sum)
        probl.compute_reference()

        # get a random replica index
        replica_index = probl.random_replica()

        # calculate the gradients based on the discrete internal weights
        probl.calculate_grad(replica_index, batch_size)

        # make updates to the weights of the replica
        probl.step(replica_index)

        # discretize the new weights
        probl.discretize(replica_index)

        # check whether an epoch as passed
        if probl.epoch_passed():
            epochs += 1
        
        # stopping criterion, either solved the problem or max amount of epochs reached
        if probl.is_solved(replica_index):
            print("Problem solved.")
            stop = True
        elif epochs>max_epochs:
            print(f"Maximum amount of epochs reached. Best cost reached= {probl.best_cost()}")
            stop = True

    # get the best replica to return 
    best_replica = probl.get_best_replica()

    return best_replica