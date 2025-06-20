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
        self.weights_continuous = np.zeros(n)


    def init_config(self):
        '''
        Initial configuration of the objective matrix
        '''
        self.X = np.random.choice([-1,1], size = (self.P, self.n))
        self.weights = np.random.choice([-1,1], size=self.n)
        self.weights_continuous = self.weights.copy()
        self.pred = self.forward()


    def compute_cost(self):
        '''
        Define the cost function and computation
        '''
        self.pred = self.forward()
        cost = np.sum( (self.pred >= 0) == (self.targets == -1) )
        return cost

    def calculate_grad(self, index):
        '''
        compute gradient from loss computed for just the batch. It returns an array of gradients for each weight
        '''
        grad = np.zeros(self.n)
        for i, weight in enumerate(self.weights):
            pred = self.pred[index] 
            new_pred = pred - 2*self.X[index, i]*weight 
            current_loss = abs(pred - self.targets[index])
            new_loss = abs(new_pred - self.targets[index])
            grad[i] = (new_loss - current_loss)/(-2*weight) #TODO TO CHANGE

        self.grad = grad

    def calculate_batch_grad(self, final, batch_size):
        '''
        compute gradient from loss computed for just the batch. It returns an array of gradients for each weight
        '''
        # print(f"X = {self.X}")
        # print(f"weights = {self.weights}")
        # print(f"targets = {self.targets}")
        batch_grads = np.zeros((batch_size, self.n))
        for index in range(final - batch_size, final):
            for i, weight in enumerate(self.weights):
                pred = self.pred[index]
                new_pred = pred - 2*self.X[index, i]*weight
                current_loss = 0 if (pred>=0)*1 ==  self.targets[index] else abs(pred)
                new_loss = 0 if (new_pred>=0)*1 ==  self.targets[index] else abs(new_pred)
                batch_grads[index-final+batch_size,i] = (new_loss - current_loss)/(- 2*weight) # TODO DA CAMBIARE TENENDO CONTO CAMBIO PREDIZIONI
                # print(f"Batch Gradients: ")
                # print(batch_grads)

        self.grad = np.mean(batch_grads, axis=0)
        # print("Grad:")
        # print(self.grad)

    def step(self, lr):
        '''
        make a continuous step towards the gradient direction
        '''
        # print(f"Step...") 
        # print(f"continuous before= {self.weights_continuous}")
        self.weights_continuous = self.weights_continuous - lr * self.grad
        # print(f"continuous after= {self.weights_continuous}")

    def discretize(self):
        # print(f"Discretize...")
        # print(f"current weights: {self.weights}")
        # print(f"current continuous weights: {self.weights_continuous}")
        self.weights = np.sign(self.weights_continuous)
        self.weights[self.weights == 0] = 1
        # print(f"new weights: {self.weights}")

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

        probl.shuffle()
        probl.pred = probl.forward()
        for i in range(probl.X.shape[0]):
            # calculate the gradients based on the discrete internal weights
            probl.calculate_grad(i)

            # make updates to the weights of the replica
            probl.step(lr)

            # discretize the new weights
            probl.discretize()

        # print(f"The current weights are: {probl.weights}")
       
        # stopping criterion, either solved the problem or max amount of epochs reached
        if probl.compute_cost() == 0:
            print("Problem solved.")
            stop = True
        elif epochs>=max_epochs:
            print(f"Maximum amount of epochs reached. Best cost reached= {probl.compute_cost()}")
            stop = True
        else: 
            print(f"Epoch {epochs+1}/{max_epochs} Completed! loss= {probl.compute_cost()}")
            epochs += 1

    return probl


# generalized for all batch sizes
def gd_batch(probl, lr: float, max_epochs: int, batch_size: int):

    probl.init_config()
    stop = False
    epochs = 0

    while stop is not True:
        probl.shuffle() 
        probl.pred = probl.forward()

        for i in range(batch_size, probl.P+1, batch_size):
            # print(f"i={i}")
            
            # calculate the gradients based on the discrete internal weights
            probl.calculate_batch_grad(i, batch_size)

            # make updates to the weights of the replica
            probl.step(lr)

            # discretize the new weights
            probl.discretize()
            
        # stopping criterion, either solved the problem or max amount of epochs reached
        if probl.compute_cost() == 0:
            print("Problem solved.")
            stop = True
        
        elif epochs>=max_epochs:
            print(f"Maximum amount of epochs reached. Best cost reached= {probl.compute_cost()}")
            stop = True

        else: 
            print(f"Epoch {epochs+1}/{max_epochs} Completed! loss= {probl.compute_cost()}")
            epochs += 1

    return probl