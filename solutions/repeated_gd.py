import numpy as np
from copy import deepcopy
import time

class BinaryPerceptronGD:
    def __init__(self, n: int , P: int, seed: int=None):
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
        wrong_bool = (self.pred * self.targets) < 0
        cost = wrong_bool.sum()
        return cost


    def calculate_grad(self, index):
        '''
        compute gradient from loss computed for just the batch. It returns an array of gradients for each weight
        '''
        grad = np.zeros(self.n)
        for i, weight in enumerate(self.weights):
            pred = self.pred[index] + 0.001
            new_pred = pred - 2*self.X[index, i]*weight + 0.001
            current_loss = int( (pred >= 0)*1 == self.targets[index] )
            new_loss = int( (new_pred >= 0)*1 == self.targets[index] )
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
                current_loss = int( (pred >= 0)*1 == self.targets[index] )
                new_loss = int( (new_pred >= 0)*1 == self.targets[index] )
                # current_loss = int((pred * self.targets[index]) < 0)
                # new_loss = int((new_pred * self.targets[index]) < 0)
                batch_grads[index-final+batch_size,i] = (new_loss - current_loss)/(- 2*weight) #TODO TO CHANGE

        self.grad = np.mean(batch_grads, axis=0)


    def step(self, lr):
        '''
        make a continuous step towards the gradient direction
        '''
        self.weights_continuous = self.weights_continuous - lr * self.grad


    def discretize(self):
        self.weights = np.sign(self.weights_continuous)
        self.weights[self.weights == 0] = 1


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



class RepeatedGD:
    def __init__(self, n: int = 10, P: int = 10, num_replicas: int = 10, seed: int = 1):
        self.n = n
        self.P = P
        self.num_replicas = num_replicas
        self.seed = seed
        self.costs = np.zeros(num_replicas)
        self.replicas_weights = np.zeros(shape=(n, num_replicas))
        self.replicas_targets = np.zeros(shape=(P, num_replicas))
        self.grads = np.zeros(shape=(n, num_replicas))


    def init_config(self):
        '''
        initialize all the replicas to same initial condition
        '''
        n = self.n
        P = self.P
        seed = self.seed
        self.batches_final_indeces = np.zeros(self.num_replicas, dtype=int)

        num_replicas = self.num_replicas
        self.X = np.random.normal(loc = 0, scale = 1, size = (P, n))
        self.replicas = [BinaryPerceptronGD(n,P) for _ in range(num_replicas)]

        for i, replica in enumerate(self.replicas):
            replica.init_config(seed)
            self.costs[i] = replica.compute_cost() # set gamma and distance at zero because they are all at the same spot
            self.replicas_weights[:,i] = replica.weights
            self.replicas_targets[:,i] = replica.targets

            # check that inizializations are all the same
            assert np.all(self.replicas_weights[:,i] == self.replicas_weights[:,0])
            assert np.all(self.replicas_targets[:,i] == self.replicas_targets[:,0])
            assert np.all(replica.X == self.replicas[0].X)


    def compute_reference(self):
        '''
        compute the reference replica (only average)
        '''
        mean = np.sum(self.replicas_weights, axis=1)
        self.reference = mean


    def random_replica(self):
        ''' 
        Method that returns a random replica index to propose actions
        '''
        return np.random.randint(self.num_replicas)


    def calculate_grad(self, replica_index, batch_size):
        ''' 
        Method that calculate the gradient needed for the update
        '''
        final_index = self.batches_final_indeces[replica_index]
        if final_index >= self.P:
            return None
        else:
            self.replicas[replica_index].calculate_batch_grad(final_index, batch_size)
            self.batches_final_indeces[replica_index] += batch_size


    def step(self, replica_index, gamma, beta, lr):
        ''' 
        Method that modifies the specific replica weights according to the update rule
        '''
        self.replicas[replica_index].step(lr)
        self.replicas[replica_index].discretize()
        # self.compute_reference() #TODO??
        self.replicas[replica_index].weights_continuous += gamma/(beta*lr) * (np.tanh(gamma*self.reference) - self.replicas[replica_index].weights)


    def epoch_passed(self):
        ''' 
        Method that check whether a full epoch as passed, if yes it reinitializes the record
        '''
        return np.all(self.batches_final_indeces >= self.P)
        

    def reset_batch_counter(self):
        ''' It resents the final indeces of each replica, to start from zeros for each replica'''
        self.batches_final_indeces = np.zeros(self.num_replicas, dtype=int)


    def get_best(self):
        ''' 
        method to get the best performing replica and return it
        '''
        best = np.inf
        for replica in self.replicas:
            c = replica.compute_cost()
            if c <= best:
                best = c
                best_replica = replica
        return best, best_replica

    def shuffle(self):
        for replica in self.replicas:
            replica.shuffle()



def replicated_gd(probl, lr: float, max_epochs: int, batch_size: int, gamma0, gamma1, beta):

    probl.init_config()
    stop = False
    epoch = 0
    gammas = np.linspace(gamma0, gamma1, max_epochs)
    gamma = gammas[epoch]
    best_cost = np.inf
    best_replica = None

    while stop is not True:
        
        # compute the reference weights (just as the sum)
        probl.compute_reference()

        # get a random replica index
        replica_index = probl.random_replica()

        # calculate the gradients based on the discrete internal weights
        probl.calculate_grad(replica_index, batch_size)

        # make updates to the weights of the replica
        probl.step(replica_index, gamma, beta, lr)

        # discretize the new weights
        probl.replicas[replica_index].discretize()

        b_cost, b_replica = probl.get_best()

        if b_cost < best_cost:
            best_cost = b_cost
            best_replica = b_replica

        # stopping criterion, either solved the problem or max amount of epochs reached
        if best_cost == 0:
            print("Problem solved.")
            stop = True
        elif epoch>=max_epochs:
            print(f"Maximum amount of epochs reached. Best cost reached= {best_cost}")
            stop = True
        # check whether an epoch has passed
        elif probl.epoch_passed():
            print(f"Epoch {epoch+1}/{max_epochs} Completed! best loss= {best_cost}")
            epoch += 1
            gamma = gammas[epoch]
            probl.shuffle()
            probl.reset_batch_counter()

    return best_replica