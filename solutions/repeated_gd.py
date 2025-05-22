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
            current_loss = int((pred * self.targets[index]) < 0)
            new_loss = int((new_pred * self.targets[index]) < 0)
            grad[i] = (new_loss - current_loss)/(-2*weight)

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
                current_loss = int((pred * self.targets[index]) < 0)
                new_loss = int((new_pred * self.targets[index]) < 0)
                batch_grads[index-final+batch_size,i] = (new_loss - current_loss)/(- 2*weight)
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


class RepeatedGD:
    def __init__(self, n: int = 10, P: int = 10, num_replicas: int = 10, reference_type: str = "average", seed: int = 1):
        self.reference_type = reference_type
        self.n = n
        self.P = P
        self.num_replicas = num_replicas
        self.seed = seed
        self.costs = np.zeros(num_replicas)
        self.replicas_weights = np.zeros(shape=(n, num_replicas))
        self.replicas_targets = np.zeros(shape=(P, num_replicas))
        self.grads = np.zeros(shape=(n, num_replicas))
        self.epoch_records = np.full(self.num_replicas, -1)

    def init_config(self, batch_size):
        '''
        initialize all the replicas to same initial condition
        '''
        n = self.n
        P = self.P
        seed = self.seed
        self.batches_final_indeces = np.zeros(self.num_replicas)

        num_replicas = self.num_replicas
        self.X = np.random.normal(loc = 0, scale = 1, size = (P, n))
        self.replicas = [BinaryPerceptronGD(n,P) for _ in range(num_replicas)]

        for i, replica in enumerate(self.replicas):
            replica.init_config(seed)
            self.costs[i] = replica.compute_cost(0, 0) # set gamma and distance at zero because they are all at the same spot
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
        ''' Method that returns a random replica index to propose actions'''
        return np.random.randint(self.num_replicas)

    def calculate_grad(self, replica_index, batch_size, epoch):
        ''' Method that calculate the gradient needed for the update'''
        final_index = self.batches_final_indeces[replica_index]
        if self.epoch_records[replica_index] >= self.P:
            return None
        else:
            self.replicas[replica_index].calculate_batch_grad(final_index, batch_size)
            self.batches_final_indeces[replica_index] += batch_size

    def step(self, replica_index, gamma, beta, lr):
        ''' Method that modifies the specific replica weights according to the update rule'''
        self.replicas[replica_index].step(lr)
        self.compute_reference()
        self.replicas[replica_index].weights += gamma/(beta*lr) * (np.tanh(gamma*self.reference) - self.replicas[replica_index].weights)

    def discretize(self):
        ''' Method that discretize the weights after the step'''

    def epoch_passed(self):
        ''' Method that check whether a full epoch as passed, if yes it reinitializes the record'''
        return np.all(self.batches_final_indeces >= self.P)
        
    def reset_batch_counter(self):
        ''' It resents the final indeces of each replica, to start from zeros for each replica'''
        self.batches_final_indeces = np.zeros(self.num_replicas)

    def best_cost(self):
        ''' method to get the best cost across all the replicas results'''

    def get_best_replica(self):
        ''' method to get the best performing replica and return it'''

    def shuffle(self):
        for replica in self.replicas:
            replica.shuffle()



def replicated_gd(probl, lr: float, max_epochs: int, batch_size: int, scooping_steps: int, gamma0, gamma1, beta):

    probl.init_config(batch_size)
    stop = False
    epoch = 0
    gammas = np.zeros(scooping_steps)
    gammas[:-1] = np.linspace(gamma0, gamma1, scooping_steps-1)
    gamma = gammas[epoch]


    while stop is not True:
        
        # compute the reference weights (just as the sum)
        probl.compute_reference()

        # get a random replica index
        replica_index = probl.random_replica()

        # calculate the gradients based on the discrete internal weights
        probl.calculate_grad(replica_index, batch_size, epoch)

        # make updates to the weights of the replica
        probl.step(replica_index, gamma, beta, lr)

        # discretize the new weights
        probl.discretize(replica_index)

        # check whether an epoch as passed
        if probl.epoch_passed():
            epoch += 1
            gamma = gammas[epoch]
            probl.shuffle()
            probl.reset_batch_counter()

        
        # stopping criterion, either solved the problem or max amount of epochs reached
        if probl.is_solved(replica_index):
            print("Problem solved.")
            stop = True
        elif epoch>max_epochs:
            print(f"Maximum amount of epochs reached. Best cost reached= {probl.best_cost()}")
            stop = True

    # get the best replica to return 
    best_replica = probl.get_best_replica()

    return best_replica