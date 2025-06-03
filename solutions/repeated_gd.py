import numpy as np
from copy import deepcopy

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
        pred_labels = np.where(self.pred >= 0, 1, -1)
        cost = np.sum(pred_labels != self.targets)
        return cost
        

    # def calculate_batch_grad(self, final, batch_size):
    #     '''
    #     compute gradient from loss computed for just the batch. It returns an array of gradients for each weight
    #     '''
    #     batch_grads = np.zeros((batch_size, self.n))
    #     self.pred = self.forward()
    #     for index in range(final - batch_size, final):
    #         for i, weight in enumerate(self.weights):
    #             pred = self.pred[index]
    #             new_pred = pred - 2*self.X[index, i]*weight
    #             current_loss = 0 if (pred>=0)*1 == self.targets[index] else abs(pred)
    #             new_loss = 0 if (new_pred>=0)*1 == self.targets[index] else abs(new_pred)
    #             batch_grads[index-final+batch_size,i] = (new_loss - current_loss)/(- 2*weight) 

    #     self.grad = np.mean(batch_grads, axis=0)

    def calculate_batch_grad(self, final, batch_size):
        batch_grads = np.zeros((batch_size, self.n))
        batch_indices = range(final - batch_size, final)
        
        for idx, sample_idx in enumerate(batch_indices):
            prediction = np.dot(self.X[sample_idx], self.weights)
            target = self.targets[sample_idx]
            
            # Usa la stessa regola di decisione del tuo codice originale
            predicted_label = 1 if prediction >= 0 else -1
            
            # Regola del perceptron: aggiorna solo se classificazione errata
            if predicted_label != target:
                batch_grads[idx] = -target * self.X[sample_idx]
            else:
                batch_grads[idx] = np.zeros(self.n)
        
        self.grad = np.mean(batch_grads, axis=0)


    def step(self, lr):
        '''
        make a continuous step towards the gradient direction
        '''
        self.weights_continuous = self.weights_continuous - lr * self.grad


    def discretize(self):
        new_weights = np.where(self.weights_continuous>=0, 1, -1)
        self.weights = new_weights


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
        self.grads = np.zeros(shape=(n, num_replicas))


    def init_config(self, batch_size):
        '''
        initialize all the replicas to same initial condition
        '''
        n = self.n
        P = self.P
        seed = self.seed

        self.batches_final_indeces = np.ones(self.num_replicas, dtype=int)*batch_size
        self.replicas = [BinaryPerceptronGD(n,P) for _ in range(self.num_replicas)]

        for replica in self.replicas:
            replica.init_config(seed)
            # check that inizializations are all the same
            assert np.all(replica.targets == self.replicas[0].targets)
            assert np.all(replica.X == self.replicas[0].X)


    def compute_reference(self):
        '''
        compute the reference replica 
        '''
        current_weights = np.array([replica.weights for replica in self.replicas])
        self.reference = np.mean(current_weights, axis=0)


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

        if final_index <= self.P:
            self.replicas[replica_index].calculate_batch_grad(final_index, batch_size)
            self.batches_final_indeces[replica_index] += batch_size


    def step(self, replica_index, gamma, beta, lr):
        ''' 
        Method that modifies the specific replica weights according to the update rule
        # '''
        # self.replicas[replica_index].step(lr)
        # self.replicas[replica_index].discretize()
        # # self.compute_reference() 
        # self.replicas[replica_index].weights_continuous += gamma/(beta*lr) * (np.tanh(gamma*self.reference) - self.replicas[replica_index].weights)
        # update = - lr*self.replicas[replica_index].grad + gamma/(beta*lr) * (np.tanh(gamma*self.reference) - self.replicas[replica_index].weights)
        update = - lr*self.replicas[replica_index].grad + gamma*lr * (self.reference - self.replicas[replica_index].weights)
        current_weights = self.replicas[replica_index].weights_continuous
        self.replicas[replica_index].weights_continuous = current_weights + update


    def epoch_passed(self):
        ''' 
        Method that check whether a full epoch as passed, if yes it reinitializes the record
        '''
        return np.all(self.batches_final_indeces >= self.P)
        

    def reset_batch_counter(self, batch_size):
        ''' It resents the final indeces of each replica, to start from zeros for each replica'''
        self.batches_final_indeces = np.ones(self.num_replicas, dtype=int)*batch_size


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

    probl.init_config(batch_size)
    stop = False
    epoch = 0
    gammas = np.linspace(gamma0, gamma1, max_epochs+1)
    gamma = gammas[epoch]
    best_cost = np.inf
    best_replica = None
    best_idx = None

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
            best_replica = b_replica.copy()
            best_idx = replica_index

        # stopping criterion, either solved the problem or max amount of epochs reached
        if best_cost == 0:
            print("Problem solved.")
            stop = True
            best_replica.error_rate = 0
            best_replica.epochs = epoch
        elif epoch>=max_epochs:
            print(f"Maximum amount of epochs reached. Best cost reached= {best_cost}")
            stop = True
            best_replica.error_rate = round(best_cost / best_replica.P, 2)
            best_replica.epochs = epoch
        # check whether an epoch has passed
        elif probl.epoch_passed():
            print(f"Epoch {epoch+1}/{max_epochs} Completed! best loss= {b_cost}")
            epoch += 1
            gamma = gammas[epoch]
            probl.shuffle()
            probl.reset_batch_counter(batch_size)

    return best_replica