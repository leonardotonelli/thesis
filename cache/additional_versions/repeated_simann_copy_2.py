import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# SAME AS THE MAIN, BUT WITH SEPARATED DELTAS - CALCULATIONS CORRESPOND TO DELTA CALCULATED FROM EQUATION 1 OF S6

class BinaryPerceptronRepeated:
    def __init__(self, n: int , P: int, seed = None):
        '''
        Initializations
        '''
        self.n = n
        self.P = P
        self.alpha = P/n
        self.seed = seed
        self.weights = np.zeros(n)


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
        cost = int(wrong_bool.sum())
        self.cost = cost
        new_c = cost 
        return new_c

  
    def compute_delta_cost(self, action, reference_weights):
        '''
        Compute delta cost of a given action efficiently
        '''
        # current pred
        current_pred = self.pred
        current_cost = self.cost

        # delta predictions mathematically correct
        delta_pred = -2 * self.X[:, action] * self.weights[action]
        # derive new pred from the delta
        new_pred = current_pred + delta_pred.flatten()
        new_errors = (new_pred * self.targets) < 0
        new_cost = int(np.sum(new_errors))
        
        # compute delta
        delta_energy = new_cost - current_cost
        delta_reference = (2*self.weights[action]*reference_weights[action])

        return delta_energy, delta_reference


    def accept_action(self, action):
        '''
        Update the internal states given the taken action
        '''
        # update predictions
        delta_pred = (-2 * self.X[:, action] * self.weights[action]).flatten()
        self.pred = self.pred + delta_pred

        # update weights
        self.weights[action] = - self.weights[action]

        # update cost
        new_errors = (self.pred * self.targets) < 0
        self.cost = np.sum(new_errors)


    def propose_action(self):
        '''
        Propose a move based on some criteria
        '''
        if self.seed is not None:
            np.random.seed(self.seed)
        index = np.random.choice(range(self.n), size=1)
        return index

    def compute_distance(self, reference_weights):
        ''' 
        Function that computes the distance between the given replica to the reference
        '''
        d = np.sum((self.weights - reference_weights)**2)/2 ## 
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


class RepeatedSimann:
    def __init__(self, n: int = 10, P: int = 10, num_replicas: int = 10, reference_type: str = "average", seed: int = 1):
        self.reference_type = reference_type
        self.n = n
        self.P = P
        self.num_replicas = num_replicas
        self.seed = seed
        self.costs = np.zeros(num_replicas)
        self.replicas_weights = np.zeros(shape=(n, num_replicas))
        self.replicas_targets = np.zeros(shape=(P, num_replicas))

    def init_config(self):
        '''
        initialize all the replicas to same initial condition
        '''
        n = self.n
        P = self.P
        seed = self.seed

        num_replicas = self.num_replicas
        self.X = np.random.normal(loc = 0, scale = 1, size = (P, n))
        self.replicas = [BinaryPerceptronRepeated(n,P) for _ in range(num_replicas)]

        for i, replica in enumerate(self.replicas):
            replica.init_config(seed)
            self.costs[i] = replica.compute_cost() 
            self.replicas_weights[:,i] = replica.weights
            self.replicas_targets[:,i] = replica.targets

            # check that inizializations are all the same
            assert np.all(self.replicas_weights[:,i] == self.replicas_weights[:,0])
            assert np.all(self.replicas_targets[:,i] == self.replicas_targets[:,0])
            assert np.all(replica.X == self.replicas[0].X)
        

    def compute_best_cost(self, gamma, distance):
        '''
        compute the best cost among the replicas
        '''
        best = np.inf
        for i, replica in enumerate(self.replicas):
            cost = replica.compute_cost(gamma, distance)
            if cost < best:
                best = cost
                best_replica = replica
        return best, best_replica
    
    
    def compute_reference(self):
        '''
        compute the reference replica (only average for now)
        '''
        mean = np.mean(self.replicas_weights, axis=1)
        self.reference = mean


    def compute_deltas(self, replica_index, action):
        '''
        compute delta cost for the given replica, move and the reference weights
        '''
        reference_weights = self.reference
        delta_energy, delta_reference = self.replicas[replica_index].compute_delta_cost(action, reference_weights)
        return delta_energy, delta_reference


    def accept_action(self, replica_index, action):
        '''
        accept the move for the specific replica only
        '''
        self.replicas[replica_index].accept_action(action)
        self.costs[replica_index] = self.replicas[replica_index].cost
        self.replicas_weights[:, replica_index] = self.replicas[replica_index].weights


    def get_replica(self, replica_index):
        '''
        return the replica object requested 
        '''
        return self.replicas[replica_index]


    def get_convergence_status(self):
        '''
        return whether the replicas are converged or not 
        '''
        weights = self.replicas_weights
        converged = np.prod(np.all(weights == weights[0,:], axis = 0))
        return converged

    
def repeated_simann(probl, beta0, beta1, gamma0, gamma1, annealing_steps = 10, scooping_steps = 10, mcmc_steps = 10, seed = None):
    
    if seed != None:
        np.random.seed(seed)
        
    # initialize all the replicas to same initial condition
    probl.init_config()

    # compute the best initial cost for every replica (the same)
    best_cost, best_replica = probl.compute_best_cost(0,0)
    print(f"Initial cost: {best_cost}")

    # pick the replica with lowest cost? Not really needed.
    # best_probl = probl.copy() 
    # best_cost = cx

    # set all the betas to use for annealing
    betas = np.zeros(annealing_steps)
    betas[:-1] = np.linspace(beta0, beta1, annealing_steps-1)
    betas[-1] = np.inf

    # set all the gammas to use for scooping
    gammas = np.zeros(scooping_steps)
    gammas[:-1] = np.linspace(gamma0, gamma1, scooping_steps-1)


    for i in range(annealing_steps): # assume that annealing steps are the same for scooping for now
        accepted_moves = 0
        for t in range(mcmc_steps):

            # compute the reference replica (only average for now)
            probl.compute_reference()

            for replica_index, replica in enumerate(probl.replicas):

                # get a random replica and propose a random (for now) move
                action = replica.propose_action() 

                # compute delta cost for the given replica, move and the reference weights
                deltas = probl.compute_deltas(replica_index, action)

                if accept_with_prob(deltas, betas[i], gammas[i]):
                    accepted_moves += 1

                    # accept the move for the specific replica only
                    probl.accept_action(replica_index, action) 
                    cx = probl.costs[replica_index]

                    if cx < best_cost:
                        best_cost = cx
                        best_replica = probl.get_replica(replica_index).copy()
                        if best_cost == 0:
                            print("Solved.")
                            return (best_replica, best_cost)

        best_replica.display() # TODO
        print(f"beta = {betas[i]}, gamma = {gammas[i]}, c={cx}, best_c={best_cost}, accepted_freq={accepted_moves/mcmc_steps*probl.num_replicas}")
        
    # function to define whether the replica converged or not
    converged = probl.get_convergence_status()
    if converged:
        print("The replica converged to the same final assignment")
    else:
        print("The replicas did not converge")

        # stopping criterion -> maximum number of iteration reached
    return (best_replica, best_cost)


def accept_with_prob(deltas, beta, gamma):
    delta_energy, delta_reference = deltas
    if delta_energy <= 0:
        return True
    if beta == np.inf:
        return False
    

    prob = np.exp(-beta * delta_energy + gamma* delta_reference)
    return np.random.random() < prob


