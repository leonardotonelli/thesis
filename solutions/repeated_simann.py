import numpy as np
import matplotlib.pyplot as plt
from problems.repeatedv2 import BinaryPerceptronRepeated


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
            self.costs[i] = replica.compute_cost(0, 0) # set gamma and distance at zero because they are all at the same spot
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
        
        
    def propose_action(self):
        '''
        get a random replica and propose a random (for now) move
        '''
        replica_index = np.random.randint(self.num_replicas)
        action = self.replicas[replica_index].propose_action()

        return replica_index, action


    def compute_delta_cost(self, replica_index, action, gamma):
        '''
        compute delta cost for the given replica, move and the reference weights
        '''
        reference_weights = self.reference
        delta_cost = self.replicas[replica_index].compute_delta_cost(action, gamma, reference_weights)
        return delta_cost


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

            # get a random replica and propose a random (for now) move
            replica_index, action = probl.propose_action() 

            # compute delta cost for the given replica, move and the reference weights
            delta_c = probl.compute_delta_cost(replica_index, action, gammas[i])

            if accept_with_prob(delta_c, betas[i]):
                accepted_moves += 1

                # accept the move for the specific replica only
                probl.accept_action(replica_index, action) 
                cx = probl.costs[replica_index]

                if cx < best_cost:
                    best_cost = cx
                    best_replica = probl.get_replica(replica_index).copy()


        best_replica.display() # TODO
        print(f"beta = {betas[i]}, gamma = {gammas[i]}, c={cx}, best_c={best_cost}, accepted_freq={accepted_moves/mcmc_steps}")
        
    # function to define whether the replica converged or not
    converged = probl.get_convergence_status()
    if converged:
        print("The replica converged to the same final assignment")
    else:
        print("The replicas did not converge")

        # stopping criterion -> maximum number of iteration reached
    return (best_replica, best_cost)


def accept_with_prob(delta_cost, beta):
    if delta_cost <= 0:
        return True
    if beta == np.inf:
        return False
    
    prob = np.exp(-beta * delta_cost)
    return np.random.randn() < prob


