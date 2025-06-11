import numpy as np
import matplotlib.pyplot as plt
import time
from copy import deepcopy

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


    def compute_cost(self, gamma, distance):
        '''
        Define the cost function and computation
        '''
        self.pred = self.forward()
        cost = np.sum( (self.pred >= 0) == (self.targets == 1) )
        self.cost = cost
        new_c = cost + gamma*distance
        return new_c

  
    def compute_delta_cost(self, action, gamma, reference_weights):
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
        new_cost = np.sum( (new_pred >= 0) == (self.targets == 1) )
        
        # compute delta
        delta = new_cost - current_cost
        
        # compute the final delta incorporating the gamma parameter
        final_delta = float(delta + gamma * (2*self.weights[action]*reference_weights[action]))

        # VERIFICATION CORRECTNESS
        distance1 = self.compute_distance(reference_weights)

        # current pred
        starting_cost = self.compute_cost(gamma, distance1)
        temp_problem = self.copy()
        temp_problem.accept_action(action)  

        distance2 = temp_problem.compute_distance(reference_weights)
        new_cost = temp_problem.compute_cost(gamma, distance2)
        delta_real = (new_cost - starting_cost)

        # check whether the new delta cost is working
        assert round(delta_real,2) == round(final_delta,2)

        return final_delta


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
        # new_errors = (self.pred * self.targets) < 0
        # self.cost = np.sum(new_errors)
        self.cost = np.sum( (self.pred >= 0) == (self.targets == 1) )


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
        self.iterations_to_solution = np.nan
        self.time_to_solution = np.nan

    def init_config(self):
        '''
        initialize all the replicas to same initial condition
        '''
        n = self.n
        P = self.P
        seed = self.seed

        num_replicas = self.num_replicas
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
        compute the reference replica (only average)
        '''
        mean = np.mean(self.replicas_weights, axis=1)
        self.reference = mean
        
        
    def propose_action(self):
        '''
        get a random replica and propose a random move
        '''
        replica_index = np.random.randint(self.num_replicas)
        action = self.replicas[replica_index].propose_action()

        return replica_index, action

    def compute_delta_distances_from_reference(self, replica_index, ):
        dist = np.zeros(self.num_replicas)
        for i, replica in self.replicas:
            dist[i] = replica.compute_distance(self.reference)
        return np.sum(dist)

    def compute_delta_cost(self, replica_index, action, gamma):
        '''
        compute delta cost for the given replica, move and the reference weights
        '''
        reference_weights = self.reference
        delta_cost = self.replicas[replica_index].compute_delta_cost(action, gamma, reference_weights)

        # probl_copy = self.copy()
        # probl_copy.accept_action(replica_index, action)
        
        # dist = np.zeros(self.num_replicas)
        # dist_copy = np.zeros(self.num_replicas)

        # for i in range(self.num_replicas):
        #     dist[i] = self.replicas[replica_index].compute_distance(self.reference)
        #     dist_copy[i]= probl_copy.replicas[replica_index].compute_distance(probl_copy.reference)

        # delta_distances = np.sum(dist) - np.sum(dist_copy)


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
        return whether the replicas converged or not 
        '''
        weights = self.replicas_weights
        converged = np.prod(np.all(weights == weights[0,:], axis = 0))
        return converged

    
def repeated_simann(probl, beta0, beta1, gamma0, gamma1, annealing_steps = 10, scooping_steps = 10, mcmc_steps = 10, seed = None, verbose=0, collect=0):
    
    if seed != None:
        np.random.seed(seed)
        
    # initialize all the replicas to same initial condition
    probl.init_config()

    # compute the best initial cost for every replica (the same)
    best_cost, best_replica = probl.compute_best_cost(0,0)
    cx = best_cost

    if verbose:
        print(f"Initial cost: {best_cost}")

    # set all the betas to use for annealing
    betas = np.zeros(annealing_steps)
    betas[:-1] = np.linspace(beta0, beta1, annealing_steps-1)
    betas[-1] = np.inf

    # set all the gammas to use for scooping
    gammas = np.zeros(scooping_steps)
    gammas[:-1] = np.linspace(gamma0, gamma1, scooping_steps-1)

    # initialize timer
    start_time = time.time()

    if collect:
        accepted_frequencies = []
        costs = []

    for i in range(annealing_steps): # assume that annealing steps are the same for scooping for now
        accepted_moves = 0
        for t in range(mcmc_steps): 

            # compute the reference replica (only average for now)
            probl.compute_reference()

            # get a random replica and propose a random (for now) move
            replica_index, action = probl.propose_action() 

            # compute delta cost for the given replica, move and the reference weights
            delta_c = probl.compute_delta_cost(replica_index, action, gammas[i]/betas[i])

            if accept_with_prob(delta_c, betas[i]):
                accepted_moves += 1

                # accept the move for the specific replica only
                probl.accept_action(replica_index, action) 
                cx = probl.costs[replica_index]

                if cx < best_cost:
                    best_cost = cx
                    best_replica = probl.get_replica(replica_index).copy()    
            
            if best_cost == 0:
                if verbose:
                    print(f"Solved. After {i} annealing steps.")
                if collect:
                    accepted_frequencies.append(accepted_moves/mcmc_steps)
                    costs.append(best_cost)
                    best_replica.collected_costs = costs
                    best_replica.collected_frequencies = accepted_frequencies
                best_replica.iterations_to_solution = i+1
                end_time = time.time()
                best_replica.time_to_solution = end_time - start_time
                return (best_replica, best_cost)
        
        if collect:
            accepted_frequencies.append(accepted_moves/mcmc_steps)
            costs.append(best_cost)

        if verbose:
            best_replica.display() # TODO
            print(f"beta = {betas[i]}, gamma = {gammas[i]}, c={cx}, best_c={best_cost}, accepted_freq={accepted_moves/mcmc_steps}")
        
    # function to define whether the replica converged or not
    if verbose:
        converged = probl.get_convergence_status()
        if converged:
            print("The replica converged to the same final assignment")
        else:
            print("The replicas did not converge")

    
    if collect:
        best_replica.collected_costs = costs
        best_replica.collected_frequencies = accepted_frequencies
        best_replica.iterations_to_solution = annealing_steps
        
    return (best_replica, best_cost)


def accept_with_prob(delta_cost, beta):
    if delta_cost <= 0:
        return True
    if beta == np.inf:
        return False
    
    prob = np.exp(-beta * delta_cost)
    return np.random.random() < prob


