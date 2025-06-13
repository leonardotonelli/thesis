import numpy as np
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
        self.iterations_to_solution = np.nan
        self.time_to_solution = np.nan


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
        cost = np.sum( (self.pred >= 0) == (self.targets == 1) )
        self.cost = cost
        new_c = cost
        return new_c

  
    def compute_delta_cost(self, action):
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

        return delta


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
        self.iterations_to_solution = np.nan
        self.time_to_solution = np.nan


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
        cost = np.sum( (self.pred >= 0) == (self.targets == 1) )
        self.cost = cost
        new_c = cost
        return new_c

  
    def compute_delta_cost(self, action):
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

        return delta


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


class RepeatedSimann_alt:
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
        self.replicas = [BinaryPerceptronRepeated(n,P) for _ in range(num_replicas)]

        for i, replica in enumerate(self.replicas):
            replica.init_config(seed)
            self.costs[i] = replica.compute_cost() # set gamma and distance at zero because they are all at the same spot
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
            cost = replica.compute_cost()
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

    def compute_delta_distances_from_reference(self):
        dist = np.zeros(self.num_replicas)
        for i, replica in enumerate(self.replicas):
            dist[i] = replica.compute_distance(self.reference)
        return np.sum(dist)

    def compute_delta_cost(self, replica_index, action):
        '''
        compute delta cost for the given replica, move and the reference weights
        '''
        delta_cost = self.replicas[replica_index].compute_delta_cost(action)
        
        # delta_ref = -2* self.replicas[replica_index].weights[action]
        # new_ref = self.reference[action] + delta_ref
        # ref = self.reference[action]
        # current_weight = self.replicas[replica_index].weights[action]
        # new_weight = - current_weight

        # y = self.num_replicas
        # delta_distance = (y-1)/2 * (new_ref**2 - ref**2) + delta_ref* (np.sum(np.delete(self.replicas_weights[action,], replica_index))) + (1/2)* (new_ref - new_weight)**2 - (1/2)*(ref-current_weight)**2
        
        # verification
        current_distance = self.compute_delta_distances_from_reference()
        copy = self.copy()
        copy.accept_action(replica_index, action)
        copy.compute_reference()
        new_distance = copy.compute_delta_distances_from_reference()

        delta_real = new_distance - current_distance

        # print(f"This is the calculated: {delta_distance}. While this is the real: {delta_real}")

        
        
        
        
        
        # # Current distance from reference for the replica being modified
        # current_distance = self.replicas[replica_index].compute_distance(self.reference)
        
        # # Compute new distance after flipping the weight
        # # When we flip weight[action], the new weight becomes -old_weight
        # old_weight = self.replicas[replica_index].weights[action]
        # new_weight = -old_weight
        
        # # Distance formula is sum((weights - reference)**2)/2
        # # Only the action-th component changes, so we can compute the delta efficiently
        # old_contribution = (old_weight - self.reference[action])**2
        # new_contribution = (new_weight - self.reference[action])**2
        
        # new_distance = current_distance - old_contribution/2 + new_contribution/2
        # delta_distances = new_distance - current_distance
        
        return delta_cost, delta_real

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
    
    def copy(self):
        '''
        Copy the whole problem
        '''
        return deepcopy(self)

    
def repeated_simann_alt(probl, beta0, beta1, gamma0, gamma1, seed=None, verbose=0):
    """
    Simulated Annealing with exponential annealing and scoping.
    
    Parameters:
    - probl: Problem instance
    - beta0: Initial inverse temperature
    - gamma0: Initial interaction strength  
    - beta_f: Beta increase factor (should scale with N)
    - gamma_f: Gamma increase factor (should scale with N)
    - y: Parameter for annealing frequency (increase every 1000*y accepted moves)
    - N: System size (for stopping criterion of 1000*N*y non-improving moves)
    - seed: Random seed
    - verbose: Verbosity level
    """
    
    if seed != None:
        np.random.seed(seed)
        
    # Initialize all replicas to same random configuration
    probl.init_config()

    # Compute the best initial cost for every replica (the same)
    best_cost, best_replica = probl.compute_best_cost(0, 0)
    cx = None

    if verbose:
        print(f"Initial cost: {best_cost}")

    # Initialize parameters
    current_beta = beta0
    current_gamma = gamma0
    
    # Counters
    accepted_moves = 0
    non_improving_moves = 0
    total_moves = 0
    y = probl.num_replicas
    N = probl.n
    
    max_non_improving = 1000 * N * y

    # Initialize timer
    start_time = time.time()

    if verbose:
        print(f"Starting with beta={current_beta}, gamma={current_gamma}")
        print(f"Will increase parameters every {1000*y} accepted moves")
        print(f"Will stop after {max_non_improving} consecutive non-improving moves")

    while True:
        # Compute the reference replica (only average for now)
        probl.compute_reference()

        # Get a random replica and propose a random move
        replica_index, action = probl.propose_action() 

        # Compute delta cost for the given replica, move and the reference weights
        delta_c, delta_d = probl.compute_delta_cost(replica_index, action)
        
        total_moves += 1
        # print(non_improving_moves)
        move_accepted = False
        move_improved = False


        if accept_with_prob(delta_c, delta_d, current_beta, current_gamma):
            accepted_moves += 1
            move_accepted = True

            # Accept the move for the specific replica only
            probl.accept_action(replica_index, action) 
            cx = probl.costs[replica_index]

            if cx < best_cost:
                best_cost = cx
                best_replica = probl.get_replica(replica_index).copy()
                move_improved = True
                non_improving_moves = 0  # Reset counter on improvement
                
                # Check if we found the optimal solution
                if best_cost == 0:
                    if verbose:
                        print(f"Solved! Best cost = 0 reached.")
                    
                    best_replica.iterations_to_solution = total_moves
                    end_time = time.time()
                    best_replica.time_to_solution = end_time - start_time
                    return (best_replica, best_cost)

        # Count non-improving moves (rejected or didn't lower energy)
        if not move_accepted or not move_improved:
            non_improving_moves += 1
            
        # Check stopping criterion for non-improving moves
        if non_improving_moves >= max_non_improving:
            if verbose:
                print(f"Stopped after {max_non_improving} consecutive non-improving moves")
            break

        # Annealing and scoping: increase β and γ every 1000*y accepted moves
        if accepted_moves > 0 and accepted_moves % (1000 * y) == 0:
            current_beta *= (1 + beta1)
            current_gamma *= (1 + gamma1)
            
            if verbose:
                best_replica.display() # TODO
                print(f"beta = {current_beta}, gamma = {current_gamma}, c={cx}, best_c={best_cost}, accepted_freq={accepted_moves/(1000*y)}")

    # Final results
    if verbose:
        best_replica.display() # TODO
        print(f"beta = {current_beta}, gamma = {current_gamma}, c={cx}, best_c={best_cost}")
        
        # function to define whether the replica converged or not
        converged = probl.get_convergence_status()
        if converged:
            print("The replica converged to the same final assignment")
        else:
            print("The replicas did not converge")

        # stopping criterion -> maximum number of iteration reached

    end_time = time.time()
    # best_replica.time_to_solution = end_time - start_time
    # best_replica.iterations_to_solution = total_moves
        
    return (best_replica, best_cost)

def accept_with_prob(delta_cost, delta_d, beta, gamma):
    prob = min(1.0, np.exp(-beta * delta_cost + gamma * delta_d) )
    if beta == np.inf:
        return False
    return np.random.random() < prob


