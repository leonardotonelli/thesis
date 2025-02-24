import numpy as np
import matplotlib.pyplot as plt


def simann(probl, beta0, beta1, annealing_steps=10, mcmc_steps = 10, seed = None):
    
    if seed != None:
        np.random.seed(seed)
        
    probl.init_config()
    cx = probl.compute_cost()
    print(f"Initial cost: {cx}")
    
    best_probl = probl.copy()
    best_cost = cx
    betas = np.zeros(annealing_steps)
    betas[:-1] = np.linspace(beta0, beta1, annealing_steps-1)
    betas[-1] = np.inf
    
    for beta in betas:
        accepted_moves = 0
        for t in range(mcmc_steps):
            move = probl.propose_action()
            delta_c = probl.compute_delta_cost(move)
            print(delta_c)

            if accept_with_prob(delta_c, beta):
                accepted_moves += 1
                probl.accept_action(move)
                cx += delta_c
          
                if cx < best_cost:
                    best_cost = cx
                    best_probl = probl.copy()

        best_probl.display()
    
        print(f"beta = {beta}, c={cx}, best_c={best_cost}, accepted_freq={accepted_moves/mcmc_steps}")
    
        # stopping criterion -> maximum number of iteration reached
    return (best_probl, best_cost)


def accept_with_prob(delta_cost, beta):
    if delta_cost <= 0:
        return True
    if beta == np.inf:
        return False
    
    prob = np.exp(-beta * delta_cost)
    return np.random.randn() < prob


