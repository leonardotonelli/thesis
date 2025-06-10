import numpy as np
from solutions.repeated_simann import repeated_simann, RepeatedSimann
from solutions.repeated_gd import replicated_gd, RepeatedGD
import pandas as pd


def save_extra_data_sa(N, P, num_replicas, beta0, beta1, gamma0, gamma1, path, seed = None):
    probl = RepeatedSimann(N,P,num_replicas, seed=seed)
    best_replica, best_cost = repeated_simann(probl, beta0, beta1, gamma0, gamma1, annealing_steps = 1000, scooping_steps = 1000, mcmc_steps = 200, seed = seed, verbose=1, collect=1)

    freq = np.array(best_replica.collected_frequencies)
    costs = np.array(best_replica.collected_costs)
    iterations = range(1, best_replica.iterations_to_solution+1)

    df = pd.DataFrame({
        'freq': freq,
        'cost': costs,
        'iteration': iterations
    })

    df.to_csv(path, index=False)
    print(f"\n CSV saved to: {path}")


def save_extra_data_gd(N, P, num_replicas, lr, gamma0, gamma1, path, seed = None):
    probl = RepeatedGD(N,P,num_replicas, seed)
    best_replica = replicated_gd(probl, lr=lr, max_epochs=1000, batch_size=10, gamma0=gamma0, gamma1=gamma1, beta=0, verbose=0, collect=1)

    error_rates = np.array(best_replica.collected_error_rates)
    epochs = range(1, best_replica.epochs+1)

    df = pd.DataFrame({
        'error_rates': error_rates,
        'epochs': epochs
    })

    df.to_csv(path, index=False)
    print(f"\n CSV saved to: {path}")