from cache.additional_versions.repeated_simann_alt2 import repeated_simann_alt, RepeatedSimann_alt
from solutions.repeated_simann import repeated_simann, RepeatedSimann
import numpy as np
import pandas as pd


## SIMULATED ANNEALING ##
# def rsa_collect_size_comparison_alt(size_limit, alpha, num_replicas, sample_size=10, path="data/rsa/comparison_size_data.csv"):

#     sizes = list(range(100, size_limit+1, 50))
#     print(sizes)

#     dict = {"type": [], "size": [], "iterations": [], "time":[], "num_replicas": []}

#     for size in sizes:
#         print(f"Collecting for size {size}...")
#         n = size
#         P = int(alpha * n)
#         annealing_steps = 1000 #max annealing steps
#         mcmc_steps = 200  
#         scooping_steps = 1000
#         beta0 = 2
#         beta1 = 3

#         for i in range(sample_size):

#             # INTERACTING #
#             rep_interacting = RepeatedSimann_alt(n, P, num_replicas, seed=i)
#             best_bp_interacting, best_cost_interacting = repeated_simann_alt(rep_interacting, beta0, beta1, gamma0=.3, gamma1=0.01)
#             dict["type"].append("interacting")
#             dict["size"].append(size)
#             dict["iterations"].append(best_bp_interacting.iterations_to_solution)
#             dict["time"].append(best_bp_interacting.time_to_solution)
#             dict["num_replicas"].append(num_replicas)
#             print(f"{i+1}/{sample_size} interacting ok - Solved? {best_cost_interacting == 0}")
            
#             # NON-INTERACTING #
#             rep_non_interacting = RepeatedSimann_alt(n, P, num_replicas, seed=i)
#             best_bp_nointeracting, best_cost_nointeracting = repeated_simann_alt(rep_non_interacting, beta0, beta1, gamma0=0, gamma1=0)
#             dict["type"].append("non_interacting")
#             dict["size"].append(size)
#             dict["iterations"].append(best_bp_nointeracting.iterations_to_solution)
#             dict["time"].append(best_bp_nointeracting.time_to_solution)
#             dict["num_replicas"].append(num_replicas)
#             print(f"{i+1}/{sample_size} non-interacting ok - Solved? {best_cost_nointeracting == 0}")
            
#             # # STANDARD #
#             # rep = RepeatedSimann(n, P, num_replicas=1, seed=i)
#             # best_bp_standard, best_cost_standard = repeated_simann(rep, beta0, beta1, gamma0=0, gamma1=0)
#             # dict["type"].append("standard")
#             # dict["size"].append(size)
#             # dict["iterations"].append(best_bp_standard.iterations_to_solution)
#             # dict["time"].append(best_bp_standard.time_to_solution)
#             # dict["num_replicas"].append(1)
#             # print(f"{i+1}/{sample_size} standard ok - Solved? {best_cost_standard == 0}")

#             assert np.all(rep_interacting.replicas[0].X == rep_non_interacting.replicas[0].X)
#             assert np.all(rep_interacting.replicas[0].targets == rep_non_interacting.replicas[0].targets)
 

        
#         print(f"Collection for size {size} is completed.")

#     df = pd.DataFrame(dict)
#     df.to_csv(path, index=False)
#     print(f"\n CSV saved to: {path}")

def rsa_collect_size_comparison_alt(size_limit, alpha, num_replicas, sample_size=10, path="data/rsa/comparison_size_data.csv"):

    sizes = list(range(10, size_limit+1, 20))
    print(sizes)

    dict = {"type": [], "size": [], "iterations": [], "time":[], "num_replicas": []}

    for size in sizes:
        print(f"Collecting for size {size}...")
        n = size
        P = int(alpha * n)
        annealing_steps = 1000 #max annealing steps
        mcmc_steps = 200  
        scooping_steps = 1000
        beta0 = .1
        beta1 = 5

        for i in range(sample_size):

            # INTERACTING #
            rep_interacting = RepeatedSimann_alt(n, P, num_replicas, seed=i)
            best_bp_interacting, best_cost_interacting = repeated_simann_alt(rep_interacting, beta0, beta1, gamma0=.6, gamma1=1.5, annealing_steps = annealing_steps, scooping_steps = scooping_steps, mcmc_steps = mcmc_steps)
            dict["type"].append("interacting")
            dict["size"].append(size)
            dict["iterations"].append(best_bp_interacting.iterations_to_solution)
            dict["time"].append(best_bp_interacting.time_to_solution)
            dict["num_replicas"].append(num_replicas)
            print(f"{i+1}/{sample_size} interacting ok")
            
            # NON-INTERACTING #
            rep_non_interacting = RepeatedSimann_alt(n, P, num_replicas, seed=i)
            best_bp_nointeracting, best_cost_nointeracting = repeated_simann_alt(rep_non_interacting, beta0, beta1, gamma0=0, gamma1=0, annealing_steps = annealing_steps, scooping_steps = scooping_steps, mcmc_steps = mcmc_steps)
            dict["type"].append("non_interacting")
            dict["size"].append(size)
            dict["iterations"].append(best_bp_nointeracting.iterations_to_solution)
            dict["time"].append(best_bp_nointeracting.time_to_solution)
            dict["num_replicas"].append(num_replicas)
            print(f"{i+1}/{sample_size} non-interacting ok")
            
            # # STANDARD #
            # rep = RepeatedSimann(n, P, num_replicas=1, seed=i)
            # best_bp_standard, best_cost_standard = repeated_simann(rep, beta0, beta1, gamma0=0, gamma1=0)
            # dict["type"].append("standard")
            # dict["size"].append(size)
            # dict["iterations"].append(best_bp_standard.iterations_to_solution)
            # dict["time"].append(best_bp_standard.time_to_solution)
            # dict["num_replicas"].append(1)
            # print(f"{i+1}/{sample_size} standard ok - Solved? {best_cost_standard == 0}")

            assert np.all(rep_interacting.replicas[0].X == rep_non_interacting.replicas[0].X)
            assert np.all(rep_interacting.replicas[0].targets == rep_non_interacting.replicas[0].targets)
        
        print(f"Collection for size {size} is completed.")

    df = pd.DataFrame(dict)
    df.to_csv(path, index=False)
    print(f"\n CSV saved to: {path}")

def rsa_collect_size_comparison(size_limit, alpha, num_replicas, sample_size=10, path="data/rsa/comparison_size_data.csv"):

    sizes = list(range(100, size_limit+1, 50))
    print(sizes)

    dict = {"type": [], "size": [], "iterations": [], "time":[], "num_replicas": []}

    for size in sizes:
        print(f"Collecting for size {size}...")
        n = size
        P = int(alpha * n)
        annealing_steps = 1000 #max annealing steps
        mcmc_steps = 200  
        scooping_steps = 1000
        beta0 = .1
        beta1 = 5

        for i in range(sample_size):

            # INTERACTING #
            rep_interacting = RepeatedSimann(n, P, num_replicas, seed=i)
            best_bp_interacting, best_cost_interacting = repeated_simann(rep_interacting, beta0, beta1, gamma0=.6, gamma1=1.5, annealing_steps = annealing_steps, scooping_steps = scooping_steps, mcmc_steps = mcmc_steps)
            dict["type"].append("interacting")
            dict["size"].append(size)
            dict["iterations"].append(best_bp_interacting.iterations_to_solution)
            dict["time"].append(best_bp_interacting.time_to_solution)
            dict["num_replicas"].append(num_replicas)
            print(f"{i+1}/{sample_size} interacting ok")
            
            # NON-INTERACTING #
            rep_non_interacting = RepeatedSimann(n, P, num_replicas, seed=i)
            best_bp_nointeracting, best_cost_nointeracting = repeated_simann(rep_non_interacting, beta0, beta1, gamma0=0, gamma1=0, annealing_steps = annealing_steps, scooping_steps = scooping_steps, mcmc_steps = mcmc_steps)
            dict["type"].append("non_interacting")
            dict["size"].append(size)
            dict["iterations"].append(best_bp_nointeracting.iterations_to_solution)
            dict["time"].append(best_bp_nointeracting.time_to_solution)
            dict["num_replicas"].append(num_replicas)
            print(f"{i+1}/{sample_size} non-interacting ok")
            
            # # STANDARD #
            # rep = RepeatedSimann(n, P, num_replicas=1, seed=i)
            # best_bp_standard, best_cost_standard = repeated_simann(rep, beta0, beta1, gamma0=0, gamma1=0)
            # dict["type"].append("standard")
            # dict["size"].append(size)
            # dict["iterations"].append(best_bp_standard.iterations_to_solution)
            # dict["time"].append(best_bp_standard.time_to_solution)
            # dict["num_replicas"].append(1)
            # print(f"{i+1}/{sample_size} standard ok - Solved? {best_cost_standard == 0}")

            assert np.all(rep_interacting.replicas[0].X == rep_non_interacting.replicas[0].X)
            assert np.all(rep_interacting.replicas[0].targets == rep_non_interacting.replicas[0].targets)
        
        print(f"Collection for size {size} is completed.")

    df = pd.DataFrame(dict)
    df.to_csv(path, index=False)
    print(f"\n CSV saved to: {path}")


def rsa_collect_alpha_comparison(alpha_limit, size, num_replicas, sample_size=10, path="data/rsa/comparison_alpha_data.csv"):
    alphas = list(np.arange(0, alpha_limit, 0.1))
    dict = {"type": [], "alpha": [], "iterations": [], "time":[]}

    for alpha in alphas:
        n = size
        P = int(alpha * n)
        rep = RepeatedSimann(n, P, num_replicas)
        annealing_steps = 1000 #max annealing steps
        mcmc_steps = 200  
        scooping_steps = 1000
        beta0 = .1
        beta1 = 5

        for i in range(sample_size):
            # INTERACTING #
            rep_interacting = RepeatedSimann(n, P, num_replicas, seed=i)
            best_bp_interacting, best_cost_interacting = repeated_simann(rep_interacting, beta0, beta1, gamma0=.6, gamma1=1.5, annealing_steps = annealing_steps, scooping_steps = scooping_steps, mcmc_steps = mcmc_steps)
            dict["type"].append("interacting")
            dict["alpha"].append(alpha)
            dict["iterations"].append(best_bp_interacting.iterations_to_solution)
            dict["time"].append(best_bp_interacting.time_to_solution)
            
            # NON-INTERACTING #
            rep_non_interacting = RepeatedSimann(n, P, num_replicas, seed=i)
            best_bp_nointeracting, best_cost_nointeracting = repeated_simann(rep_non_interacting, beta0, beta1, gamma0=0, gamma1=0, annealing_steps = annealing_steps, scooping_steps = scooping_steps, mcmc_steps = mcmc_steps)
            dict["type"].append("non_interacting")
            dict["alpha"].append(alpha)
            dict["iterations"].append(best_bp_nointeracting.iterations_to_solution)
            dict["time"].append(best_bp_nointeracting.time_to_solution)
            
        print(f"Collection for alpha={alpha} is completed.")

    df = pd.DataFrame(dict)
    df.to_csv(path, index=False)
    print(f"\n CSV saved to: {path}")

if __name__ == "__main__":

    # size
    size_limit = 200
    alpha = 0.2
    num_replicas = 3
    rsa_collect_size_comparison(size_limit, alpha, num_replicas, sample_size=5, path="data/test/comparison_size_data.csv")

    # # alpha
    # alpha_limit = 0.4
    # size = 800
    # num_replicas = 3
    # rsa_collect_alpha_comparison(alpha_limit, size, num_replicas, sample_size=5, path="data/test/comparison_alpha_data.csv")
