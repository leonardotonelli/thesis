from solutions.repeated_gd import replicated_gd, RepeatedGD
import numpy as np
import pandas as pd

# GRADIENT DESCENT #
def rgd_collect_size_comparison(size_limit, alpha, num_replicas, sample_size=10, batch_size=10, path="data/rgd/comparison_size_data.csv"):

    sizes = list(range(10, size_limit, 20))
    print(sizes)

    dict = {"type": [], "size": [], "epochs": [], "error_rate": []}

    for size in sizes:
        print(f"Collecting for size {size}...")
        n = size
        P = int(alpha * n)
        gamma0 = 0.01
        gamma1 = 1
        beta = 5
        lr=0.01
        max_epochs = 1000

        for i in range(sample_size):

            # INTERACTING #
            rep_interacting = RepeatedGD(n, P, num_replicas, seed=i)
            best_bp_interacting = replicated_gd(rep_interacting, lr=lr, max_epochs=max_epochs, batch_size=batch_size, gamma0=gamma0, gamma1=gamma1, beta=beta)
            dict["type"].append("interacting")
            dict["size"].append(size)
            dict["epochs"].append(best_bp_interacting.epochs)
            dict["error_rate"].append(best_bp_interacting.error_rate)
            print(f"{i+1}/{sample_size} interacting ok")
            
            # NON-INTERACTING #
            rep_non_interacting = RepeatedGD(n, P, num_replicas, seed=i)
            best_bp_nointeracting = replicated_gd(rep_non_interacting, lr=lr, max_epochs=max_epochs, batch_size=batch_size, gamma0=0, gamma1=0, beta=beta)
            dict["type"].append("non_interacting")
            dict["size"].append(size)
            dict["epochs"].append(best_bp_nointeracting.epochs)
            dict["error_rate"].append(best_bp_nointeracting.error_rate)
            print(f"{i+1}/{sample_size} non-interacting ok")
            
            # STANDARD #
            rep = RepeatedGD(n, P, 1, seed=i)
            best_bp_standard = replicated_gd(rep, lr=lr, max_epochs=max_epochs, batch_size=batch_size, gamma0=0, gamma1=0, beta=beta)
            dict["type"].append("standard")
            dict["size"].append(size)
            dict["epochs"].append(best_bp_standard.epochs)
            dict["error_rate"].append(best_bp_standard.error_rate)
            print(f"{i+1}/{sample_size} standard ok")

            assert np.all(rep_interacting.replicas[0].X == rep_non_interacting.replicas[0].X) and \
                np.all(rep_non_interacting.replicas[0].X == rep.replicas[0].X)

            assert np.all(rep_interacting.replicas[0].targets == rep_non_interacting.replicas[0].targets) and \
                np.all(rep_non_interacting.replicas[0].targets == rep.replicas[0].targets)
 
        print(f"Collection for size {size} is completed.")

    df = pd.DataFrame(dict)
    df.to_csv(path, index=False)
    print(f"\n CSV saved to: {path}")


def rgd_collect_alpha_comparison(alpha_limit, size, num_replicas, sample_size=10, batch_size=10, path="data/rgd/comparison_alpha_data.csv"):
    alphas = list(range(0, alpha_limit, 0.1))
    dict = {"type": [], "alpha": [], "epochs": [], "error_rate": []}

    for alpha in alphas:
        n = size
        P = int(alpha * n)
        gamma0 = 0.01
        gamma1 = 1
        beta = 5
        lr=0.01
        max_epochs = 1000

        for i in range(sample_size):

            # INTERACTING #
            rep_interacting = RepeatedGD(n, P, num_replicas, seed=i)
            best_bp_interacting = replicated_gd(rep_interacting, lr=lr, max_epochs=max_epochs, batch_size=batch_size, gamma0=gamma0, gamma1=gamma1, beta=beta)
            dict["type"].append("interacting")
            dict["alpha"].append(alpha)
            dict["epochs"].append(best_bp_interacting.epochs)
            dict["error_rate"].append(best_bp_interacting.error_rate)
            print(f"{i+1}/{sample_size} interacting ok")
            
            # NON-INTERACTING #
            rep_non_interacting = RepeatedGD(n, P, num_replicas, seed=i)
            best_bp_nointeracting = replicated_gd(rep_non_interacting, lr=lr, max_epochs=max_epochs, batch_size=batch_size, gamma0=0, gamma1=0, beta=beta)
            dict["type"].append("non_interacting")
            dict["alpha"].append(alpha)
            dict["epochs"].append(best_bp_nointeracting.epochs)
            dict["error_rate"].append(best_bp_nointeracting.error_rate)
            print(f"{i+1}/{sample_size} non-interacting ok")
            
            # STANDARD #
            rep = RepeatedGD(n, P, 1, seed=i)
            best_bp_standard = replicated_gd(rep, lr=lr, max_epochs=max_epochs, batch_size=batch_size, gamma0=0, gamma1=0, beta=beta)
            dict["type"].append("standard")
            dict["alpha"].append(alpha)
            dict["epochs"].append(best_bp_standard.epochs)
            dict["error_rate"].append(best_bp_standard.error_rate)
            print(f"{i+1}/{sample_size} standard ok")

            assert np.all(rep_interacting.replicas[0].X == rep_non_interacting.replicas[0].X) and \
                np.all(rep_non_interacting.replicas[0].X == rep.replicas[0].X)

            assert np.all(rep_interacting.replicas[0].targets == rep_non_interacting.replicas[0].targets) and \
                np.all(rep_non_interacting.replicas[0].targets == rep.replicas[0].targets)
        
        print(f"Collection for alpha={alpha} is completed.")

    df = pd.DataFrame(dict)
    df.to_csv(path, index=False)
    print(f"\n CSV saved to: {path}")


def rgd_collect_replicas_comparison(replicas_limit, size, alpha, sample_size=10, batch_size=10, path="data/rgd/comparison_replicas_data.csv"):
    replicas = list(range(replicas_limit))
    dict = {"type": [], "num_replicas": [], "iterations": [], "error_rate":[]}

    for replica_num in replicas:
        n = size
        P = int(alpha * n)
        gamma0 = 0.01
        gamma1 = 1
        beta = 5
        lr=0.01
        max_epochs = 1000

        for i in range(sample_size):
            # INTERACTING #
            rep_interacting = RepeatedGD(n, P, replica_num, seed=i)
            best_bp_interacting = replicated_gd(rep_interacting, lr=lr, max_epochs=max_epochs, batch_size=batch_size, gamma0=gamma0, gamma1=gamma1, beta=beta)
            dict["type"].append("interacting")
            dict["num_replicas"].append(replica_num)
            dict["epochs"].append(best_bp_interacting.epochs)
            dict["error_rate"].append(best_bp_interacting.error_rate)
            print(f"{i+1}/{sample_size} interacting ok")
            
            # NON-INTERACTING #
            rep_non_interacting = RepeatedGD(n, P, replica_num, seed=i)
            best_bp_nointeracting = replicated_gd(rep_non_interacting, lr=lr, max_epochs=max_epochs, batch_size=batch_size, gamma0=0, gamma1=0, beta=beta)
            dict["type"].append("non_interacting")
            dict["num_replicas"].append(replica_num)
            dict["epochs"].append(best_bp_nointeracting.epochs)
            dict["error_rate"].append(best_bp_nointeracting.error_rate)
            print(f"{i+1}/{sample_size} non-interacting ok")

            assert np.all(rep_interacting.replicas[0].X == rep_non_interacting.replicas[0].X)
            assert np.all(rep_interacting.replicas[0].targets == rep_non_interacting.replicas[0].targets)
        
        print(f"Collection for replicas={replica_num} is completed.")

    df = pd.DataFrame(dict)
    df.to_csv(path, index=False)
    print(f"\n CSV saved to: {path}")



if __name__ == "__main__":
    #tests
    
    # size
    size_limit = 200
    alpha = 0.2
    num_replicas = 3
    rgd_collect_size_comparison(size_limit, alpha, num_replicas, sample_size=10, batch_size=10, path="data/test/comparison_size_data.csv")

    # alpha
    alpha_limit = 0.4
    size = 800
    num_replicas = 3
    rgd_collect_alpha_comparison(alpha_limit, size, num_replicas, sample_size=10, batch_size=10, path="data/test/comparison_alpha_data.csv")

    # replica
    replicas_limit = 10
    size = 800
    alpha = 0.2
    rgd_collect_replicas_comparison(replicas_limit, size, alpha, sample_size=10, batch_size=10, path="data/test/comparison_replicas_data.csv")