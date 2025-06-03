from solutions.repeated_gd import replicated_gd, RepeatedGD
import numpy as np
import pandas as pd


def collect_learning_rate_comparison(lr_values, size, alpha, sample_size=10, batch_size=10, path="data/sgd/comparison_lr_data.csv"):
    """
    Raccoglie dati (iterations e error_rate) al cambio del learning rate per standard gradient descent
    
    Args:
        lr_values: lista di valori di learning rate da testare
        size: dimensione del problema (n)
        alpha: rapporto P/n per determinare P
        sample_size: numero di campioni per ogni valore di lr
        batch_size: dimensione del batch
        path: percorso dove salvare il file CSV
    """
    
    dict = {"learning_rate": [], "epochs": [], "error_rate": []}
    
    for lr in lr_values:
        print(f"Collecting for learning rate {lr}...")
        n = size
        P = int(alpha * n)
        max_epochs = 1000
        
        for i in range(sample_size):
            # STANDARD GRADIENT DESCENT (1 replica, no interaction)
            rep = RepeatedGD(n, P, 1, seed=i)
            best_bp = replicated_gd(rep, lr=lr, max_epochs=max_epochs, batch_size=batch_size, 
                                  gamma0=0, gamma1=0, beta=5)
            
            dict["learning_rate"].append(lr)
            dict["epochs"].append(best_bp.epochs)
            dict["error_rate"].append(best_bp.error_rate)
            print(f"{i+1}/{sample_size} lr={lr} completed")
        
        print(f"Collection for learning rate {lr} is completed.")
    
    df = pd.DataFrame(dict)
    df.to_csv(path, index=False)
    print(f"\nCSV saved to: {path}")


def collect_gamma_interval_comparison(gamma_configs, size, alpha, num_replicas, sample_size=10, batch_size=10, path="data/rgd/comparison_gamma_data.csv"):
    """
    Raccoglie dati (iterations e error_rate) al cambio dell'intervallo gamma per repeated gradient descent
    
    Args:
        gamma_configs: lista di tuple (gamma0, gamma1) da testare
        size: dimensione del problema (n)
        alpha: rapporto P/n per determinare P
        num_replicas: numero di repliche
        sample_size: numero di campioni per ogni configurazione gamma
        batch_size: dimensione del batch
        path: percorso dove salvare il file CSV
    """
    
    dict = {"gamma0": [], "gamma1": [], "gamma_interval": [], "epochs": [], "error_rate": []}
    
    for gamma0, gamma1 in gamma_configs:
        gamma_interval = gamma1 - gamma0
        print(f"Collecting for gamma interval [{gamma0}, {gamma1}] (interval: {gamma_interval})...")
        
        n = size
        P = int(alpha * n)
        lr = 0.01
        beta = 5
        max_epochs = 1000
        
        for i in range(sample_size):
            # INTERACTING REPEATED GRADIENT DESCENT
            rep_interacting = RepeatedGD(n, P, num_replicas, seed=i)
            best_bp = replicated_gd(rep_interacting, lr=lr, max_epochs=max_epochs, batch_size=batch_size,
                                  gamma0=gamma0, gamma1=gamma1, beta=beta)
            
            dict["gamma0"].append(gamma0)
            dict["gamma1"].append(gamma1)
            dict["gamma_interval"].append(gamma_interval)
            dict["epochs"].append(best_bp.epochs)
            dict["error_rate"].append(best_bp.error_rate)
            print(f"{i+1}/{sample_size} gamma=[{gamma0}, {gamma1}] completed")
        
        print(f"Collection for gamma interval [{gamma0}, {gamma1}] is completed.")
    
    df = pd.DataFrame(dict)
    df.to_csv(path, index=False)
    print(f"\nCSV saved to: {path}")


def collect_batch_size_comparison(batch_sizes, size, alpha, sample_size=10, path="data/sgd/comparison_batch_data.csv"):
    """
    Raccoglie dati (iterations e error_rate) al cambio di batch size per standard gradient descent
    
    Args:
        batch_sizes: lista di dimensioni di batch da testare
        size: dimensione del problema (n)
        alpha: rapporto P/n per determinare P
        sample_size: numero di campioni per ogni batch size
        path: percorso dove salvare il file CSV
    """
    
    dict = {"batch_size": [], "epochs": [], "error_rate": []}
    
    for batch_size in batch_sizes:
        print(f"Collecting for batch size {batch_size}...")
        n = size
        P = int(alpha * n)
        lr = 0.01
        max_epochs = 1000
        
        for i in range(sample_size):
            # STANDARD GRADIENT DESCENT (1 replica, no interaction)
            rep = RepeatedGD(n, P, 1, seed=i)
            best_bp = replicated_gd(rep, lr=lr, max_epochs=max_epochs, batch_size=batch_size,
                                  gamma0=0, gamma1=0, beta=5)
            
            dict["batch_size"].append(batch_size)
            dict["epochs"].append(best_bp.epochs)
            dict["error_rate"].append(best_bp.error_rate)
            print(f"{i+1}/{sample_size} batch_size={batch_size} completed")
        
        print(f"Collection for batch size {batch_size} is completed.")
    
    df = pd.DataFrame(dict)
    df.to_csv(path, index=False)
    print(f"\nCSV saved to: {path}")


if __name__ == "__main__":
    # Test delle funzioni
    
    # 1. Learning Rate Comparison (Standard GD)
    print("=== LEARNING RATE COMPARISON ===")
    lr_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    size = 500
    alpha = 0.2
    collect_learning_rate_comparison(lr_values, size, alpha, sample_size=5, batch_size=10, 
                                   path="data/test/comparison_lr_data.csv")
    
    # 2. Gamma Interval Comparison (Repeated GD)
    print("\n=== GAMMA INTERVAL COMPARISON ===")
    gamma_configs = [
        (0.0, 0.5),   # intervallo 0.5
        (0.0, 1.0),   # intervallo 1.0
        (0.01, 1.0),  # intervallo 0.99
        (0.0, 2.0),   # intervallo 2.0
        (0.1, 1.5),   # intervallo 1.4
        (0.05, 0.8),  # intervallo 0.75
        (0.02, 1.2)   # intervallo 1.18
    ]
    size = 500
    alpha = 0.2
    num_replicas = 3
    collect_gamma_interval_comparison(gamma_configs, size, alpha, num_replicas, sample_size=5, 
                                    batch_size=10, path="data/test/comparison_gamma_data.csv")
    
    # 3. Batch Size Comparison (Standard GD)
    print("\n=== BATCH SIZE COMPARISON ===")
    batch_sizes = [1, 5, 10, 20, 50, 100, 200]
    size = 500
    alpha = 0.2
    collect_batch_size_comparison(batch_sizes, size, alpha, sample_size=5,
                                path="data/test/comparison_batch_data.csv")
    
    print("\nTutti i test completati!")