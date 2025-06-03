from solutions.repeated_simann import repeated_simann, RepeatedSimann
import numpy as np
import pandas as pd


def rsa_collect_beta_comparison(beta_pairs, size, alpha, num_replicas, sample_size=10, path="data/rsa/comparison_beta_data.csv"):
    """
    Raccoglie dati per standard SA al variare dell'intervallo beta (beta0, beta1)
    
    Args:
        beta_pairs: Lista di tuple (beta0, beta1) da testare
        size: Dimensione del problema (n)
        alpha: Frazione di variabili da selezionare (P = alpha * n)
        num_replicas: Numero di repliche (non utilizzato per standard SA)
        sample_size: Numero di campioni per ogni configurazione
        path: Percorso dove salvare il CSV
    """
    
    dict_data = {"beta0": [], "beta1": [], "iterations": [], "time": []}
    
    n = size
    P = int(alpha * n)
    annealing_steps = 1000
    mcmc_steps = 200
    scooping_steps = 1000
    
    for beta0, beta1 in beta_pairs:
        print(f"Collecting for beta interval ({beta0}, {beta1})...")
        
        for i in range(sample_size):
            # STANDARD SA (num_replicas=1, gamma0=0, gamma1=0)
            rep = RepeatedSimann(n, P, num_replicas=1, seed=i)
            best_bp, best_cost = repeated_simann(
                rep, beta0, beta1, 
                gamma0=0, gamma1=0,
                annealing_steps=annealing_steps, 
                scooping_steps=scooping_steps, 
                mcmc_steps=mcmc_steps
            )
            
            dict_data["beta0"].append(beta0)
            dict_data["beta1"].append(beta1)
            dict_data["iterations"].append(best_bp.iterations_to_solution)
            dict_data["time"].append(best_bp.time_to_solution)
            
            print(f"{i+1}/{sample_size} completed for beta ({beta0}, {beta1})")
        
        print(f"Collection for beta interval ({beta0}, {beta1}) completed.")
    
    df = pd.DataFrame(dict_data)
    df.to_csv(path, index=False)
    print(f"\nCSV saved to: {path}")


def rsa_collect_gamma_comparison(gamma_pairs, size, alpha, num_replicas, sample_size=10, path="data/rsa/comparison_gamma_data.csv"):
    """
    Raccoglie dati per repeated SA al variare dell'intervallo gamma (gamma0, gamma1)
    
    Args:
        gamma_pairs: Lista di tuple (gamma0, gamma1) da testare
        size: Dimensione del problema (n)
        alpha: Frazione di variabili da selezionare (P = alpha * n)
        num_replicas: Numero di repliche
        sample_size: Numero di campioni per ogni configurazione
        path: Percorso dove salvare il CSV
    """
    
    dict_data = {"gamma0": [], "gamma1": [], "iterations": [], "time": [], "num_replicas": []}
    
    n = size
    P = int(alpha * n)
    annealing_steps = 1000
    mcmc_steps = 200
    scooping_steps = 1000
    beta0 = 0.1
    beta1 = 5.0
    
    for gamma0, gamma1 in gamma_pairs:
        print(f"Collecting for gamma interval ({gamma0}, {gamma1})...")
        
        for i in range(sample_size):
            # REPEATED SA con gamma variabile
            rep = RepeatedSimann(n, P, num_replicas, seed=i)
            best_bp, best_cost = repeated_simann(
                rep, beta0, beta1,
                gamma0=gamma0, gamma1=gamma1,
                annealing_steps=annealing_steps,
                scooping_steps=scooping_steps,
                mcmc_steps=mcmc_steps
            )
            
            dict_data["gamma0"].append(gamma0)
            dict_data["gamma1"].append(gamma1)
            dict_data["iterations"].append(best_bp.iterations_to_solution)
            dict_data["time"].append(best_bp.time_to_solution)
            dict_data["num_replicas"].append(num_replicas)
            
            print(f"{i+1}/{sample_size} completed for gamma ({gamma0}, {gamma1})")
        
        print(f"Collection for gamma interval ({gamma0}, {gamma1}) completed.")
    
    df = pd.DataFrame(dict_data)
    df.to_csv(path, index=False)
    print(f"\nCSV saved to: {path}")


def rsa_collect_annealing_steps_comparison(annealing_steps_list, size, alpha, sample_size=10, path="data/rsa/comparison_annealing_steps_data.csv"):
    """
    Raccoglie dati per standard SA al variare del numero di annealing steps
    
    Args:
        annealing_steps_list: Lista di valori di annealing_steps da testare
        size: Dimensione del problema (n)
        alpha: Frazione di variabili da selezionare (P = alpha * n)
        sample_size: Numero di campioni per ogni configurazione
        path: Percorso dove salvare il CSV
    """
    
    dict_data = {"annealing_steps": [], "iterations": [], "time": []}
    
    n = size
    P = int(alpha * n)
    mcmc_steps = 200
    scooping_steps = 1000
    beta0 = 0.1
    beta1 = 5.0
    
    for annealing_steps in annealing_steps_list:
        print(f"Collecting for annealing_steps = {annealing_steps}...")
        
        for i in range(sample_size):
            # STANDARD SA (num_replicas=1, gamma0=0, gamma1=0)
            rep = RepeatedSimann(n, P, num_replicas=1, seed=i)
            best_bp, best_cost = repeated_simann(
                rep, beta0, beta1,
                gamma0=0, gamma1=0,
                annealing_steps=annealing_steps,
                scooping_steps=scooping_steps,
                mcmc_steps=mcmc_steps
            )
            
            dict_data["annealing_steps"].append(annealing_steps)
            dict_data["iterations"].append(best_bp.iterations_to_solution)
            dict_data["time"].append(best_bp.time_to_solution)
            
            print(f"{i+1}/{sample_size} completed for annealing_steps = {annealing_steps}")
        
        print(f"Collection for annealing_steps = {annealing_steps} completed.")
    
    df = pd.DataFrame(dict_data)
    df.to_csv(path, index=False)
    print(f"\nCSV saved to: {path}")


def rsa_collect_mcmc_steps_comparison(mcmc_steps_list, size, alpha, sample_size=10, path="data/rsa/comparison_mcmc_steps_data.csv"):
    """
    Raccoglie dati per standard SA al variare del numero di mcmc steps
    
    Args:
        mcmc_steps_list: Lista di valori di mcmc_steps da testare
        size: Dimensione del problema (n)
        alpha: Frazione di variabili da selezionare (P = alpha * n)
        sample_size: Numero di campioni per ogni configurazione
        path: Percorso dove salvare il CSV
    """
    
    dict_data = {"mcmc_steps": [], "iterations": [], "time": []}
    
    n = size
    P = int(alpha * n)
    annealing_steps = 1000
    scooping_steps = 1000
    beta0 = 0.1
    beta1 = 5.0
    
    for mcmc_steps in mcmc_steps_list:
        print(f"Collecting for mcmc_steps = {mcmc_steps}...")
        
        for i in range(sample_size):
            # STANDARD SA (num_replicas=1, gamma0=0, gamma1=0)
            rep = RepeatedSimann(n, P, num_replicas=1, seed=i)
            best_bp, best_cost = repeated_simann(
                rep, beta0, beta1,
                gamma0=0, gamma1=0,
                annealing_steps=annealing_steps,
                scooping_steps=scooping_steps,
                mcmc_steps=mcmc_steps
            )
            
            dict_data["mcmc_steps"].append(mcmc_steps)
            dict_data["iterations"].append(best_bp.iterations_to_solution)
            dict_data["time"].append(best_bp.time_to_solution)
            
            print(f"{i+1}/{sample_size} completed for mcmc_steps = {mcmc_steps}")
        
        print(f"Collection for mcmc_steps = {mcmc_steps} completed.")
    
    df = pd.DataFrame(dict_data)
    df.to_csv(path, index=False)
    print(f"\nCSV saved to: {path}")


if __name__ == "__main__":
    
    # Parametri comuni
    size = 100
    alpha = 0.2
    num_replicas = 3
    sample_size = 5
    
    # Test 1: Variazione dell'intervallo beta per standard SA
    beta_pairs = [(0.05, 2.5), (0.1, 5.0), (0.2, 7.5), (0.3, 10.0)]
    rsa_collect_beta_comparison(
        beta_pairs, size, alpha, num_replicas, 
        sample_size=sample_size, 
        path="data/test/comparison_beta_data.csv"
    )
    
    # Test 2: Variazione dell'intervallo gamma per repeated SA
    gamma_pairs = [(0.0, 0.0), (0.3, 0.8), (0.6, 1.5), (0.9, 2.0), (1.2, 2.5)]
    rsa_collect_gamma_comparison(
        gamma_pairs, size, alpha, num_replicas,
        sample_size=sample_size,
        path="data/test/comparison_gamma_data.csv"
    )
    
    # Test 3: Variazione degli annealing steps per standard SA
    annealing_steps_list = [500, 750, 1000, 1250, 1500, 2000]
    rsa_collect_annealing_steps_comparison(
        annealing_steps_list, size, alpha,
        sample_size=sample_size,
        path="data/test/comparison_annealing_steps_data.csv"
    )
    
    # Test 4: Variazione degli mcmc steps per standard SA
    mcmc_steps_list = [50, 100, 150, 200, 250, 300, 400]
    rsa_collect_mcmc_steps_comparison(
        mcmc_steps_list, size, alpha,
        sample_size=sample_size,
        path="data/test/comparison_mcmc_steps_data.csv"
    )