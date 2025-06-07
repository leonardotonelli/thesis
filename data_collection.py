from src.comparison_gd import rgd_collect_size_comparison, rgd_collect_alpha_comparison, rgd_collect_replicas_comparison
from src.comparison_sa import rsa_collect_size_comparison, rsa_collect_alpha_comparison, rsa_collect_replicas_comparison
from src.parameters_sa import rsa_collect_beta_comparison, rsa_collect_gamma_comparison, rsa_collect_annealing_steps_comparison, rsa_collect_mcmc_steps_comparison
from src.parameters_gd import collect_learning_rate_comparison, collect_gamma_interval_comparison, collect_batch_size_comparison

# DATA COLLECTION FOR REPLICATED SIMULATED ANNEALING #
# Hyperparameters
# size = 50
# alpha = 0.2
# num_replicas = 3
# sample_size = 5

# # Test 1: Variazione dell'intervallo beta per standard SA
# beta_pairs = [(0.05, 2.5), (0.1, 5.0), (0.2, 7.5), (0.3, 10.0)]
# rsa_collect_beta_comparison(
#     beta_pairs, size, alpha, num_replicas, 
#     sample_size=sample_size, 
#     path="data/test/comparison_beta_data.csv"
# )

# # Test 2: Variazione dell'intervallo gamma per repeated SA
# gamma_pairs = [(0.0, 0.0), (0.3, 0.8), (0.6, 1.5), (0.9, 2.0), (1.2, 2.5)]
# rsa_collect_gamma_comparison(
#     gamma_pairs, size, alpha, num_replicas,
#     sample_size=sample_size,
#     path="data/test/comparison_gamma_data.csv"
# )

# # Test 3: Variazione degli annealing steps per standard SA
# annealing_steps_list = [500, 750, 1000, 1250, 1500, 2000]
# rsa_collect_annealing_steps_comparison(
#     annealing_steps_list, size, alpha,
#     sample_size=sample_size,
#     path="data/test/comparison_annealing_steps_data.csv"
# )

# # Test 4: Variazione degli mcmc steps per standard SA
# mcmc_steps_list = [50, 100, 150, 200, 250, 300, 400]
# rsa_collect_mcmc_steps_comparison(
#     mcmc_steps_list, size, alpha,
#     sample_size=sample_size,
#     path="data/test/comparison_mcmc_steps_data.csv"
# )

# # # size
# size_limit = 100
# alpha = 0.2
# num_replicas = 3
# rsa_collect_size_comparison(size_limit, alpha, num_replicas, sample_size=5, path="data/comparison_size_data.csv")

# alpha
alpha_limit = 0.8
size = 100
num_replicas = 3
rsa_collect_alpha_comparison(alpha_limit, size, num_replicas, sample_size=5, path="data/comparison_alpha_data.csv")

# # replica
# replicas_limit = 10
# size = 800
# alpha = 0.2
# rsa_collect_replicas_comparison(alpha_limit, size, num_replicas, sample_size=5, path="data/comparison_replicas_data.csv")



# DATA COLLECTION FOR REPLICATED GRADIENT DESCENT #

# Hyperparameters
# Test delle funzioni

# # 1. Learning Rate Comparison (Standard GD)
# print("=== LEARNING RATE COMPARISON ===")
# lr_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
# size = 50
# alpha = 0.2
# collect_learning_rate_comparison(lr_values, size, alpha, sample_size=5, batch_size=10, 
#                                 path="data/test/comparison_lr_data.csv")

# # 2. Gamma Interval Comparison (Repeated GD)
# print("\n=== GAMMA INTERVAL COMPARISON ===")
# gamma_configs = [
#     (0.0, 0.5),   # intervallo 0.5
#     (0.0, 1.0),   # intervallo 1.0
#     (0.01, 1.0),  # intervallo 0.99
#     (0.0, 2.0),   # intervallo 2.0
#     (0.1, 1.5),   # intervallo 1.4
#     (0.05, 0.8),  # intervallo 0.75
#     (0.02, 1.2)   # intervallo 1.18
# ]
# size = 500
# alpha = 0.2
# num_replicas = 3
# collect_gamma_interval_comparison(gamma_configs, size, alpha, num_replicas, sample_size=5, 
#                                 batch_size=10, path="data/test/comparison_gamma_data.csv")

# # 3. Batch Size Comparison (Standard GD)
# print("\n=== BATCH SIZE COMPARISON ===")
# batch_sizes = [1, 5, 10, 20, 50, 100, 200]
# size = 500
# alpha = 0.2
# collect_batch_size_comparison(batch_sizes, size, alpha, sample_size=5,
#                             path="data/test/comparison_batch_data.csv")

# print("\nTutti i test completati!")


# # size
# size_limit = 500
# alpha = 0.3
# num_replicas = 3
# rgd_collect_size_comparison(size_limit, alpha, num_replicas, sample_size=50, batch_size=10, path="data/rgd/comparison_size_data.csv")

# # alpha
# alpha_limit = 0.4
# size = 800
# num_replicas = 3
# rgd_collect_alpha_comparison(alpha_limit, size, num_replicas, sample_size=10, batch_size=10, path="data/rgd/comparison_alpha_data.csv")

# # replica
# replicas_limit = 10
# size = 800
# alpha = 0.2
# rgd_collect_replicas_comparison(replicas_limit, size, alpha, sample_size=10, batch_size=10, path="data/rgd/comparison_replicas_data.csv")
