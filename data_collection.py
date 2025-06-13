from src.comparison_gd import rgd_collect_size_comparison, rgd_collect_alpha_comparison
from src.comparison_sa import rsa_collect_size_comparison, rsa_collect_size_comparison_alt, rsa_collect_alpha_comparison
from src.parameters import save_extra_data_sa, save_extra_data_gd


# DATA COLLECTION FOR REPLICATED SIMULATED ANNEALING #
# hyperparameters comparison

# case 1: benchmark, with correct hyperparameters
N=250
P=75
num_replicas = 1
beta0 = .1
beta1 = 5
# save_extra_data_sa(N, P, num_replicas, beta0, beta1, 0, 0, path="data\\rsa\\extra_benchmark.csv", seed = 1)

# case 2: starting beta too low
beta0 = .01
beta1 = .5
# save_extra_data_sa(N, P, num_replicas, beta0, beta1, 0, 0, path="data\\rsa\\extra_exploration.csv", seed = 1)


# case 3: starting beta too high
N=250
P=75 # increase to make the landscape more difficult
beta0 = 9
beta1 = 10
# save_extra_data_sa(N, P, num_replicas, beta0, beta1, 0, 0, path="data\\rsa\\extra_optimization.csv", seed = 1)


# case 4: starting gamma too high
N=250
P=75 # increase to make the landscape more difficult
num_replicas = 3
beta0 = .1
beta1 = 5
gamma0 = 1
gamma1 = 2
# save_extra_data_sa(N, P, num_replicas, beta0, beta1, gamma0, gamma1, path="data\\rsa\\extra_narrow.csv", seed = 1)


# case 5: starting gamma too low
N=250
P=75 # increase to make the landscape more difficult
num_replicas = 3
beta0 = .1
beta1 = 5
gamma0 = 0.0001
gamma1 = 0.0002
# save_extra_data_sa(N, P, num_replicas, beta0, beta1, gamma0, gamma1, path="data\\rsa\\extra_wide.csv", seed = 1)


# comparison by size
size_limit = 110
alpha = 0.3
num_replicas = 3
# rsa_collect_size_comparison(size_limit, alpha, num_replicas, sample_size=10, path="data/rsa/comparison_size_data1.csv")


# GRADIENT DESCENT #
# case 1: lr benchmark
N=1100
P=440
num_replicas = 1
lr=0.01
# save_extra_data_gd(N, P, num_replicas, lr, path="data\\rgd\\extra_benchmark.csv", seed = 2)


# case 2: lr too high
N=1100
P=440
num_replicas = 1
lr=10
# save_extra_data_gd(N, P, num_replicas, lr, path="data\\rgd\\extra_high.csv", seed = 2)


# case 3: lr too low
N=1100
P=440
num_replicas = 1
lr=0.00001
# save_extra_data_gd(N, P, num_replicas, lr, path="data\\rgd\\extra_low.csv", seed = 2)



# size
size_limit = 1600
alpha = 0.4
num_replicas = 3
# rgd_collect_size_comparison(size_limit, alpha, num_replicas, sample_size=10, batch_size=10, path="data/rgd/comparison_size_data.csv")

# alpha
alpha_limit = 0.7
size = 500
num_replicas = 3
rgd_collect_alpha_comparison(alpha_limit, size, num_replicas, sample_size=10, batch_size=10, path="data/rgd/comparison_alpha_data.csv")

