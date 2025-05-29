from solutions.repeated_gd import replicated_gd
from solutions.repeated_gd import RepeatedGD
from solutions.repeated_gd import BinaryPerceptronGD
from problems.perceptron_repeated import BinaryPerceptronRepeated
from solutions.repeated_simann  import RepeatedSimann, repeated_simann

# # SIMULATED ANNEALING ##
# n = 3
# P = 3
# bp = BinaryPerceptronRepeated(n, P, seed=10)
# rep = RepeatedSimann(n = 200, P = 60, num_replicas = 3)
# annealing_steps = 1000
# mcmc_steps = 200  
# scooping_steps = 1000
# beta0 = .1
# beta1 = 5
# gamma0 = .6
# gamma1 = 1.5


# best_bp, best_cost = repeated_simann(rep, beta0, beta1, gamma0, gamma1, annealing_steps = annealing_steps, scooping_steps = scooping_steps, mcmc_steps = mcmc_steps, seed = 6, verbose=1)

# print(f"Final best cost = {best_cost}")


# ## REPLICATED GRADIENT DESCENT ##
gamma0 = .6
gamma1 = 1.5
beta = 0.5
rep = RepeatedGD(n = 5, P = 4, num_replicas = 2, seed=22)
best = replicated_gd(rep, lr=0.001, max_epochs=100, batch_size=1, 
                     gamma0=gamma0, gamma1=gamma1, beta=beta)
