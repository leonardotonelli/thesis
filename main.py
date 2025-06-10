from solutions.repeated_gd import replicated_gd, RepeatedGD
from solutions.repeated_simann_alt2  import RepeatedSimann_alt, repeated_simann_alt
from solutions.repeated_simann_alternative  import RepeatedSimann_alt, repeated_simann_alt
from solutions.repeated_simann  import RepeatedSimann, repeated_simann

# SIMULATED ANNEALING ##
rep = RepeatedSimann_alt(n = 100, P = 30, num_replicas = 3)
annealing_steps = 1000
mcmc_steps = 200
scooping_steps = 1000
beta0 = .1
beta1 = 5
gamma0 = 0.6
gamma1 = 1.5
best_bp, best_cost = repeated_simann(rep, beta0, beta1, gamma0, gamma1, annealing_steps = annealing_steps, scooping_steps = scooping_steps, mcmc_steps = mcmc_steps, seed = 6, verbose=1)

# print(f"Final best cost = {best_cost}")


# # ## REPLICATED GRADIENT DESCENT ##
# gamma0 = 0.01
# gamma1 = 1
# beta = 5
# rep = RepeatedGD(n = 1605, P = 802, num_replicas = 3, seed=4)
# best = replicated_gd(rep, lr=0.01, max_epochs=1000, batch_size=10, 
#                      gamma0=gamma0, gamma1=gamma1, beta=beta, verbose=1)
