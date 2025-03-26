from solutions.simann import simann
from problems.repeatedv2 import BinaryPerceptronRepeated
from solutions.repeated_simann import RepeatedSimann, repeated_simann


# n = 3
# P = 3
# bp = BinaryPerceptronRepeated(n, P, seed=10)
rep = RepeatedSimann(n = 100, P = 70, num_replicas = 10)
annealing_steps = 1000
mcmc_steps = 500
scooping_steps = 1000
beta0 = .01
beta1 = 5
gamma0 = .01
gamma1 = 100


best_bp, best_cost = repeated_simann(rep, beta0, beta1, gamma0, gamma1, annealing_steps = annealing_steps, scooping_steps = scooping_steps, mcmc_steps = mcmc_steps, seed = 5)

print(f"Final best cost = {best_cost}")