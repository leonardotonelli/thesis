from solutions.simann import simann
from problems.repeatedv2 import BinaryPerceptronRepeated
from solutions.repeated_simann import RepeatedSimann, repeated_simann


# n = 3
# P = 3
# bp = BinaryPerceptronRepeated(n, P, seed=10)
rep = RepeatedSimann(n = 101, P = 30, num_replicas = 5)
annealing_steps = 1000
mcmc_steps = 100
scooping_steps = 1000
beta0 = .1
beta1 = 5
gamma0 = .6
gamma1 = 1.5


best_bp, best_cost = repeated_simann(rep, beta0, beta1, gamma0, gamma1, annealing_steps = annealing_steps, scooping_steps = scooping_steps, mcmc_steps = mcmc_steps, seed = 6)

print(f"Final best cost = {best_cost}")