from solutions.simann import simann
from problems.basic_binary_perceptron import BasicBinaryPerceptron

n = 10
P = 5
bp = BasicBinaryPerceptron(n, P, seed=10)
best_bp, best_cost = simann(bp, beta0=5., beta1=10.,annealing_steps=1000, mcmc_steps = 1000, seed=11)

print(f"Final best cost = {best_cost}")