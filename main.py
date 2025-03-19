from solutions.simann import simann
from problems.basic_binary_perceptron import BasicBinaryPerceptron
from problems.second_basic_perceptron import BinaryPerceptron

n = 5
P = 5
bp = BinaryPerceptron(n, P, seed=10)
annealing_steps = 100
mcmc_steps = 100
beta0 = 1
beta1 = 2

best_bp, best_cost = simann(bp, beta0=beta0, beta1=beta1, annealing_steps=annealing_steps, mcmc_steps = mcmc_steps, seed=11)

print(f"Final best cost = {best_cost}")