from solutions.simann import simann
# from problems.basic_binary_perceptron import BasicBinaryPerceptron
# from problems.second_basic_perceptron import BinaryPerceptron
from problems.v3 import BinaryPerceptron

n = 3
P = 3
bp = BinaryPerceptron(n, P, seed=10)
annealing_steps = 10
mcmc_steps = 50
beta0 = .01
beta1 = 5

best_bp, best_cost = simann(bp, beta0=beta0, beta1=beta1, annealing_steps=annealing_steps, mcmc_steps = mcmc_steps, seed=11)

print(f"Final best cost = {best_cost}")