from solutions.simann import simann
from problems.basic_binary_perceptron import BasicBinaryPerceptron
from problems.second_basic_perceptron import BinaryPerceptron

n = 20
P = 20
bp = BinaryPerceptron(n, P, seed=10)
best_bp, best_cost = simann(bp, beta0=1, beta1=2.,annealing_steps=10, mcmc_steps = 10, seed=11)

print(f"Final best cost = {best_cost}")