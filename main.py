from solutions.simann import simann
from problems.basic_binary_perceptron import BasicBinaryPerceptron
# from problems.second_basic_perceptron import BinaryPerceptron
from problems.third import BinaryPerceptron

n = 20
P = 20
bp = BinaryPerceptron(n, P, seed=10)
annealing_steps = 1000
mcmc_steps = 1000
beta0 = .01
beta1 = 10

best_bp, best_cost = simann(bp, beta0=beta0, beta1=beta1, annealing_steps=annealing_steps, mcmc_steps = mcmc_steps, seed=11)

print(f"Final best cost = {best_cost}")