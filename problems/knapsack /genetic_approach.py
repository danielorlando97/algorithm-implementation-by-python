# %%
import sys
sys.path.append('../../')

from src.heuristics import genetic_algorithm as gea
import random
from functools import partial

# %%

class Knapsack:
    def __init__(self) -> None:
        self.W = random.randint(200, 500)
        self.V = random.randint(200, 500)
        self.n = random.randint(100, 200)
        self.weight = [random.uniform(1, self.W - 1) for _ in range(self.n)]
        self.volumes = [random.uniform(1, self.V - 1) for _ in range(self.n)]
        self.prices = [random.uniform(1, 10) for _ in range(self.n)]

# %%

def sampler_solution(bag: Knapsack):
    W_temp, V_temp = bag.W, bag.V
    result = [0] * bag.n
    iter_ = list(range(bag.n))
    random.shuffle(iter_)
    for i in iter_:
        result[i] = random.randint(
            0, int(min(W_temp/bag.weight[i], V_temp/bag.volumes[i]))
        )

        W_temp -= result[i] * bag.weight[i]
        V_temp -= result[i] * bag.volumes[i]

    return result

def evaluate_solution(bag: Knapsack, solution):
    V = sum([v * c for c, v in zip(solution, bag.volumes)])
    W = sum([w * c for c, w in zip(solution, bag.weight)])

    if W > bag.W or V > bag.V:
        return None
    
    return sum([p * c for c, p in zip(solution, bag.prices)])

# %%
problem = Knapsack()

#%%
v, s = gea.genetic_search(
    solution_sampler=partial(sampler_solution, problem),
    evaluate_solution=partial(evaluate_solution, problem),
    epochs_tol= 200,
    population_count=1000,
    epochs=2000
)
# %%
