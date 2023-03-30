from typing import List, Callable, Union, Tuple
import random as rand

Solution = List[float]
Population = List[Tuple[float, Solution]]


def increase_mutation(s: Solution) -> Solution:
    index = rand.randint(0, len(s) - 1)
    return [x + (i == index) for i, x in enumerate(s)]


def segmentation_crossing(s1: Solution, s2: Solution) -> Solution:
    size = rand.uniform(0, 1) < 0.5
    m = len(s1) >> 1
    if size:
        return s1[0: m] + s2[m:]

    return s2[0: m] + s1[m:]


def selection_process(count: int, p: Population) -> Population:
    while len(p) != count:
        a = rand.randint(0, int(len(p)/2 - 1))
        b = rand.randint(int(len(p)/2), len(p) - 1)

        (v1, _), (v2, _) = p[a], p[b]

        if v1 > v2:
            p.pop(b)
        else:
            p.pop(a)
    
    return p


def genetic_search(
    solution_sampler: Callable[[], Population],
    evaluate_solution: Callable[[Solution], Union[None, float]],
    mutation_function: Callable[[Solution], Solution] = increase_mutation,
    crossing_function: Callable[
        [Solution, Solution], Solution] = segmentation_crossing,
    selection_precess: Callable[
        [int, Population], Population] = selection_process,
    population_count: int = 500,
    epochs: int = 1000,
    optimization_direction=1,
    mutation_factor=1/2,
    epochs_tol: int = None
):

    population = []
    while len(population) != population_count:
        s = solution_sampler()
        if (v := evaluate_solution(s)):
            population.append((v, s))

    optimal = 0
    the_solution = None
    mutation_count = int(population_count * mutation_factor)
    crossing_count = population_count - mutation_count
    breaker = epochs_tol

    for i in range(epochs):
        new_generation = [x for x in population]

        for _, s in rand.choices(population, k=mutation_count):
            new_s = mutation_function(s)
            if (v := evaluate_solution(new_s)):
                new_generation.append((v, new_s))

        for _ in range(crossing_count):
            (_, s1), (_, s2) = rand.choices(population, k=2)
            new_s = crossing_function(s1, s2)
            if (v := evaluate_solution(new_s)):
                new_generation.append((v, new_s))

        population = selection_precess(population_count, new_generation)
        max_v, solution = max(population)

        if optimal < max_v * optimization_direction:
            optimal, the_solution = max_v, solution
            breaker = epochs_tol
        elif breaker:
            breaker -= 1

        if breaker == 0:
            break

        print(f'epochs: {i} \t optimal: {optimal} \t solution: {the_solution}')

    return optimal, the_solution
