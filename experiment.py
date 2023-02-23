import random

import numpy as np

from fitness import CountingOne, Trap
from mutation import UniformCrossover, TwoPointCrossover
from structures import Population


# TODOs: -test stuff; -tracking info
# then run!


def found_global_optimum(population):
    return np.any([0 not in c for c in population])


def stopping_criteria(population, generation_info):
    optimum = found_global_optimum(population)
    has_improvement = population.generation >= generation_info[0] + 10 # not improved in previous 10 gens

    return optimum or has_improvement


def best_fitness(population, fitness):
    return np.max([fitness.score(c) for c in population])


def run_experiment(fitness, mutation, population_size=10, solution_length=40):

    current_population_size    = population_size
    lowerbound_population_size = population_size
    upperbound_population_size = None

    optimum_found = False

    while current_population_size <= 1280:
        current_population = Population(current_population_size, solution_length, generation=0)
        generation_info = [0, best_fitness(current_population, fitness)] # (best gen, best fitness)

        while not stopping_criteria(current_population, generation_info):
            # one generation:
            random.shuffle(current_population)                                       # shuffle
            parents = list(zip(current_population[:-1:2], current_population[1::2])) # pairing
            families = [(*p, *mutation.mutate(*p)) for p in parents]                 # parents + offspring 
            next_pop = [sorted(f, key= fitness.score)[:-2] for f in families]        # fight to the death
            # Note: `sorted()` sorts in-order, meaning children will remain at the end of the list
            next_pop = [item for sublist in next_pop for item in sublist]
# CHECK THIS


            next_generation = Population(current_population_size, solution_length, generation=current_population.generation + 1, x=next_pop)
            next_fitness = best_fitness(next_generation, fitness)

            # if new generation better than previous ones
            if next_fitness > generation_info[1]:
                generation_info[0] = next_generation.generation
                generation_info[1] = next_fitness

            current_population = next_generation

        optimum_found = found_global_optimum(current_population)
        print(generation_info)

        # Bisection Search
        if optimum_found:
            upperbound_population_size = current_population_size
            current_population_size = (current_population_size + lowerbound_population_size) / 2
        else:
            lowerbound_population_size = current_population_size
            if upperbound_population_size == None:
                current_population_size *= 2 # instead of binary search (0 -> 640), bisection search special case
            else:
                current_population_size = (current_population_size + upperbound_population_size) / 2

        if upperbound_population_size - upperbound_population_size <= population_size:
            break

    return upperbound_population_size


if __name__ == "__main__":

    print(f"Counting ones and Uniform crossover")
    for _ in range(20):
        fitness = CountingOne()
        mutation = UniformCrossover()

        print(run_experiment(fitness, mutation))

    print(f"Counting ones and Two point crossover")
    for _ in range(20):
        fitness = CountingOne()
        mutation = TwoPointCrossover()

        print(run_experiment(fitness, mutation))

    print(f"Trap deceptive tightly linked and Uniform crossover")
    for _ in range(20):
        fitness = Trap(k=4, d=1, tightly_linked=True)
        mutation = UniformCrossover()

        print(run_experiment(fitness, mutation))

    print(f"Trap deceptive tightly linked and Two point crossover")
    for _ in range(20):
        fitness = Trap(k=4, d=1, tightly_linked=True)
        mutation = TwoPointCrossover()

        print(run_experiment(fitness, mutation))

    print(f"Trap non-deceptive tightly linked and Uniform crossover")
    for _ in range(20):
        fitness = Trap(k=4, d=2.5, tightly_linked=True)
        mutation = UniformCrossover()

        print(run_experiment(fitness, mutation))

    print(f"Trap non-deceptive tightly linked and Two point crossover")
    for _ in range(20):
        fitness = Trap(k=4, d=2.5, tightly_linked=True)
        mutation = TwoPointCrossover()

        print(run_experiment(fitness, mutation))

    print(f"Trap deceptive not tightly linked and Uniform crossover")
    for _ in range(20):
        fitness = Trap(k=4, d=1, tightly_linked=False)
        mutation = UniformCrossover()

        print(run_experiment(fitness, mutation))

    print(f"Trap deceptive not tightly linked and Two point crossover")
    for _ in range(20):
        fitness =  Trap(k=4, d=1, tightly_linked=False)
        mutation = TwoPointCrossover()

        print(run_experiment(fitness, mutation))

    print(f"Trap non-deceptive not tightly linked and Uniform crossover")
    for _ in range(20):
        fitness = Trap(k=4, d=2.5, tightly_linked=False)
        mutation = UniformCrossover()

        print(run_experiment(fitness, mutation))

    print(f"Trap non-deceptive not tightly linked and Two point crossover")
    for _ in range(20):
        fitness = Trap(k=4, d=2.5, tightly_linked=False)
        mutation = TwoPointCrossover()

        print(run_experiment(fitness, mutation))


