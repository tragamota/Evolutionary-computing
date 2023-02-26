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


def run_generation(population_size, solution_length, fitness, mutation):
    current_population = Population(population_size, solution_length, generation=0)
    generation_info = [0, best_fitness(current_population, fitness)]  # (best gen, best fitness)

    while not stopping_criteria(current_population, generation_info):
        # one generation:
        random.shuffle(current_population)
        parents = list(zip(current_population[:-1:2], current_population[1::2]))
        families = [(p, mutation.mutate(*p)) for p in parents]

        next_pop = []
        for parents, offspring in families:
            for i in range(2):
                if fitness.score(offspring[i]) >= fitness.score(parents[i]):
                    next_pop.append(offspring[i])
                else:
                    next_pop.append(parents[i])

        # fight to the death
        # Note: `sorted()` sorts in-order, meaning children will remain at the end of the list
        # next_pop = [sorted(f, key=fitness.score)[:-2] for f in families]
        # next_pop = [item for sublist in next_pop for item in sublist]

        next_generation = Population(population_size, solution_length,
                                     generation=current_population.generation + 1, x=next_pop)
        next_fitness = best_fitness(next_generation, fitness)

        # if new generation is better than previous generation
        if next_fitness > generation_info[1]:
            generation_info[0] = next_generation.generation
            generation_info[1] = next_fitness

        current_population = next_generation

    return current_population, generation_info


def run_experiment(fitness, mutation, population_size=10, max_population_size=1280, solution_length=40):
    global optimum_found

    current_population_size = population_size
    previous_population_size = population_size

    while current_population_size < max_population_size:
        last_generation, generation_info = run_generation(current_population_size, solution_length, fitness, mutation)

        optimum_found = found_global_optimum(last_generation)

        if optimum_found:
            break

        previous_population_size = current_population_size
        current_population_size *= 2

    if not optimum_found and current_population_size == max_population_size:
        return False, current_population_size

    upperbound = current_population_size
    lowerbound = previous_population_size

    while (upperbound - lowerbound) > population_size:
        current_population_size = lowerbound + (upperbound - lowerbound) // 2

        last_generation, generation_info = run_generation(current_population_size, solution_length, fitness, mutation)

        optimum_found = found_global_optimum(last_generation)

        if optimum_found:
            upperbound = current_population_size
        else:
            lowerbound = current_population_size

    return True, current_population_size


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


