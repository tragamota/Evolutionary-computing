import random
import time

import numpy as np

from fitness import CountingOne, Trap
from crossover import UniformCrossover, TwoPointCrossover
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

    generations = []

    while not stopping_criteria(current_population, generation_info):
        # one generation:
        random.shuffle(current_population)
        parents = list(zip(current_population[:-1:2], current_population[1::2]))
        families = [(*p, *mutation.mutate(*p)) for p in parents]

        # fight to the death
        # Note: `sorted()` sorts in-order, meaning children will remain at the end of the list
        next_pop = [sorted(f, key=fitness.score)[-2:] for f in families]
        next_pop = [item for sublist in next_pop for item in sublist]

        next_generation = Population(population_size, solution_length,
                                     generation=current_population.generation + 1, x=next_pop)
        next_fitness = best_fitness(next_generation, fitness)

        # if new generation is better than previous generation
        if next_fitness > generation_info[1]:
            generation_info[0] = next_generation.generation
            generation_info[1] = next_fitness

        generations.append(current_population)

        current_population = next_generation

    return current_population, generation_info, generations


def run_experiment(fitness, mutation, population_size=10, max_population_size=1280, solution_length=40):
    optimum_found = False

    current_population_size = population_size
    previous_population_size = population_size

    successful_runs = []

    while current_population_size <= max_population_size:
        successful_runs = []
        for i in range(20):
            last_generation, generation_info, generations = run_generation(current_population_size, solution_length, fitness, mutation)
            optimum_found_run = found_global_optimum(last_generation)
            successful_runs.append(optimum_found_run)

        if np.sum(successful_runs) >= 19:
            optimum_found = True
            break

        print(current_population_size)

        if current_population_size < max_population_size:
            previous_population_size = current_population_size
            current_population_size *= 2
        else:
            break

    if not optimum_found and current_population_size == max_population_size:
        return False, np.sum(successful_runs), current_population_size

    upperbound = current_population_size
    lowerbound = previous_population_size

    optimum_found = False

    generation_count = 0
    starting_time = 0

    while (upperbound - lowerbound) > population_size:
        generation_count = 0
        successful_runs = []
        print(current_population_size)
        current_population_size = lowerbound + (upperbound - lowerbound) // 2
        starting_time = time.perf_counter()
        for i in range(20):
            last_generation, generation_info, generations = run_generation(current_population_size, solution_length, fitness, mutation)
            optimum_found_run = found_global_optimum(last_generation)
            if optimum_found_run:
                generation_count += len(generations) / 20
            successful_runs.append(optimum_found_run)

        if np.sum(successful_runs) >= 19:
            optimum_found = True
            upperbound = current_population_size
        else:
            optimum_found = False
            lowerbound = current_population_size

    if not optimum_found:
        current_population_size = upperbound

        generation_count = 0
        successful_runs = []
        starting_time = time.perf_counter()
        for i in range(20):
            last_generation, generation_info, generations = run_generation(current_population_size, solution_length,
                                                                           fitness, mutation)
            optimum_found_run = found_global_optimum(last_generation)
            generation_count += len(generations) / 20
            successful_runs.append(optimum_found_run)

    end_time = time.perf_counter()

    return True, np.sum(successful_runs), current_population_size, generation_count, (end_time - starting_time) / 20


if __name__ == "__main__":

    print(f"Counting ones and Uniform crossover")
    fitness = CountingOne()
    mutation = UniformCrossover()

    print(run_experiment(fitness, mutation))

    print(f"Counting ones and Two point crossover")
    fitness = CountingOne()
    mutation = TwoPointCrossover()

    print(run_experiment(fitness, mutation))

    print(f"Trap deceptive tightly linked and Uniform crossover")
    fitness = Trap(k=4, d=1, tightly_linked=True)
    mutation = UniformCrossover()

    print(run_experiment(fitness, mutation))

    print(f"Trap deceptive tightly linked and Two point crossover")
    fitness = Trap(k=4, d=1, tightly_linked=True)
    mutation = TwoPointCrossover()

    print(run_experiment(fitness, mutation))

    print(f"Trap non-deceptive tightly linked and Uniform crossover")
    fitness = Trap(k=4, d=2.5, tightly_linked=True)
    mutation = UniformCrossover()

    print(run_experiment(fitness, mutation))

    print(f"Trap non-deceptive tightly linked and Two point crossover")
    fitness = Trap(k=4, d=2.5, tightly_linked=True)
    mutation = TwoPointCrossover()

    print(run_experiment(fitness, mutation))

    print(f"Trap deceptive not tightly linked and Uniform crossover")
    fitness = Trap(k=4, d=1, tightly_linked=False)
    mutation = UniformCrossover()

    print(run_experiment(fitness, mutation))

    print(f"Trap deceptive not tightly linked and Two point crossover")
    fitness =  Trap(k=4, d=1, tightly_linked=False)
    mutation = TwoPointCrossover()

    print(run_experiment(fitness, mutation))

    print(f"Trap non-deceptive not tightly linked and Uniform crossover")
    fitness = Trap(k=4, d=2.5, tightly_linked=False)
    mutation = UniformCrossover()

    print(run_experiment(fitness, mutation))

    print(f"Trap non-deceptive not tightly linked and Two point crossover")
    fitness = Trap(k=4, d=2.5, tightly_linked=False)
    mutation = TwoPointCrossover()

    print(run_experiment(fitness, mutation))




