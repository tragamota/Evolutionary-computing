import random

import numpy as np

from fitness import CountingOne, Trap
from mutation import UniformCrossover, TwoPointCrossover
from structures import Population


def found_optima(population):
    return np.any([0 not in c.x for c in population])


def stopping_criteria(population, generation_info):
    optima = found_optima(population)
    has_improvement = population.generation >= generation_info[0] + 10

    return optima or has_improvement


def best_fitness(population, fitness):
    return np.max([fitness.score(c) for c in population])


def run_experiment(fitness, mutation, population_size=10, solution_length=40):
    current_population_size = population_size
    previous_population_size = population_size

    optimum_found = False

    while current_population_size <= 1280:
        current_population = Population(current_population_size, solution_length, generation=0)
        generation_info = [0, best_fitness(current_population, fitness)]

        while stopping_criteria(current_population, generation_info) == False:
            random.shuffle(current_population)

            parents = list(zip(current_population[:-1:2], current_population[1::2]))
            families = [(p, mutation.mutate(*p)) for p in parents]

            next_pop = []
            for family in families:
                parents, offspring = family

                for i in range(2):
                    if fitness.score(offspring[i]) >= fitness.score(parents[i]):
                        next_pop.append(offspring[i])
                    else:
                        next_pop.append(parents[i])

            # next_pop = [sorted(f, key=fitness.score)[:-2] for f in families]
            # next_pop = [item for sublist in next_pop for item in sublist]

            next_generation = Population(current_population_size, solution_length, generation=current_population.generation + 1, x=next_pop)
            next_fitness = best_fitness(next_generation, fitness)

            if next_fitness > generation_info[1]:  # BIG ASS TODO - TEXT UNCLEAR
                generation_info[0] = next_generation.generation
                generation_info[1] = next_fitness

            current_population = next_generation

        optimum_found = found_optima(current_population)
        previous_population_size = current_population_size
        current_population_size *= 2

        print(generation_info)

        if optimum_found:
            break

    # TODO: Bisection????

    return optimum_found


if __name__ == "__main__":
    # experiment 1

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

    print(f"Trap deceptive not tightly linked and Uniform crossover")
    for _ in range(20):
        fitness = Trap(k=4, d=2.5, tightly_linked=False)
        mutation = UniformCrossover()

        print(run_experiment(fitness, mutation))

    print(f"Trap deceptive not tightly linked and Two point crossover")
    for _ in range(20):
        fitness = Trap(k=4, d=2.5, tightly_linked=False)
        mutation = TwoPointCrossover()

        print(run_experiment(fitness, mutation))


