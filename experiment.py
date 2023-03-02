import random
import time

import numpy as np
import matplotlib.pyplot as plt

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
    global selection_decisions

    current_population = Population(population_size, solution_length, generation=0)
    generation_info = [0, best_fitness(current_population, fitness)]  # (best gen, best fitness)

    generations = []

    while not stopping_criteria(current_population, generation_info):

        # one generation:
        random.shuffle(current_population) # 1. shuffle
        parents = list(zip(current_population[:-1:2], current_population[1::2])) # 2. pairwise matching
        families = [(*p, *mutation.mutate(*p)) for p in parents] # 3. family competition
        _next_pop = [sorted(f, key=fitness.score)[-2:] for f in families] # 4. fight to the death
        # Note: `sorted()` sorts in-order, meaning children will remain at the end of the list
        next_pop = [item for sublist in _next_pop for item in sublist]

        # Here we check:
        # given the families:
        # 1. are at the parents any two bits different?
        # 2. at those indices, did both who won have the same bit-value?
        # 3. if so, add to the respective summation. We just want to plot the absolute number.
        #    we only want to compare the two. No need to average or something
        for i in range(len(families)): # a four-tuple (parent1, parent2, child1, child2)
            par1 = families[i][0]
            par2 = families[i][1]
            win1 = _next_pop[i][0]
            win2 = _next_pop[i][1]
            for j in range(len(par1.x)):
                if par1[j] != par2[j] and win1[j] == win2[j]:
                    selection_decisions[win1[j]] += 1
        # TODO NOW DO IT BASED ON GENERATIONS!!

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



def plotMeasures():

    global selection_decisions
    selection_decisions = [0,0] # TODO

    N = 200
    l = 40
    fitness = CountingOne()
    mutation = TwoPointCrossover()
    # "a single run is sufficient"
    _, _, generations = run_generation(N, l, fitness, mutation)

    gen_range = range(0, len(generations))
    # 1. "Plot the proportion prop(t) of bits-1 in the entire population as a function of the generation t."
    prop_1_bits = [np.average([sum(sol.x) / len(sol.x) for sol in gen]) for gen in generations]
    # 2. "Plot the number of selection errors Err(t) and the number of correct selection decisions"
    selection_errors = selection_decisions[0]
    correct_selection = selection_decisions[1]
    print(selection_errors)
    print(correct_selection)
    # 3. "Plot the number of solutions in the population that are member of the respective schemata"
    no_1_schemata = [np.sum([1 for sol in gen if sol.x[0] == 1]) / N for gen in generations]
    fitni_1_schemata = [[fitness.score(sol) for sol in gen if sol.x[0] == 1] for gen in generations]
    fitni_0_schemata = [[fitness.score(sol) for sol in gen if sol.x[0] == 0] for gen in generations]
    avg_fitness_1_schemata = [np.average(fitness) for fitness in fitni_1_schemata]
    avg_fitness_0_schemata = [np.average(fitness) for fitness in fitni_0_schemata]
    std_fitness_1_schemata = [np.std(fitness) for fitness in fitni_1_schemata]
    std_fitness_0_schemata = [np.std(fitness) for fitness in fitni_0_schemata]

    ax = plt.subplot()
    ax.plot(gen_range, prop_1_bits)
    ax.set_xlabel("generation")
    ax.set_ylabel("proportion of 1-bits in population")
    ax.set_xlim(0, len(generations)-1)
    ax.set_ylim((0,1))
    plt.show()

    _, axes = plt.subplots(1, 2)
    ax = axes[0]
    ax.plot(gen_range, no_1_schemata)
    ax.plot(gen_range, [1 - i for i in no_1_schemata])
    ax.set_xlabel("generation")
    ax.set_ylabel("proportion of solutions part of schemata")
    ax.set_xlim(0, len(generations)-1)
    ax.set_ylim(0, 1)
    ax.legend(["1**..** schemata", "0**..** schemata"])
    ax = axes[1]
    # ax.plot(gen_range, avg_fitness_1_schemata)
    plt.errorbar(x=gen_range, y=avg_fitness_1_schemata, yerr=std_fitness_1_schemata)
    # ax.plot(gen_range, avg_fitness_0_schemata)
    plt.errorbar(x=[x+0.1 for x in gen_range], y=avg_fitness_0_schemata, yerr=std_fitness_0_schemata)
    # small offset in x, such that standard deviations don't overlap
    ax.set_xlabel("generation")
    ax.set_ylabel("average fitness of schemata")
    ax.legend(["1**..** schemata", "0**..** schemata"])
    ax.set_xlim(0, len(gen_range)-1)
    plt.show()


if __name__ == "__main__":

    plotMeasures()
    quit()



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




