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
    global collect_data

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
        if collect_data:
            gen = current_population.generation
            while len(selection_decisions) < gen + 1:
                selection_decisions.append([0,0])
            for i in range(len(families)): # a four-tuple (parent1, parent2, child1, child2)
                par1 = families[i][0]
                par2 = families[i][1]
                win1 = _next_pop[i][0]
                win2 = _next_pop[i][1]
                # print(len(families[i]), len(_next_pop[i]), '\n', par1, '\n', par2, '\n', win1, '\n', win2, '\n')
                for j in range(len(par1.x)):
                    if par1[j] != par2[j] and win1[j] == win2[j]:
                        selection_decisions[gen][win1[j]] += 1

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
    current_population_size = population_size
    previous_population_size = population_size

    successful_runs = []

    while current_population_size <= max_population_size:
        successful_runs = []
        for i in range(20): # 20 runs
            last_generation, generation_info, generations = run_generation(current_population_size, solution_length, fitness, mutation)
            optimum_found_run = found_global_optimum(last_generation)
            successful_runs.append(optimum_found_run)

        print(current_population_size)

        # at least 19 out of 20 runs succesful, break from this while-loop
        if np.sum(successful_runs) >= 19:
            break

        # double population size
        if current_population_size < max_population_size:
            previous_population_size = current_population_size
            current_population_size *= 2
        # except if reached 1280 population size and still no convergence: just stop
        else:
            return False, np.sum(successful_runs), current_population_size


    upperbound = current_population_size
    lowerbound = previous_population_size

    while (upperbound - lowerbound) > population_size:
        successful_runs = []
        current_population_size = lowerbound + (upperbound - lowerbound) // 2

        for i in range(20):
            last_generation, generation_info, generations = run_generation(current_population_size, solution_length, fitness, mutation)
            optimum_found_run = found_global_optimum(last_generation)
            successful_runs.append(optimum_found_run)

        print(current_population_size)

        if np.sum(successful_runs) >= 19:
            upperbound = current_population_size
        else:
            lowerbound = current_population_size


    # Here, we have the optimal population size
    # Lastly, do this last run again, now collecting info:
    current_population_size = upperbound # 1. minimal population size
    successful_runs = []
    generation_count = [] # 2. average number of generations (+ 3. average number of fitness func evals)
    run_times = [] # 4. average cpu time

    for i in range(20):
        starting_time = time.perf_counter()
        last_generation, generation_info, generations = run_generation(current_population_size, solution_length,
                                                                        fitness, mutation)
        end_time = time.perf_counter()

        optimum_found_run = found_global_optimum(last_generation)
        successful_runs.append(optimum_found_run)
        generation_count.append(len(generations))
        run_times.append(end_time - starting_time)

    return True, np.sum(successful_runs), current_population_size, (generation_count, np.average(generation_count), np.std(generation_count)), (run_times, np.average(run_times), np.std(run_times))


# only the last one - optimizing the CountingOnes function
def plotMeasures():
    global selection_decisions
    global collect_data
    collect_data = True
    selection_decisions = []

    N = 200
    l = 40
    fitness = CountingOne()
    mutation = TwoPointCrossover()
    # "a single run is sufficient"
    _, _, generations = run_generation(N, l, fitness, mutation)

    gen_range = range(0, len(generations))
    # 1. "Plot the proportion prop(t) of bits-1 in the entire population as a function of the generation t."
    prop_1_bits = [np.average([sum(sol.x) / len(sol.x) for sol in gen]) for gen in generations]
    prop_0_bits = [1 - x for x in prop_1_bits]
    # 2. "Plot the number of selection errors Err(t) and the number of correct selection decisions"
    selection_errors = [d[0] for d in selection_decisions]
    correct_selections = [d[1] for d in selection_decisions]
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
    ax.plot(gen_range, prop_0_bits)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Proportion of bits in population")
    ax.set_xlim(0, len(generations)-1)
    ax.set_ylim((0,1))
    ax.legend(["1-bit", "0-bit"])
    plt.show()

    ax = plt.subplot()
    ax.plot(gen_range, selection_errors)
    ax.plot(gen_range, correct_selections)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Selection decisions")
    ax.set_xlim(0, len(generations)-1)
    ax.legend(["selection errors", "correct selections"])
    plt.show()

    _, axes = plt.subplots(1, 2, constrained_layout = True) # figsize
    ax = axes[0]
    ax.plot(gen_range, no_1_schemata)
    ax.plot(gen_range, [1 - i for i in no_1_schemata])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Proportion of solutions part of schemata")
    ax.set_xlim(0, len(generations)-1)
    ax.set_ylim(0, 1)
    ax.legend(["1**..** schemata", "0**..** schemata"])
    ax = axes[1]
    # ax.plot(gen_range, avg_fitness_1_schemata)
    plt.errorbar(x=gen_range, y=avg_fitness_1_schemata, yerr=std_fitness_1_schemata)
    # ax.plot(gen_range, avg_fitness_0_schemata)
    plt.errorbar(x=[x+0.1 for x in gen_range], y=avg_fitness_0_schemata, yerr=std_fitness_0_schemata)
    # small offset in x, such that standard deviations don't overlap
    ax.set_xlabel("Generation")
    ax.set_ylabel("Average fitness of schemata")
    ax.legend(["1**..** schemata", "0**..** schemata"])
    ax.set_xlim(0, len(gen_range)-1)
    plt.show()


if __name__ == "__main__":

    # plotMeasures()
    # quit()

    global collect_data
    collect_data = False # to make sure we don't collect unneeded data; plotMeasures already collects that

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




