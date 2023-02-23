import numpy as np
import random


# member object
class chromosome:
    def __init__(self, x):
        self.x = x

    def __str__(self):
        return str([int(i) for i in self.x])

    def __getitem__(self, key):
        return self.x[key]


class population:
    N : int
    M : int
    gen : int
    def __init__(self, M, N, gen, x=None):
        self.N = N
        self.M = M
        self.gen = gen
        if x == None:
            self.x = [random_chromosome(N) for i in range(M)]
        else:
            self.x = x

    def __str__(self):
        return "POPULATION\n" + '\n'.join([str(i) for i in self.x]) + '\n'

    def __getitem__(self, key):
        return self.x[key]
    
    def next_generation(self, crossover_func, k, fitness_func):
        random.shuffle(self.x)                                           # 1. shuffle
        parents  = list(zip(self.x[:-1:2], self.x[1::2]))                # 2. pair up
        families = [(*p,*crossover_func(*p, k)) for p in parents]        # 3. family
        next_pop  = [sorted(f, key=fitness_func)[:-2] for f in families] # 4. fight to the death
        next_pop = [item for sublist in next_pop for item in sublist]
        return population(next_pop, self.M, self.N, self.gen+1) # return a new population

    def best_chromosome(self, fitness_func):
        # np.max([fitness_func(c) for c in self.x])
        sorted(self.x, key=fitness_func)[0]


class genetic_algorithm:
    N : int # chromosome length
    M : int # population size
    (best_chromo, best_fitness, best_gen) : tuple
    pop : population
    crossover_func : function
    k : float
    fitness_func : function

    def __init__(self, M, crossover_func, k, fitness_func):
        self.N = 10
        self.M = M
        self.crossover_func = crossover_func
        self.k = k
        self.fitness_func = fitness_func
        self.population = population(self.M, self.N)
        self.best_fitness = 0
    def stopping_criteria(self):
        return np.any([0 not in c.x for c in self.pop.x]) or (self.pop.gen >= self.best_gen + 10)
    def run_experiment(self):
        Ns = [False for i in range(128)] # 1 - 128 :: all values of N to try


        # TODO UNFINISHED
        # passing_by function for all the collecting of info

        # bisection search thingy
        # new gen loop
        new_pop = self.pop.next_generation(self.crossover_func, self.k, self.fitness_func)
        new_best_chromo = new_pop.best_chromosome(self.fitness_func)
        new_fitness = self.fitness_func(new_best_chromo)
        if new_fitness > self.best_fitness: # BIG ASS TODO - TEXT UNCLEAR
            self.best_gen = new_pop.gen
            self.best_fitness = new_fitness
            self.best_chromo = new_best_chromo
        self.pop = new_pop
        if self.stopping_criteria:
            return new_best_chromo

# passer function. Collects data to plot / graph, puts it in a file, and passes the result through
def collect_data():
    pass


def random_chromosome(N): # even shorter!
    return chromosome(np.random.rand(N) < 0.5)

def crossover_uniform(x1, x2, N, k=0.05): # also even shorter!
    bs = np.random.rand(N) < 0.5
    y1 = [x1[i] if bs[i] else x2[i] for i in range(len(bs))]
    y2 = [x2[i] if bs[i] else x1[i] for i in range(len(bs))]
    return chromosome(y1), chromosome(y2)

def crossover_2point(x1, x2, N, k=0):
    split1 = random.randrange(0,N); split2 = random.randrange(0,N)
    minsplit = min(split1, split2); maxsplit = max(split1, split2)
    y1 = [x1[i] for i in range(0, minsplit)] + [x2[i] for i in range(minsplit, maxsplit)] + [x1[i] for i in range(maxsplit, N)] # from the minimum split value to the maximum split value, the values are swapped
    y2 = [x2[i] for i in range(0, minsplit)] + [x1[i] for i in range(minsplit, maxsplit)] + [x2[i] for i in range(maxsplit, N)]
    return chromosome(y1), chromosome(y2)


def counting_ones(x):
    return sum(x)

def trap_deceptive_linked(x):
    return trap(x, k=4, d=1, linked=True)

def trap_nondeceptive_linked(x):
    return trap(x, k=4, d=2.5, linked=True)

def trap_deceptive_unlinked(x):
    return trap(x, k=4, d=1, linked=False)

def trap_nondeceptive_unlinked(x):
    return trap(x, k=4, d=2.5, linked=False)

def trap(c, k, d, linked):
    # divide x into blocks of k
    x = c.x
    blocks = linked_block(x,k) if linked else unlinked_block(x,k)

    fitness = 0
    for block in blocks:
        if counting_ones(block) == k:
            fitness += k
        else:
            fitness += k - d - (k-d)/(k-1)*counting_ones(block)
    return fitness

# TODO NP ARRAY?
def linked_block(x, k):
    l = len(x) // k
    blocks = np.array_split(x, l)
    return blocks

def unlinked_block(x, k):
    m = len(x) // k
    blocks = [[] for _ in range(m)]
    for i in range(len(x)):
        blocks[i % m].append(x[i])
    return blocks


def main():
    return 0

if __name__== "__main__":
    main()
