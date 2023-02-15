import numpy as np
import random

N = 10 # chromosome length
M = 16 # population size

# member object
class chromosome:
    def __init__(self, x):
        self.x = x

    def __str__(self):
        return str([int(i) for i in self.x])

    def __getitem__(self, key):
        return self.x[key]


class population:
    def __init__(self, x=None):
        if x == None:
            self.x = [random_chromosome() for i in range(M)]
        else:
            self.x = x

    def __str__(self):
        return "POPULATION\n" + '\n'.join([str(i) for i in self.x]) + '\n'

    def __getitem__(self, key):
        return self.x[key]
    
    def shuffle(self):
        random.shuffle(self.x)

    def next_generation(self):
        self.shuffle()                                                 # 1. shuffle
        # print(population(self.x))
        parents  = list(zip(self.x[:-1:2], self.x[1::2]))              # 2. pair up
        # print(population(parents))
        families = [(*p,*crossover_uniform(*p, 0.1)) for p in parents] # 3. family
        # print(population(families))
        next_pop  = [sorted(f, key=objective)[:-2] for f in families]  # 4. fight to the death
        next_pop = [item for sublist in next_pop for item in sublist]
        # print(population(next_pop))
        return population(next_pop) # return a new population

            
# def random_chromosome():
    #  return chromosome([bool(random.randrange(0, 2)) for _ in range(N)])

def random_chromosome(): # even shorter!
    return chromosome(np.random.rand(N) < 0.5)
    
# def crossover_uniform(x1, x2, k):
#     y1 = np.zeros(N)
#     y2 = np.zeros(N)
#     for i in range(N):
#         if random.random() < k: # crossover
#             y1[i] = x2[i]
#             y2[i] = x1[i]
#         else:                   # not crossover
#             y1[i] = x1[i]
#             y2[i] = x2[i]
#     return chromosome(y1), chromosome(y2)

def crossover_uniform(x1, x2, k): # also even shorter!
    bs = np.random.rand(N) < 0.5
    y1 = [x1[i] if bs[i] else x2[i] for i in range(len(bs))]
    y2 = [x2[i] if bs[i] else x1[i] for i in range(len(bs))]
    return chromosome(y1), chromosome(y2)

# shit code ahead. WARNING. APPROACH WITH CAUTION
def crossover_2point(x1, x2):
    split1 = random.randrange(0,N); split2 = random.randrange(0,N)
    minsplit = min(split1, split2); maxsplit = max(split1, split2)
    y1 = [x1[i] for i in range(0, minsplit)] + [x2[i] for i in range(minsplit, maxsplit)] + [x1[i] for i in range(maxsplit, N)] # from the minimum split value to the maximum split value, the values are swapped
    y2 = [x2[i] for i in range(0, minsplit)] + [x1[i] for i in range(minsplit, maxsplit)] + [x2[i] for i in range(maxsplit, N)]
    return chromosome(y1), chromosome(y2)
    

# objective function
def objective(x):
	return sum(x)

def main():
    pep = population()
    print(pep)

    pep.next_generation()

if __name__== "__main__":
    main()
