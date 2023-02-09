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
    def __init__(self, x):
            self.x = x

    def __init__(self):
         self.x = [random_chromosome() for i in range(M)]

    def __str__(self):
        return "POPULATION\n" + '\n'.join([str(i) for i in self.x]) + '\n'

    def __getitem__(self, key):
         return self.x[key]
    
    def shuffle(self):
         random.shuffle(self.x)
            


def random_chromosome():
     return chromosome([bool(random.randrange(0, 2)) for _ in range(N)])

    
def crossover_uniform(x1, x2, k):
    y1 = np.zeros(N)
    y2 = np.zeros(N)
    for i in range(N):
        if random.random() < k: # crossover
            y1[i] = x2[i]
            y2[i] = x1[i]
        else:                   # not crossover
            y1[i] = x1[i]
            y2[i] = x2[i]
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

    pep.shuffle()       # 1. shuffle

    # newpep = population([sorted((list(crossover_uniform(p1,p2,0.2)) + [p1,p2]), key=objective)[:2] for (p1,p2) in zip(pep.x[:-1], pep.x[1:])])

    quit()

    print(x1, x2)

    # y1, y2 = crossover_uniform(x1, x2, 0.2)
    y1, y2 = crossover_2point(x1, x2)


    print(y1, y2)


if __name__== "__main__":
    main()
