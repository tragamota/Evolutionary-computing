import numpy as np

from random import randrange
from structures import Solution


class MutationInterface:
    def mutate(self, x, y):
        pass


class UniformCrossover(MutationInterface):
    def __init__(self, k=0.05):
        self.k = k

    def mutate(self, x, y):
        assert len(x) is len(y)

        bs = np.random.rand(len(x)) < self.k
        y1 = [x[i] if bs[i] else y[i] for i in range(len(bs))]
        y2 = [y[i] if bs[i] else x[i] for i in range(len(bs))]

        return Solution(y1), Solution(y2)


class TwoPointCrossover(MutationInterface):
    def mutate(self, x, y):
        assert len(x) is len(y)

        N = len(x)

        split1 = randrange(0, N)
        split2 = randrange(0, N)

        min_split = min(split1, split2)
        max_split = max(split1, split2)

        # from the minimum split value to the maximum split value, the values are swapped
        y1 = [x[i] for i in range(0, min_split)] + [y[i] for i in range(min_split, max_split)] + [x[i] for i in
                                                                                                  range(max_split, N)]
        y2 = [y[i] for i in range(0, min_split)] + [x[i] for i in range(min_split, max_split)] + [y[i] for i in
                                                                                                  range(max_split, N)]

        return Solution(y1), Solution(y2)
