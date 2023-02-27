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
        y1 = np.where(bs, x, y)
        y2 = np.where(bs, y, x)

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
        y1 = np.concatenate((x[:min_split], y[min_split:max_split], x[max_split:]))
        y2 = np.concatenate((y[:min_split], x[min_split:max_split], y[max_split:]))

        return Solution(y1), Solution(y2)
