import numpy as np


def create_unlinked_block(x, k):
    m = len(x) // k
    blocks = [[] for _ in range(m)]

    for i in range(len(x)):
        blocks[i % m].append(x[i])

    return blocks


def create_linked_blocks(x, k):
    block_count = len(x) // k

    return np.array_split(x, block_count)


class FitnessInterface:
    def score(self, x):
        pass


class CountingOne(FitnessInterface):
    def score(self, x):
        return sum(x)


class Trap(FitnessInterface):
    def __init__(self, k, d, tightly_linked=True):
        self.tightly_linked = tightly_linked
        self.k = k
        self.d = d

        self.counting_ones = CountingOne()

    def score(self, x):
        blocks = create_linked_blocks(x, self.k) if self.tightly_linked else create_unlinked_block(x, self.k)

        fitness = 0

        for block in blocks:
            if self.counting_ones.score(block) == self.k:
                fitness += self.k
            else:
                fitness += self.k - self.d - (self.k - self.d) / (self.k - 1) * self.counting_ones.score(block)

        return fitness

