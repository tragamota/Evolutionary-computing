import numpy as np


# Chromosome
class Solution:
    def __init__(self, x):
        self.x = self.random_solution(x) if type(x) == int else x

    def __str__(self):
        return str([int(i) for i in self.x])

    def __getitem__(self, key):
        return self.x[key]

    def __setitem__(self, key, value):
        self.x[key] = value

    def __len__(self):
        return len(self.x)

    @staticmethod
    def random_solution(solution_length):
        return np.random.rand(solution_length) < 0.5


class Population:

    def __init__(self, population_size, solution_length, generation, x=None):
        self.population_size = population_size
        self.solution_length = solution_length
        self.generation = generation

        self.x = x if x is not None else [Solution(solution_length) for i in range(self.population_size)]

    def __getitem__(self, key):
        return self.x[key]

    def __setitem__(self, key, value):
        self.x[key] = value

    def __len__(self):
        return len(self.x)
