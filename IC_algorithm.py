import numpy


class Country:
    def __init__(self, location, name):
        self.location = location
        self.fitness = None
        self.vassal_of_empire = -1
        #when -1 no empire, if the country is empire than it is same as the name
        self.name = name
        self.norm_imperialist_power = 0

    def evaluate_fitness(self, objective_function):
        self.fitness = objective_function(self.location)


def rastrigin(x):
    a = 10
    return a * 2 + sum(x ** 2 - a * numpy.cos(2 * numpy.pi * x))


class ImperialistCompetitiveAlgorithm:
    def __init__(self, population_size, dimension, max_iter, objective_function):
        self.population_size = population_size
        self.dimension = dimension
        self.max_iter = max_iter
        self.objective_function = objective_function
        self.population = [Country(numpy.random.uniform(low=-5.12, high=5.12, size=dimension), i) for i in
                           range(population_size)]
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []

    def assimilation(self, empires, colonies):
        for colony in colonies:
            nearest_imperialist = min(empires, key=lambda x: numpy.linalg.norm(x.location - colony.location))
            if colony.fitness < nearest_imperialist.fitness:
                nearest_imperialist = colony

    def revolution(self):
        for country in self.population:
            country.location += numpy.random.normal(scale=0.1, size=country.location.shape)

    def optimize(self, lb, ub):
        for _ in range(self.max_iter):
            for country in self.population:
                country.evaluate_fitness(self.objective_function)
                if country.fitness < self.best_fitness:
                    self.best_solution = country.location
                    self.best_fitness = country.fitness

            self.fitness_history.append(self.best_fitness)
            self.population.sort(key=lambda x: x.fitness)
            n_empire = int(0.1 * self.population_size)
            empires = self.population[:n_empire]
            for empire in empires:
                empire.vassal_of_empire = empire.name
            colonies = self.population[n_empire:]

            for colony in colonies:
                nearest_imperialist = min(empires, key=lambda x: numpy.linalg.norm(x.location - colony.location))
                if colony.fitness < nearest_imperialist.fitness:
                    nearest_imperialist = colony
                colony.vassal_of_empire = nearest_imperialist.name

            for colony in colonies:
                if numpy.random.rand() < 0.05:
                    colony.location += numpy.random.uniform(low=lb, high=ub, size=self.dimension)



