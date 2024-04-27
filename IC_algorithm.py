import numpy


class Country:
    def __init__(self, location):
        self.location = location
        self.fitness = None

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
        self.population = [Country(numpy.random.uniform(low=-5.12, high=5.12, size=dimension)) for _ in
                           range(population_size)]
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []

    def assimilation(self, imperialists, colonies):
        for colony in colonies:
            nearest_imperialist = min(imperialists, key=lambda x: numpy.linalg.norm(x.location - colony.location))
            if colony.fitness < nearest_imperialist.fitness:
                nearest_imperialist = colony

    def revolution(self):
        for country in self.population:
            country.location += numpy.random.normal(scale=0.1, size=country.location.shape)

    def optimize(self, lb, ub):
        for _ in range(self.max_iter):
            for country in self.population:
                country.evaluate_fitness(self.objective_function)

            for country in self.population:
                if country.fitness < self.best_fitness:
                    self.best_solution = country.location
                    self.best_fitness = country.fitness

            self.fitness_history.append(self.best_fitness)
            self.population.sort(key=lambda x: x.fitness)
            n_empire = int(0.1 * self.population_size)
            empires = self.population[:n_empire]
            colonies = self.population[n_empire:]

            for colony in colonies:
                nearest_imperialist = min(empires, key=lambda x: numpy.linalg.norm(x.location - colony.location))
                if colony.fitness < nearest_imperialist.fitness:
                    nearest_imperialist = colony

            for colony in colonies:
                if numpy.random.rand() < 0.05:
                    colony.location += numpy.random.uniform(low=lb, high=ub, size=self.dimension)

            if len(empires) > 1:
                weakest_emp_ind = numpy.argmin([empire.fitness for empire in empires])
                weak_colonies = [colony for colony in colonies if
                                 colony.fitness == min([colony.fitness for colony in colonies])]
                colonies_takeover_size = int(len(weak_colonies) * 0.5)
                takeover_ind = numpy.argmax([numpy.abs((empire.fitness + 0.05 * empire.fitness) / sum(
                    [(empire.fitness + 0.05 * empire.fitness) for empire in empires])) for empire in empires])

                for i in range(colonies_takeover_size):
                    weakest_colony_ind = colonies.index(min(weak_colonies, key=lambda x: x.fitness))
                    colonies[weakest_colony_ind].location = empires[takeover_ind].location
                    colonies[weakest_colony_ind].fitness = empires[takeover_ind].fitness
                    weak_colonies.remove(weak_colonies[weakest_colony_ind])

                if len(weak_colonies) == 0:
                    colonies.append(Country(empires[weakest_emp_ind].location))
                    colonies[-1].fitness = empires[weakest_emp_ind].fitness
                    del empires[weakest_emp_ind]
