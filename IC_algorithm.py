import numpy
import copy


class Country:
    def __init__(self, location, name):
        self.location = location
        self.fitness = None
        self.vassal_of_empire = -1
        # when -1 no empire, if the country is empire than it is same as the name
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

    def create_empires(self):
        n_empire = int(0.1 * self.population_size)
        empires = self.population[:n_empire]
        for empire in empires:
            empire.vassal_of_empire = empire.name
            empire.norm_imperialist_power = numpy.abs(
                empire.fitness / numpy.sum([empire.fitness for empire in empires]))
        return empires

    def create_colonies(self, empires):
        colonies = self.population[len(empires):]
        colonies_in_empires = [numpy.floor(empire.norm_imperialist_power * len(colonies)) for empire in empires]
        if numpy.sum(colonies_in_empires) < self.population_size:
            diff = self.population_size - numpy.sum(colonies_in_empires)
            i = 0
            while True:
                temp = numpy.ceil(diff * empires[i].norm_imperialist_power)
                diff -= temp
                colonies_in_empires[i] += temp
                if diff == 0:
                    break
        list_of_index = list(range(len(colonies)))
        for i in range(len(empires)):
            temp = numpy.random.choice(len(list_of_index), colonies_in_empires[i])
            for j in range(colonies_in_empires[i]):
                colonies[list_of_index[temp[i]]].vassal_of_empire = empires[i].name
            del list_of_index[temp]
        return colonies

    def assimilation(self, empires, colonies, beta=2):
        for colony in colonies:
            d = numpy.linalg.norm(self.population[colony.vassal_of_empire].location - colony.location)
            shift = numpy.random.uniform(low=0, high=beta * d, size=1)
            new_location = []
            for i in range(self.dimension):
                new_location[i] = colony.location[i] + shift * (self.population[colony.vassal_of_empire].location - colony.location[i]) / d
            colony.location = numpy.copy(new_location)

    def revolution(self, gamma=numpy.pi/4):
        for country in self.population:
            country.location += numpy.random.uniform(low=-gamma, high=gamma, size=self.dimension)



    def optimize(self, lb, ub):
        empires = self.create_empires()
        colonies = self.create_colonies(empires)
        for _ in range(self.max_iter):
            for country in self.population:
                country.evaluate_fitness(self.objective_function)
                if country.fitness < self.best_fitness:
                    self.best_solution = country.location
                    self.best_fitness = country.fitness

            self.fitness_history.append(self.best_fitness)
            self.population.sort(key=lambda x: x.fitness)

            for colony in colonies:
                nearest_imperialist = min(empires, key=lambda x: numpy.linalg.norm(x.location - colony.location))
                if colony.fitness < nearest_imperialist.fitness:
                    nearest_imperialist = colony
                colony.vassal_of_empire = nearest_imperialist.name

            for colony in colonies:
                if numpy.random.rand() < 0.05:
                    colony.location += numpy.random.uniform(low=lb, high=ub, size=self.dimension)
