import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy
import random


# matplotlib.use("TkAgg")


def rastrigin(x):
    a = 10
    return a * 2 + sum(x ** 2 - a * numpy.cos(2 * numpy.pi * x))


class Country:
    def __init__(self, location):
        self.location = location
        self.fitness = float('-inf')
        self.vassal_of_empire = None
        # when -1 no empire, if the country is empire than it is same as the name
        self.index_in_list = -1
        self.norm_imperialist_power = 0
        self.vassals = []
        self.colour = None

    def evaluate_fitness(self, objective_function):
        self.fitness = objective_function(self.location)

    def weakest_vassal_removal(self):
        result = self.vassals[0]
        index = 0
        for i in range(len(self.vassals)):
            if result.fitness > self.vassals[i].fitness:
                result = self.vassals[i]
                index = i
        del self.vassals[index]
        return result

    def add_vassal(self, vassal: "Country"):
        self.vassals.append(vassal)

    def add_emperor(self, emperor: "Country"):
        self.vassal_of_empire = emperor


class Imperialist_competitive_algorithm:
    def __init__(self, population_size: int, dimension: int, max_iter: int, objective_function):
        self.population_size = population_size
        self.dimension = dimension
        self.max_iter = max_iter
        self.objective_function = objective_function
        self.population = [Country(numpy.random.uniform(low=-5.12, high=5.12, size=dimension)) for i in
                           range(population_size)]
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        self.colours = []
        self.empires = []
        self.colonies = []

    def random_colours(self, n_empire):
        res = []
        for i in range(n_empire):
            r = numpy.random.rand()
            g = numpy.random.rand()
            b = numpy.random.rand()
            res.append((r, g, b))
        return res

    def create_empires(self):
        n_empire = int(0.1 * self.population_size)
        self.empires = self.population[:n_empire]
        self.colours = self.random_colours(n_empire)
        for i in range(n_empire):
              self.empires[i].index_in_list = i
              self.empires[i].colour = self.colours[i]
        for empire in self.empires:
            empire.norm_imperialist_power = numpy.abs(
                empire.fitness / numpy.sum([empire.fitness for empire in self.empires]))

    def create_colonies(self):
        self.colonies = self.population[len(self.empires):]
        colonies_in_empires = [numpy.floor(empire.norm_imperialist_power * len(self.colonies)) for empire in self.empires]
        if numpy.sum(colonies_in_empires) < self.population_size:
            diff = len(self.colonies) - numpy.sum(colonies_in_empires)
            i = 0
            while True:
                temp = numpy.ceil(diff * self.empires[i].norm_imperialist_power)
                diff -= temp
                colonies_in_empires[i] += temp
                if diff == 0:
                    break
        list_of_index = list(range(len(self.colonies)))
        for i in range(len(self.empires)):
            temp = random.sample(range(int(len(list_of_index))), int(colonies_in_empires[i]))
            temp.sort(reverse=True)
            for j in range(int(colonies_in_empires[i])):
                self.empires[i].vassals.append(self.colonies[list_of_index[temp[j]]])
                self.colonies[list_of_index[temp[j]]].vassal_of_empire =   self.empires[i]
                self.colonies[list_of_index[temp[j]]].colour =   self.empires[i].colour
            for k in range(len(temp)):
                del list_of_index[temp[k]]

    def assimilation(self, beta=2):
        for colony in self.colonies:
            distance = numpy.linalg.norm(
                [colony.vassal_of_empire.location[i] - colony.location[i] for i in range(self.dimension)])
            shift = numpy.random.uniform(low=0, high=beta * distance, size=1)
            new_location = numpy.zeros(self.dimension)
            if distance != 0:
                for i in range(self.dimension):
                    new_location[i] = colony.location[i] + shift * (
                            colony.vassal_of_empire.location[i] - colony.location[i]) / distance
            else:
                new_location = colony.location
            colony.location = numpy.copy(new_location)

    def revolution(self, gamma=numpy.pi / 4):
        for colony in self.colonies:
            colony.location += numpy.random.uniform(low=-gamma, high=gamma, size=self.dimension)

    def mutiny(self):
        for colony in self.colonies:
            nearest_imperialist = min(self.empires, key=lambda x: numpy.linalg.norm(x.location - colony.location))
            if colony.fitness < nearest_imperialist.fitness:
                colony.vassals = nearest_imperialist.vassals
                nearest_imperialist.vassal_of_empire = colony
                self.empires[nearest_imperialist.index_in_list] = colony
                colony.index_in_list = numpy.copy(nearest_imperialist.index_in_list)
                self.colonies[self.colonies.index(colony)] = nearest_imperialist
            else:
                colony.vassal_of_empire = nearest_imperialist

    def empirial_war(self, eta=0.1):
        total_power = 0
        powers_of_empire = []
        normalized_total_power = []
        temp_sum = 0
        num_of_empires = len(self.empires)

        for empire in self.empires:
            powers_of_empire.append(empire.fitness + eta * sum([vassal.fitness for vassal in empire.vassals]))
            if total_power < powers_of_empire[-1]:
                total_power = powers_of_empire[-1]
        for i in range(num_of_empires):
            normalized_total_power.append(powers_of_empire[i] - total_power)
            temp_sum += normalized_total_power[-1]

        possession_probability = [numpy.round(normalized_total_power[i] / temp_sum) for i in range(num_of_empires)]
        random_numbers = numpy.random.uniform(low=0, high=1, size=len(self.empires))
        D = [possession_probability[i] - random_numbers[i] for i in range(num_of_empires)]
        weakest_empire_index = D.index(numpy.min(D))
        strongest_empire_index = D.index(numpy.max(D))
        # if len(empires[weakest_empire_index].vassals) != delta:
        # for _ in   self.empires[weakest_empire_index].vassals:
        # weakest_vassal =   self.empires[weakest_empire_index].weakest_vassal_removal()
        #   self.empires[strongest_empire_index].vassals.append(weakest_vassal)
        # weakest_vassal.vassal_of_empire =   self.empires[strongest_empire_index]
        if len(self.empires[weakest_empire_index].vassals) != 0:
            weakest_vassal = self.empires[weakest_empire_index].weakest_vassal_removal()
            self.empires[strongest_empire_index].vassals.append(weakest_vassal)
            weakest_vassal.vassal_of_empire = self.empires[strongest_empire_index]
            weakest_vassal.colour = self.empires[strongest_empire_index].colour
        if len(self.empires[weakest_empire_index].vassals) == 0:
            self.empires[strongest_empire_index].vassals.append(self.empires[weakest_empire_index])
            self.empires[weakest_empire_index].vassal_of_empire = self.empires[strongest_empire_index]
            self.empires[weakest_empire_index].colour = self.empires[strongest_empire_index].colour
            self.colonies.append(self.empires[weakest_empire_index])
            for i in range(weakest_empire_index, len(self.empires)):
              self.empires[i].index_in_list -= 1
              self.empires.pop(weakest_empire_index)

    def calculate_fitness(self):
        for country in self.population:
            country.evaluate_fitness(self.objective_function)
            if country.fitness < self.best_fitness:
                self.best_solution = country.location
                self.best_fitness = country.fitness

    def print_number_of_vassals(self):
        for i in range(len(self.empires)):
            temp = "Number of vassals in empire " + str(i) + " is " + str(len(self.empires[i].vassals))
            print(temp)
        print("Optimal coordinates: " + str(self.best_solution))
        print("Optimal value: " + str(self.best_fitness))

    def optimize(self, lb: int, ub: int, beta, gamma, eta):
        self.calculate_fitness()
        self.population.sort(key=lambda x: x.fitness)
        self.create_empires()
        self.create_colonies()
        for i in range(self.max_iter):
            print("#############Start of " + str(i) + ". iteration. ########################################")
            self.calculate_fitness()
            self.fitness_history.append(self.best_fitness)
            self.assimilation(beta)
            self.revolution(gamma)
            self.mutiny()
            self.empirial_war(eta)
            self.print_number_of_vassals()
            print("#####################################################################################")
            if len(self.empires) == 1:
                self.calculate_fitness()
                break

    def animate_optimization(self):
        fig, ax = plt.subplots()
        self.calculate_fitness()
        self.population.sort(key=lambda x: x.fitness)
        self.create_empires()
        self.create_colonies()

        def update(frame):
            ax.clear()
            old_empires =   self.empires
            old_colonies = self.colonies

            # Run one iteration of optimization
            if frame % 4 == 0:
                self.assimilation()
            elif frame % 4 == 1:
                self.revolution()
            elif frame % 4 == 2:
                self.mutiny()
            else:
                self.empirial_war()

            self.print_number_of_vassals()
            self.calculate_fitness()
            self.fitness_history.append(self.best_fitness)

            # Plot   self.empires and self.colonies
            self.setup_plot(ax)

            return ax,

        anim = FuncAnimation(fig, update, frames=range(self.max_iter), blit=True)
        plt.show()

    def setup_plot(self,  ax):
        locations_empires = numpy.zeros([len(self.empires), self.dimension])
        locations_colonies = numpy.zeros([len(self.colonies), self.dimension])
        for i in range(self.dimension):
            for j in range(len(self.empires)):
                locations_empires[j, i] = self.empires[j].location[i]
            for j in range(len(self.colonies)):
                locations_colonies[j, i] = self.colonies[j].location[i]
        ax.scatter(locations_empires[:, 0], locations_empires[:, 1], marker="s", c=[empire.colour for empire in self.empires])
        ax.scatter(locations_colonies[:, 0], locations_colonies[:, 1], c=[colony.colour for colony in self.colonies])
