import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy
import random


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
        empires = self.population[:n_empire]
        self.random_colours(n_empire)
        for i in range(n_empire):
            empires[i].index_in_list = i
        for empire in empires:
            empire.norm_imperialist_power = numpy.abs(
                empire.fitness / numpy.sum([empire.fitness for empire in empires]))
        return empires

    def create_colonies(self, empires: list[Country]):
        colonies = self.population[len(empires):]
        colonies_in_empires = [numpy.floor(empire.norm_imperialist_power * len(colonies)) for empire in empires]
        if numpy.sum(colonies_in_empires) < self.population_size:
            diff = len(colonies) - numpy.sum(colonies_in_empires)
            i = 0
            while True:
                temp = numpy.ceil(diff * empires[i].norm_imperialist_power)
                diff -= temp
                colonies_in_empires[i] += temp
                if diff == 0:
                    break
        list_of_index = list(range(len(colonies)))
        for i in range(len(empires)):
            temp = random.sample(range(int(len(list_of_index))), int(colonies_in_empires[i]))
            temp.sort(reverse=True)
            for j in range(int(colonies_in_empires[i])):
                empires[i].vassals.append(colonies[list_of_index[temp[j]]])
                colonies[list_of_index[temp[j]]].vassal_of_empire = empires[i]
            for k in range(len(temp)):
                del list_of_index[temp[k]]
        return colonies

    def assimilation(self, empires: list[Country], colonies: list[Country], beta=2):
        for colony in colonies:
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

    def revolution(self, colonies: list[Country], gamma=numpy.pi / 4):
        for colony in colonies:
            colony.location += numpy.random.uniform(low=-gamma, high=gamma, size=self.dimension)

    def mutiny(self, empires: list[Country], colonies: list[Country]):
        for colony in colonies:
            nearest_imperialist = min(empires, key=lambda x: numpy.linalg.norm(x.location - colony.location))
            if colony.fitness < nearest_imperialist.fitness:
                colony.vassals = nearest_imperialist.vassals
                nearest_imperialist.vassal_of_empire = colony
                empires[nearest_imperialist.index_in_list] = colony
                colony.index_in_list = numpy.copy(nearest_imperialist.index_in_list)
                colonies[colonies.index(colony)] = nearest_imperialist
            else:
                colony.vassal_of_empire = nearest_imperialist

    def empirial_war(self, empires: list[Country], colonies, eta=0.1):
        total_power = 0
        powers_of_empire = []
        normalized_total_power = []
        temp_sum = 0
        num_of_empires = len(empires)

        for empire in empires:
            powers_of_empire.append(empire.fitness + eta * sum([vassal.fitness for vassal in empire.vassals]))
            if total_power < powers_of_empire[-1]:
                total_power = powers_of_empire[-1]
        for i in range(num_of_empires):
            normalized_total_power.append(powers_of_empire[i] - total_power)
            temp_sum += normalized_total_power[-1]

        possession_probability = [numpy.round(normalized_total_power[i] / temp_sum) for i in range(num_of_empires)]
        random_numbers = numpy.random.uniform(low=0, high=1, size=len(empires))
        D = [possession_probability[i] - random_numbers[i] for i in range(num_of_empires)]
        weakest_empire_index = D.index(numpy.min(D))
        strongest_empire_index = D.index(numpy.max(D))
        # if len(empires[weakest_empire_index].vassals) != delta:
        # for _ in empires[weakest_empire_index].vassals:
        # weakest_vassal = empires[weakest_empire_index].weakest_vassal_removal()
        # empires[strongest_empire_index].vassals.append(weakest_vassal)
        # weakest_vassal.vassal_of_empire = empires[strongest_empire_index]
        if len(empires[weakest_empire_index].vassals) != 0:
            weakest_vassal = empires[weakest_empire_index].weakest_vassal_removal()
            empires[strongest_empire_index].vassals.append(weakest_vassal)
            weakest_vassal.vassal_of_empire = empires[strongest_empire_index]
        if len(empires[weakest_empire_index].vassals) == 0:
            empires[strongest_empire_index].vassals.append(empires[weakest_empire_index])
            empires[weakest_empire_index].vassal_of_empire = empires[strongest_empire_index]
            colonies.append(empires[weakest_empire_index])
            for i in range(weakest_empire_index, len(empires)):
                empires[i].index_in_list -= 1
            empires.pop(weakest_empire_index)

    def calculate_fitness(self):
        for country in self.population:
            country.evaluate_fitness(self.objective_function)
            if country.fitness < self.best_fitness:
                self.best_solution = country.location
                self.best_fitness = country.fitness

    def print_number_of_vassals(self, empires: list[Country]):
        for i in range(len(empires)):
            temp = "Number of vassals in empire " + str(i) + " is " + str(len(empires[i].vassals))
            print(temp)

    def animation(self, fig, ax, empires, colonies, old_empires, old_colonies):
        diff_empires = numpy.zeros([len(empires), len(empires)])
        diff_colonies = numpy.zeros([len(colonies), len(colonies)])
        for dimension_idx in range(self.dimension):
            for empire_idx in range(len(empires)):
                diff_empires[empire_idx, dimension_idx] = numpy.abs(
                    empires[empire_idx].location[dimension_idx] - old_empires[empire_idx].location[dimension_idx])
            for colony_idx in range(len(colonies)):
                diff_colonies[colony_idx, dimension_idx] = numpy.abs(
                    colonies[colony_idx].location[dimension_idx] - old_colonies[colony_idx].location[dimension_idx])

        def update_locations(frame):
            ax.clear()
            new_locations_empires = numpy.zeros([len(empires), len(empires)])
            new_locations_colonies = numpy.zeros([len(colonies), len(colonies)])
            for i in range(self.dimension):
                for j in range(len(empires)):
                    new_locations_empires[i, j] = empires[j].location[i] + frame * diff_empires[i, j]
                for j in range(len(colonies)):
                    new_locations_colonies[i, j] = colonies[j].location[i] + frame * diff_colonies[i, j]
            ax.scatter([new_locations_empires[0, j] for j in range(len(empires))],
                       [new_locations_empires[1, j] for j in range(len(empires))], marker="s")
            ax.scatter([new_locations_colonies[0, j] for j in range(len(colonies))],
                       [new_locations_colonies[1, j] for j in range(len(colonies))])
            return fig

        ani = FuncAnimation(fig, update_locations, frames=numpy.arange(0, 100), interval=200)
        plt.show()

    def setup_plot(self, empires, colonies, ax):
        locations_empires = numpy.zeros([len(empires), len(empires)])
        locations_colonies = numpy.zeros([len(colonies), len(colonies)])
        for i in range(self.dimension):
            for j in range(len(empires)):
                locations_empires[i, j] = empires[j].location[i]
            for j in range(len(colonies)):
                locations_colonies[i, j] = colonies[j].location[i]
        ax.scatter(locations_empires[0], locations_empires[1], marker="s", c=[empire.colour for empire in empires])
        ax.scatter(locations_colonies[0], locations_colonies[1], c=[colony.colour for colony in colonies])

    def optimize(self, lb: int, ub: int, beta, gamma, eta):
        self.calculate_fitness()
        self.population.sort(key=lambda x: x.fitness)
        empires = self.create_empires()
        colonies = self.create_colonies(empires)
        for i in range(self.max_iter):
            print("#############Start of " + str(i) + ". iteration. ########################################")
            self.calculate_fitness()
            self.fitness_history.append(self.best_fitness)
            self.assimilation(empires, colonies, beta)
            self.revolution(colonies, gamma)
            self.mutiny(empires, colonies)
            self.empirial_war(empires, colonies, eta)
            self.print_number_of_vassals(empires)
            print("#####################################################################################")
            if len(empires) == 1:
                self.calculate_fitness()
                break
