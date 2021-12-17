from random import random
from time import time
import numpy as np
import matplotlib.pyplot as plt


class Ant(object):
    """
    This class represents an ant and stores the information about its route and fitness.
    Attributes
    ----------
    route : array(tuple(int, int))
        a list of coordinates that represent the bin-item configuration.
    fitness : int
        the current fitness of the ants route.
    bins : array(Bin)
        holds the bin configuration if the ant is chosen as a generational champion.
    Methods
    -------
    distribute_pheromones(graph)
        Distributes a pheromone weight on the graph at the positions defined in the route attribute.
    copy()
        Similar to deepcopy - creates a replica object of the ant.
    get_route_as_str()
        Format the route in a human readable format.
    """
    route = []
    bins = []
    fitness = -1

    def distribute_pheromones(self, graph):
        """Distributes a pheromone weight on the graph at the positions defined in the route attribute."""
        pheromone_weight = 100.0 / self.fitness
        previous_bin = 0
        for new_bin, item in self.route:
            graph.graph[previous_bin, item, new_bin] += pheromone_weight
            previous_bin = new_bin

    def copy(self):
        """Creates a replica object of the ant."""
        new_ant = Ant()
        new_ant.route = [r for r in self.route]
        new_ant.bins = self.bins.copy()
        new_ant.fitness = self.fitness
        return new_ant

    def get_route_as_str(self):
        """Format the route in a human readable format."""
        return " -> ".join("Item %d in Bin %d" % (point[1] + 1, point[0]) for point in self.route)


class Bin(object):
    """
    This class represents bin that can hold items and caches the current fitness.
    Attributes
    ----------
    total_weight : int
        the sum of the items in the bin.
    items : array(int)
        the item weights currently in the bin.
    Methods
    -------
    add_item(item)
        Add an item to the bin and increase the total weight.
    copy()
        Similar to deepcopy - creates a replica object of the bin.
    empty()
        Reset the contents of the bin.
    """
    total_weight = 0
    items = []

    def __repr__(self):
        return "Bin with %d items weighing %d: %s" \
            % (len(self.items), self.total_weight, self.items)

    def add_item(self, item):
        """Add an item to the bin and increase the total weight."""
        self.items.append(item)
        self.total_weight += item

    def copy(self):
        """Creates a replica object of the bin."""
        bin_copy = Bin()
        bin_copy.total_weight = self.total_weight
        bin_copy.items = [item for item in self.items]
        return bin_copy

    def empty(self):
        """Reset the contents of the bin."""
        self.items = []
        self.total_weight = 0


class Graph(object):
    """
    This class represents the pheromone weights across the bin-item matrix.
    Attributes
    ----------
    graph : np.array(int)
        3-d array of pheromone weights.
    evaporation_rate : float
        Scalar value to evaporate the pheromone weights.
    Methods
    -------
    evaporate()
        Reduce the pheromone weights across the graph.
    """
    def __init__(self, bins, items):
        self.graph = np.random.rand(bins, items, bins)
        # print(self.graph)
        # print(np.shape(self.graph))

    def evaporate(self, evaporation):
        """Reduce the pheromone weights across the graph."""
        self.graph *= evaporation


def generate_bin_items(quantity=500, bpp1=True):
    """This function creates an array of bin weights."""
    if bpp1:
        return [i for i in range(1, quantity + 1)]
    return [(i ** 2) for i in range(1, quantity + 1)]


def create_bins(quantity):
    """This function creates a number of bins and returns them as an array"""
    return [Bin() for _ in range(quantity)]


class ACO(object):
    """This class holds all relevant information for objects required for running the ACO algorithm.
    ...
    Attributes
    ----------
    bins : Bin
        a bin object that holds items and a total weight.
    items : array(int)
        an array of integers representing the weights of items.
    ants : array(Ant)
        an array of Ant objects to be controlled during the algorithms run.
    best_ant : Ant
        an ant object - the best ant of the final generation of a algorithm run.
    graph : Graph
        a graph object to store the pheromone weights.
    num_of_evaluations : int
        the number of routes evaluated.
    limit : int
        the maximum number of evaluations allowed.
    verbose : boolean
        whether or not to print to the console when log() is called.
    ran : boolean
        has the ACO been run.
    runtime : float
        time duration of the last run.
    avg_fitness : array(float)
        the timeseries of average fitnesses over each cycle.
    Methods
    -------
    summary()
        prints a summary of the last run if there is one.
    stats()
        returns the best fitness and time elapsed over last run if there is one.
    run()
        runs the ACO algorithm.
    explore()
        runs one cycle of route creation and evaporation.
    ant_run(ant)
        reset the ant and recreate its route.
    create_route(ant)
        create a route through the graph of pheromones.
    route_step(prev_bin, item)
        return a step from the current bin to the next bin position.
    route_fitness()
        calculate the fitness for the current bin configuration.
    set_champion()
        set the best ant for the current generation.
    empty_bins()
        reset all bins.
    log(message)
        prints to the console if verbose is true.
    graph_averages()
        create a graph using the data from avg_fitness.
    """

    def __init__(self, bins, items, population, evaporation, limit=10000, verbose=False):
        """Initialise the ACO object with the required parameters."""
        self.bins = bins
        self.items = items

        self.ants = [Ant() for _ in range(population)]
        self.best_ant = None

        self.graph = Graph(len(bins), len(items))

        self.evaporation = evaporation
        self.num_paths = 0
        self.limit = limit
        self.verbose = verbose

        self.num_of_evaluations = 0
        self.ran = False
        self.runtime = 0
        self.best_ant = None
        self.avg_fitness = []

    def summary(self):
        """Print a summary of the last run if there is one."""
        if hasattr(self, 'ran') and self.ran:
            print("Run was successful and took %d seconds." % int(self.runtime))
            print("--- Best fitness: %d" % self.best_ant.fitness)
            # displays the full bin configuration for the best ant in the final set
            # print("--- Best bin config:")
            # for i, b in enumerate(self.best_ant.bins):
            #     print("%4d. %s" % (i + 1, b))
        return self.best_ant.fitness

    def stats(self):
        """Return the best fitness achieved in the final generation and the time taken to run the ACO"""
        if hasattr(self, 'ran') and self.ran:
            return self.best_ant.fitness, self.runtime

    def run(self):
        """Runs a full ACO run."""
        self.ran = False
        self.best_fits = []
        self.avg_fitness = []
        start_time = time()

        while self.num_of_evaluations < self.limit:
            self.explore()

        self.set_champion()

        self.ran = True
        self.runtime = time() - start_time

    def explore(self):
        """Create a route for all ants and evaporate the graph."""
        self.ants = [*map(self.create_path, self.ants)]
        best = None
        for ant in self.ants:
            ant.distribute_pheromones(self.graph)
        fitnesses = [ant.fitness for ant in self.ants]
        self.best_fits.append(min(fitnesses) / sum(self.items))
        self.avg_fitness.append(sum(fitnesses) / len(fitnesses))
        self.graph.evaporate(self.evaporation)

    def create_path(self, ant):
        """Reset the bins and create a route for the given ant.

        :param ant: ant object
        :returns: ant with new path
        """
        self.empty_bins()
        ant = self.create_route(ant)
        ant.bins = self.bins.copy()
        return ant

    def create_route(self, ant):
        """Calculate a route through the pheromone graph."""
        current_bin = 0
        ant.route = []
        for item in enumerate(self.items):
            current_bin, item = self.route_step(current_bin, item)
            ant.route.append((current_bin, item))

        ant.fitness = self.route_fitness()
        self.num_of_evaluations += 1
        return ant

    def route_step(self, current_bin, item):
        """Get the index of the next bin to place the item in.
        
        :param currentBin: index of the current bin
        :param item: item weight
        :returns: next bin index
        """
        column = self.graph.graph[current_bin][item[0]].tolist()
        total = sum(column)
        threshold = total * random()

        current = 0.0
        for index, weight in enumerate(column):
            if current + weight >= threshold:
                self.bins[index].add_item(item[1])
                return index, item[0]
            current += weight

    def route_fitness(self):
        """Calculate the fitness of the current bin configuration.
        
        :returns: current fitness
        """
        max_weight = self.bins[0].total_weight
        min_weight = self.bins[0].total_weight

        for b in self.bins:
            if b.total_weight > max_weight:
                max_weight = b.total_weight
            if b.total_weight < min_weight:
                min_weight = b.total_weight

        return max_weight - min_weight

    def set_champion(self):
        """Allocate the best ant of the generation to the best_ant."""
        for ant in self.ants:
            if self.best_ant and ant.fitness < self.best_ant.fitness:
                    self.best_ant = ant.copy()
            elif not self.best_ant:
                self.best_ant = ant.copy()

    def empty_bins(self):
        """Resets the bin configuration."""
        [b.empty() for b in self.bins]

    def log(self, message):
        """Prints a message to the console if verbose is true."""
        if self.verbose:
            print(message)

    def graph_averages(self):
        """Output a graph to the user based on the values in avg_fitness"""
        plt.plot(self.avg_fitness)
        plt.title("Average Fitness Over Time - BPP" + str(val) + " where p = " + str(
            population) + " and e = " + str(evaporation))
        plt.xlabel('Number of Iterations')
        plt.ylabel('Average Fitness Value')
        plt.show()


if __name__ == '__main__':
    val = input("Please select BPP problem. Write 1 to run BPP1, 2 to run BPP2. ")

    if val == "1":
        bins = create_bins(10)
        items = generate_bin_items(quantity=500)
    elif val == "2":
        bins = create_bins(50)
        items = generate_bin_items(quantity=500, bpp1=False)
    # print(items)

    print("Running single ACO trial with %d bins and %d items."
          % (len(bins), len(items)))

    population = int(input("Specify population size: "))
    evaporation = float(input("Specify evaporation rate: "))

    trial = ACO(bins, items, population, evaporation)
    trial.run()
    trial.graph_averages()

    print("Run took %d seconds." % int(trial.runtime))
    print("Best fitness: %d" % trial.best_ant.fitness)
