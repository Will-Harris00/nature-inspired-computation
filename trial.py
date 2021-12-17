from operator import itemgetter
from time import time

from aco import ACO
from aco import create_bins
from aco import generate_bin_items


def run_bpp(num_of_bins=10, bpp1=True, num_of_items=500, num_of_trials=5):
    """Run a full bin packing problem test with the specified parameters and return the results."""

    results = []
    rules = [
        {'population': 100, 'evaporation': 0.9},
        {'population': 100, 'evaporation': 0.5},
        {'population': 10,  'evaporation': 0.9},
        {'population': 10,  'evaporation': 0.5},
    ]

    # Test each rule configuration.
    for rule in rules:
        print("Current trial Parameters: Bins=%d, Items=%d, BPP1=%s, Population=%d, Evaporation=%.1f" %
              (num_of_bins, num_of_items, bpp1, rule['population'], rule['evaporation'])
              )

        result = run_test(num_of_bins, num_of_items, rule['population'], rule['evaporation'], bpp1=bpp1,  num_of_trials=num_of_trials, verbose=True)
        results.append(result)

        print(" -- Achieved average fitness %.1f in %.2f seconds.\n" % (result['average_fitness'], result['average_time']))

        print("MIN: %s, MAX: %s, Fitness AVG: %.1f" % (
            result['min_fitness'],
            result['max_fitness'],
            result['average_fitness']
        ))

        print("MIN: %.2f, MAX: %.2f, Time AVG: %.2f\n" % (
            result['min_time'],
            result['max_time'],
            result['average_time']
        ))
    return results


def run_test(num_of_bins, num_of_items, population, evaporation, bpp1=True, num_of_trials=5, verbose=False):
    """Run ACO trial and return an object of the calculated results.

    :param num_of_bins: b number of bins
    :param num_of_items: k number of items
    :param population: p population size
    :param evaporation: e evaporation
    :param bpp1: boolean indicating which problem to run
    :param numOfTrials: number of wanted trials for each rule
    """
    results = []
    average_fitness = 0
    average_time = 0
    final_best_ant = 0

    total_time = 0

    # Run 5 tests of the ACO algorithm and compile a set of averages.
    for i in range(num_of_trials):
        bins = create_bins(num_of_bins)
        items = generate_bin_items(quantity=num_of_items, bpp1=bpp1)

        trial = ACO(bins, items, population, evaporation, verbose=False)
        trial.run()

        final_best_ant = trial.summary() # shows the best ant for final generation of last run

        fitness, time = trial.stats()
        results.append((fitness, time))
        average_fitness += fitness * 0.2
        average_time += time * 0.2

    log("Test finished in %d seconds." % total_time, verbose)
    log("Stats:", verbose)
    log(" -- Average Fitness of Test: %f" % average_fitness, verbose)
    log(" -- Average Time Per Test Run: %f" % average_time, verbose)
    log(" -- Final Best Ant: %d\n" % final_best_ant, verbose)

    return {
        'raw_results': results,
        'bins': num_of_bins,
        'items': num_of_items,
        'population': population,
        'evaporation': evaporation,
        'bpp1': bpp1,
        'final_best_ant': final_best_ant,
        'min_fitness': min(results, key=itemgetter(0))[0],
        'max_fitness': max(results, key=itemgetter(0))[0],
        'average_fitness': average_fitness,
        'min_time': min(results, key=itemgetter(1))[1],
        'max_time': max(results, key=itemgetter(1))[1],
        'average_time': average_time,
        'total_time': total_time
    }


def pretty_print_results(results):
    """This function helps to format the results object in a readable format."""
    for i, bpp_results in enumerate(results):
        print("\nResult Set For BPP%s\n" % str(i+1))
        for j, test in enumerate(bpp_results):
            print("Test Conditions: - B=%d, I=%d, P=%d, E=%f, BPP1=%s" %
                  (test['bins'], test['items'], test['population'], test['evaporation'], test['bpp1'])
                  )
            print("Best Ant: %i,  MIN: %d,  MAX: %d,  Fitness - AVG: %6.1f,  MIN: %6.2f,  MAX: %6.2f,  Time - AVG: %6.2f,\n" % (
                test['final_best_ant'],
                test['min_fitness'],
                test['max_fitness'],
                test['average_fitness'],
                test['min_time'],
                test['max_time'],
                test['average_time']
            ))


def log(message, verbose=False):
    """Output to the console if verbose is true."""
    if verbose:
        print(message)


if __name__ == "__main__":
    print("Starting Full Test...")
    start_time = time()
    total_res = []
    print("Running Bin-Packing Problem 1\nBPP1 Parameters: 10 Bins and 500 Items with i weights\n")
    total_res.append(run_bpp())
    print("Finished BPP1.")
    print("Running Bin-Packing Problem 2\nBPP2 Parameters: 50 Bins and 500 Items with i^2 weights\n")
    total_res.append(run_bpp(50, False))
    print("Finished BPP2.")

    print("Full program executed in %.2f" % float(time() - start_time))
    print("Results...\n\n")
    pretty_print_results(total_res)
