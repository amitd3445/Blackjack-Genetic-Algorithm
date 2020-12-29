import blackjack_actions as ba
import matrix_generator as mg
import selector as sel
import crossover as cross
import mutator as mut
import concurrent.futures
import statistics

def testing_strategy(child):
    blackjack_series = ba.Series(number_of_hands=25000, decision_matrix=child.decision_matrix, number_of_decks=6, minimum_number_of_cards=50, dealer_max_total_to_hit = 17)
    blackjack_series.start_series()
    child.fitness_score = blackjack_series.cummulative_results

    return child

def create_next_generation(children, number_of_parents_crossover, number_of_parents_survived, tourny_k, population_size, mutation_rate):
    selector = sel.selector(children, number_of_parents_crossover, number_of_parents_survived)
    parent_decision_matrices = selector.tournament_selector(tourny_k)

    crossover = cross.crossover(parent_decision_matrices, population_size-number_of_parents_survived)
    children = crossover.uniform_crossover()

    mutator = mut.mutator(children, mutation_rate)
    children = mutator.random_reset()

    return children

if __name__ == '__main__':

    population_size = 100
    number_of_generations = 50
    children = [mg.matrix_generator() for _ in range(population_size)]

    for _ in range(number_of_generations):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            children = list(executor.map(testing_strategy, children))
        print(max(children, key=lambda x: x.fitness_score).decision_matrix, statistics.mean([child.fitness_score for child in children]))
        children = create_next_generation(children, 25, 0, 5, population_size, 0.05)
