import numpy as np

class selector:
    def __init__(self, decision_matrices, number_of_parents_crossover, number_of_parents_survived):
        self.original_parent_decision_matrices = decision_matrices
        self.chosen_parent_decision_matrices = []
        self.number_of_parents_crossover = number_of_parents_crossover
        self.number_of_parents_survived = number_of_parents_survived
    
    def tournament_selector(self, k):
        temp_parent_decision_matrices = self.original_parent_decision_matrices[:]

        for _ in range(self.number_of_parents_crossover):
            random_matrices_selected = np.random.choice(temp_parent_decision_matrices, k)
            selected_decision_matrix = max(random_matrices_selected, key=lambda x: x.fitness_score)
            temp_parent_decision_matrices.remove(selected_decision_matrix)
            self.chosen_parent_decision_matrices.append(selected_decision_matrix)

        return self.chosen_parent_decision_matrices

    def suvival_selector(self):
        sorted_parent_decision_matrices = sorted(self.original_parent_decision_matrices, key=lambda x: x.fitness_score)
        
        return sorted_parent_decision_matrices[self.number_of_parents_survived:]
