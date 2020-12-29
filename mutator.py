import matrix_generator as mg
import numpy as np

class mutator:
    def __init__(self, children_decision_matrices, mutation_rate):
        self.children_decision_matrices_nonmutated = children_decision_matrices
        self.children_decision_matrices_mutated = []
        self.number_of_mutations = int(370 * mutation_rate)

    def random_reset(self):
        actions = ['H', 'D', 'S', 'P']
        actions_no_split = ['H', 'D', 'S']

        for child in self.children_decision_matrices_nonmutated:
            child_decision_matrix_flaten = child.decision_matrix.to_numpy().flatten()
            mutation_indices = np.random.choice(np.arange(360), self.number_of_mutations)
            
            for mutation_index in mutation_indices:
                
                if mutation_index < 270:
                    child_decision_matrix_flaten[mutation_index] = np.random.choice(actions_no_split, 1)[0]
                else:
                    child_decision_matrix_flaten[mutation_index] = np.random.choice(actions, 1)[0]
                
            child.decision_matrix = mg.generate_decision_matrix(child_decision_matrix_flaten.reshape((37, 10)))      
            self.children_decision_matrices_mutated.append(child)
        
        return self.children_decision_matrices_nonmutated