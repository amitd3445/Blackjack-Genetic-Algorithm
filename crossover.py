import matrix_generator as mg
import numpy as np
import pandas as pd

class crossover:    
    def __init__(self, parent_decision_matrices, number_of_children):
        self.parent_decision_matrices = parent_decision_matrices
        self.children_decision_matrices = []
        self.number_of_children = number_of_children

    def uniform_crossover(self):
        while len(self.children_decision_matrices) < self.number_of_children:
            parent_list = self.parent_decision_matrices[:]
            
            parent_one = np.random.choice(parent_list, 1, replace=False)[0]
            parent_one = parent_one.decision_matrix.to_numpy()
            parent_one = parent_one.flatten()

            parent_two = np.random.choice(parent_list, 1)[0]
            parent_two = parent_two.decision_matrix.to_numpy()
            parent_two = parent_two.flatten()

            # determine random indices to pull decision from both parents to use to create child

            all_indices = np.arange(parent_one.size)
            parent_one_indices = np.random.choice(all_indices, 180, replace=False)
            parent_two_indices = np.setdiff1d(all_indices, parent_one_indices)

            child = np.zeros(370, dtype='O')
            child[parent_one_indices] = parent_one[parent_one_indices]
            child[parent_two_indices] = parent_two[parent_two_indices]
            child = child.reshape((37, 10))

            self.children_decision_matrices.append(mg.matrix_generator(child))   

        return self.children_decision_matrices         