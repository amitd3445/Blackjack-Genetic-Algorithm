import pandas as pd
import numpy as np

class matrix_generator:
    def __init__(self, matrix = None):
        if matrix is None:
            self.decision_matrix = generate_random_decision_matrix()
        else:
            self.decision_matrix = generate_decision_matrix(matrix)
        self.fitness_score = None
        
def generate_random_decision_matrix():
    actions = ['H', 'D', 'S', 'P']
    actions_no_split = ['H', 'D', 'S']

    hard_and_soft_strategy = np.random.choice(actions_no_split, (27, 10))
    split_strategy = np.random.choice(actions, (10, 10))
    consolidated_strategy = np.vstack((hard_and_soft_strategy, split_strategy))

    return generate_decision_matrix(consolidated_strategy)

def generate_decision_matrix(matrix):
    index = ['20', '19', '18', '17', '16', '15', '14', '13', '12', '11', '10', '9', '8', '7', '6', '5', '4', 'Soft: 21', 'Soft: 20', 'Soft: 19',
                'Soft: 18', 'Soft: 17', 'Soft: 16', 'Soft: 15', 'Soft: 14', 'Soft: 13', 'Soft: 12', 'Split: A', 'Split: 10', 'Split: 9', 'Split: 8',
                'Split: 7', 'Split: 6', 'Split: 5', 'Split: 4', 'Split: 3', 'Split: 2']
    columns = ['11', '10', '9', '8', '7', '6', '5', '4', '3', '2']  

    return pd.DataFrame(data=matrix, index=index, columns=columns)