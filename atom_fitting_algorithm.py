import numpy as np


class AtomFitter:
    def __init__(self, trap_array, target_configuration, number_of_current_atoms, number_of_target_atoms, number_of_threshold_atoms):
        self.trap_array = trap_array
        self.target_configuration = target_configuration
        # these should be extracted automatically
        # self.number_of_current_atoms = number_of_current_atoms
        # self.number_of_target_atoms = number_of_target_atoms
        # self.number_of_thrershold_atoms = number_of_threshold_atoms

    def validate_inputs(self):
        # TODO write checks to validate simple things like config and trap array hold same shape 

        return None

    def fit(self):
        # check if there are enough atoms for this problem to be solvable 
        if self.number_of_current_atoms - self.number_of_target_atoms < self.number_of_thrershold_atoms:
            return "failure"
        
        # use np.sum to sum each column instead of iterating through array
        column_imbalance = np.sum(self.trap_array, axis = 0) - np.sum(self.target_configuration, axis = 0)
    
    # helper method for 1d array solver to modify initial array to print out to visually inspect
    def apply_moves_to_1d_column(self, column, moves):
        updated = column.copy()
        for from_pos, to_pos in moves:
            updated[from_pos] = 0 
            updated[to_pos] = 1
        return updated 

    def solve_1d_column(self, column_index):
        # take out the column
        current_col = self.trap_array[:, column_index]
        target_col = self.target_configuration[:, column_index]

        # find current positions of atoms 
        current_atoms = np.where(current_col == 1)[0]
        
        # find desired target positions 
        target_atoms = np.where(target_col == 1)[0]

        # sorting atom arrays before matching them is optimal in 1D
        current_atoms = np.sort(current_atoms)
        target_atoms = np.sort(target_atoms)

        moves = []
        total_cost = 0
        
        # loop through each atom-target pair, stopping when we run out of atoms or targets, whichever comes first
        for from_pos, to_pos in zip(current_atoms, target_atoms):
            moves.append((from_pos, to_pos))
            total_cost += abs(from_pos - to_pos)

        return moves, total_cost


