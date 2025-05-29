import numpy as np


class AtomFitter:
    def __init__(self, trap_array, target_configuration, number_of_threshold_atoms):
        self.trap_array = trap_array
        self.target_configuration = target_configuration
        self.number_of_current_atoms = int(np.sum(trap_array))
        self.number_of_target_atoms = int(np.sum(target_configuration))
        self.shape = trap_array.shape
        self.number_of_threshold_atoms = number_of_threshold_atoms

    def validate_inputs(self):
        # TODO write checks to validate simple things like config and trap array hold same shape 
        
        # check if there are enough atoms for this problem to be solvable 
        if self.number_of_current_atoms - self.number_of_target_atoms < self.number_of_threshold_atoms:
            return "failure, not enought atoms in trap"
        
        if self.trap_array.shape != self.target_configuration.shape:
            raise ValueError("trap array and target config array must have the same shape")

    def fit(self):
        
        self.validate_inputs()

        # use np.sum to sum each column instead of iterating through array
        column_imbalance = np.sum(self.trap_array, axis = 0) - np.sum(self.target_configuration, axis = 0)

        # track moves + cost
        tracked_moves = []
        total_cost = 0

        solved_columns = set() 

        # solve all "neutral" columns aka enough atoms but not in right position
        for col_index, imbalance in enumerate(column_imbalance):
            if imbalance == 0:
                moves, cost = self.solve_1d_column(col_index)
                tracked_moves.append((col_index, moves))
                total_cost += cost
                solved_columns.add(col_index)

        # label remaining columns

        donors = []
        receivers = []

        # label remaining columns 
        for col_index, imbalance in enumerate(column_imbalance):
            if imbalance > 0:
                donors.append((col_index, imbalance))
            elif imbalance < 0:
                receivers.append((col_index, imbalance))

        # time to redistrubute between donor/receiver columns
        while receivers:
            # select nearest donor-receiver pair and solve + redistribute
            donor_index, _ = donors.pop(0)
            receiver_index, _ = receivers.pop(0)

            moves, cost = self.redistribute_atoms_between_columns(donor_index, receiver_index)
            tracked_moves.append(((donor_index, receiver_index), moves))
            total_cost += cost

            solved_columns.add(donor_index)
            solved_columns.add(receiver_index)
        
        # solve last donor if exists
        for donor_index, _ in donors:
            if donor_index not in solved_columns:
                moves, cost = self.solve_1d_column(donor_index)
                tracked_moves.append((donor_index, moves))
                total_cost += cost
                solved_columns.add(donor_index)

    def redistribute_atoms_between_columns(self, donor_index, receiver_index):
        # identify atom positions in donor and target
        donor_col_array = self.trap_array[:, donor_index]
        receiver_col_array = self.trap_array[:, receiver_index]

        donor_atoms = np.where(donor_col_array == 1)[0]
        receiver_atoms = np.where(receiver_col_array == 0)[0]

        # number of atoms to redistribute
        num_to_redistribute = len(receiver_atoms)

        # choose atoms starting from the ones located the farthest away from the target region 
        receiver_center = np.mean(receiver_atoms)
        sorted_donor_atoms = sorted(donor_atoms, key = lambda x: abs(x - receiver_center), reverse=True)
        redistributed_atoms = sorted_donor_atoms[:num_to_redistribute]

        # assign these atoms to distribution rows 
        distribution_rows = []
        for r in range(self.shape[0]):
            if self.trap_array[r, receiver_index] == 0 and self.trap_array[r, donor_index] == 0:
                distribution_rows.append(r)

        # distribution_rows = distribution_rows[:num_to_redistribute]
        
        moves = []
        total_cost = 0
        
        for donor_row, dist_row in zip(redistributed_atoms, distribution_rows):
            # extract from donor_row
            moves.append(("extract", (donor_row, donor_index)))
            total_cost += 1  # extraction cost

            # move to dist_row - vertical placement
            if donor_row != dist_row:
                moves.append(("vertical move", (donor_row, dist_row)))
                total_cost += abs(donor_row - dist_row)
            
            # move horizontally to receiver
            moves.append(('horizontal move'), (donor_index, receiver_index))
            total_cost += abs(donor_index - receiver_index)

            # implant into receiver column
            moves.append(("implant", (dist_row, receiver_index)))
            total_cost += 1

        # reconfiugre receiver column with 1d algo
        receiver_moves, receiver_cost = self.solve_1d_column(receiver_index)
        moves.extend([("1d move", move) for move in receiver_moves])
        total_cost += receiver_cost
            
        return moves, total_cost


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


