import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import atom_fitting_algorithm as afa

def test_solve_1d_column():
    trap_array = np.array([
        [0, 1],
        [1, 0],
        [0, 1],
        [1, 0],
    ])

    target_configuration = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ])

    fitter = afa.AtomFitter(trap_array, target_configuration, 0, 0, 0)

    moves, total_cost = fitter.solve_1d_column(column_index=0)
    moves2, total_cost2 = fitter.solve_1d_column(column_index=1)

    expected_moves = [(1, 0), (3,1)]
    expected_cost = abs(1 - 0) + abs(3 - 1)

    assert moves == expected_moves, f"Expected moves {expected_moves}, but got {moves}"
    assert total_cost == expected_cost, f"Expected total cost {expected_cost}, but got {total_cost}"
    
    updated_column = fitter.apply_moves_to_1d_column(trap_array[:, 0], moves)
    updated_column2 = fitter.apply_moves_to_1d_column(trap_array[:, 1], moves2)
    updated_array = np.column_stack((updated_column, updated_column2))
    print("Updated array:")
    print(updated_array)
    print("Target configuration:")
    print(target_configuration)
    print("1d column test passed")

test_solve_1d_column()



