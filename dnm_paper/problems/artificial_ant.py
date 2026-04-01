from pathlib import Path
from typing import Union

import numpy as np
from eckity.genetic_encodings.ga import IntVector


def load_map(path: Union[str, Path]) -> np.ndarray:
    matrix = []
    with open(path, "r", encoding="utf-8") as map_file:
        for line in map_file:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            for cell in stripped_line:
                assert cell in [".", "S", "#"], stripped_line
            matrix.append(list(stripped_line))
    return np.array(matrix)


class ArtificialAntEvaluator:
    START_LOCATION_TOKEN = "S"
    FOOD_TOKEN = "#"
    EMPTY_CELL_TOKEN = "."
    DIRECTION_MAP = {
        0: (1, 0),
        1: (-1, 0),
        2: (0, -1),
        3: (0, 1),
    }

    def __init__(self, map_array, penalize_out_of_bounds=0.0):
        self.map_array = np.array([list(row) for row in map_array])
        start_location = np.argwhere(self.map_array == self.START_LOCATION_TOKEN)
        if start_location.size == 0:
            raise ValueError("Map must contain a start location token 'S'.")
        if start_location.shape[0] > 1:
            raise ValueError("Map must contain only one start location token 'S'.")
        self.start_location = tuple(start_location[0])
        self.fitness_dict = {}
        self.penalize_out_of_bounds = penalize_out_of_bounds

    def evaluate_individual(self, individual, fitness_dict=None):
        fitness_dict = self.fitness_dict if fitness_dict is None else fitness_dict
        if isinstance(individual, IntVector):
            individual = individual.vector

        individual_key = tuple(individual)
        if individual_key in fitness_dict:
            return fitness_dict[individual_key]

        row, col = self.start_location
        fitness = 0.0
        map_copy = self.map_array.copy()

        for move in individual:
            if move not in self.DIRECTION_MAP:
                raise ValueError(f"Invalid move {move}. Valid moves are: {list(self.DIRECTION_MAP.keys())}.")

            row_delta, col_delta = self.DIRECTION_MAP[move]
            new_row, new_col = row + row_delta, col + col_delta
            if 0 <= new_row < self.map_array.shape[0] and 0 <= new_col < self.map_array.shape[1]:
                row, col = new_row, new_col
                if map_copy[row, col] == self.FOOD_TOKEN:
                    fitness += 1
                    map_copy[row, col] = self.EMPTY_CELL_TOKEN
            else:
                fitness -= self.penalize_out_of_bounds

        fitness_dict[individual_key] = fitness
        return fitness
