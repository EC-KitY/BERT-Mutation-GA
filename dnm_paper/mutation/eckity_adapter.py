import numpy as np
from eckity.genetic_operators import GeneticOperator


class EckityCustomMutation(GeneticOperator):
    """Wrap a mutation model so it can be applied through ECKITY."""

    def __init__(self, mutation_operator, population_size, probability=1.0, events=None):
        super().__init__(probability=1.0, arity=population_size, events=events)
        self.mutation_operator = mutation_operator
        self.mutation_probability = probability

    def apply(self, individuals):
        mutation_masks = np.random.rand(len(individuals)) < self.mutation_probability
        individuals_matrix = np.array([individual.vector for individual in individuals])[mutation_masks]

        if len(individuals_matrix) == 0:
            return individuals

        mutated_individuals = self.mutation_operator.get_mutation(individuals_matrix)
        eckity_mutated_individuals = [
            individual for index, individual in enumerate(individuals) if mutation_masks[index]
        ]
        assert len(mutated_individuals) == len(
            eckity_mutated_individuals
        ), "Mutated individuals length mismatch"

        for index, individual in enumerate(eckity_mutated_individuals):
            individual.set_vector(list(mutated_individuals[index]))

        return individuals
