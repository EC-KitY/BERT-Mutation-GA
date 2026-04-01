from eckity.creators import GAVectorCreator
from eckity.genetic_encodings.ga import IntVector


class GAIntegerStringVectorCreator(GAVectorCreator):
    def __init__(self, length=1, bounds=(0, 1), gene_creator=None, events=None):
        super().__init__(
            length=length,
            bounds=bounds,
            gene_creator=gene_creator,
            vector_type=IntVector,
            events=events,
        )

    def individual_from_vector(self, vector):
        individual = self.type(
            length=self.length,
            bounds=self.bounds,
            fitness=self.fitness_type(higher_is_better=True),
        )
        individual.set_vector(vector)
        return individual
