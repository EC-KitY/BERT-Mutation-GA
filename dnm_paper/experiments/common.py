from __future__ import annotations

from eckity.genetic_operators import TournamentSelection, VectorKPointsCrossover

from dnm_paper.config import MutationModelConfig
from dnm_paper.mutation import BertMutation, EckityCustomMutation


class ECKITYEvaluatorAdapter:
    def __init__(self, base_evaluator):
        self.base_evaluator = base_evaluator

    def evaluate_individual(self, individual):
        return self.base_evaluator.evaluate_individual(individual.vector)


def build_bert_mutation_operator(
    mutation_probability: float,
    population_size: int,
    individual_evaluator,
    input_sequence_length: int,
    max_int_val: int,
    mutation_config: MutationModelConfig,
) -> EckityCustomMutation:
    bert_instance = BertMutation(
        max_int_val=max_int_val,
        get_fitness_func=individual_evaluator.evaluate_individual,
        batch_size=mutation_config.batch_size,
        epsilon_greedy=mutation_config.epsilon_greedy,
        context_size=input_sequence_length + 1,
        higher_is_better=mutation_config.higher_is_better,
        full_trajectory_query=mutation_config.full_trajectory_query,
        learning_rate=mutation_config.learning_rate,
        mask_probability=mutation_config.mask_probability,
        word_embedding_dim=mutation_config.word_embedding_dim,
        internal_size=mutation_config.internal_size,
        n_layers=mutation_config.n_layers,
        n_attention_heads=mutation_config.n_attention_heads,
        normalize_batches=mutation_config.normalize_batches,
        best_ind_sample_size=mutation_config.best_ind_sample_size,
    )
    return EckityCustomMutation(
        population_size=population_size,
        probability=mutation_probability,
        mutation_operator=bert_instance,
    )


def build_variation_operators(
    crossover_probability: float,
    crossover_n_points: int,
    tournament_size: int,
    higher_is_better: bool = True,
):
    crossover_operator = VectorKPointsCrossover(
        probability=crossover_probability,
        k=crossover_n_points,
    )
    selection_operator = TournamentSelection(
        tournament_size=tournament_size,
        higher_is_better=higher_is_better,
    )
    return crossover_operator, selection_operator
