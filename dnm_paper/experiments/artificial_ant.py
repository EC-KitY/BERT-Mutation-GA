import argparse
import logging
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Optional

from eckity.algorithms import SimpleEvolution
from eckity.breeders import SimpleBreeder
from eckity.subpopulation import Subpopulation

from dnm_paper.config import EvolutionConfig, MutationModelConfig
from dnm_paper.experiments.common import (
    ECKITYEvaluatorAdapter,
    build_bert_mutation_operator,
    build_variation_operators,
)
from dnm_paper.individuals import GAIntegerStringVectorCreator
from dnm_paper.logging_utils import FileLogger
from dnm_paper.problems.artificial_ant import ArtificialAntEvaluator, load_map


logger = logging.getLogger(__name__)


def run_artificial_ant_experiments(
    evolution_config: EvolutionConfig,
    mutation_config: Optional[MutationModelConfig] = None,
):
    mutation_config = mutation_config or MutationModelConfig()
    crossover_operator, selection_operator = build_variation_operators(
        crossover_probability=evolution_config.crossover_probability,
        crossover_n_points=evolution_config.crossover_n_points,
        tournament_size=evolution_config.tournament_size,
    )

    for instance_name, individual_length in evolution_config.instances_to_run.items():
        instance_path = evolution_config.maps_dir / instance_name
        if not instance_path.exists():
            raise FileNotFoundError(f"Map file does not exist: {instance_path}")

        logger.info("Running instance %s", instance_path)
        instance_matrix = load_map(instance_path)
        evaluator = ArtificialAntEvaluator(instance_matrix)
        mutation_operator = build_bert_mutation_operator(
            mutation_probability=evolution_config.mutation_probability,
            population_size=evolution_config.population_size,
            individual_evaluator=evaluator,
            input_sequence_length=individual_length,
            max_int_val=3,
            mutation_config=mutation_config,
        )
        mutation_operator.mutation_operator.fitness_dict = evaluator.fitness_dict
        creator = GAIntegerStringVectorCreator(length=individual_length, bounds=(0, 3))

        for run_index in range(evolution_config.runs):
            evaluator.fitness_dict.clear()
            run_output_path = evolution_config.output_root / instance_path.stem / "bert_mutation" / str(run_index)
            run_output_path.mkdir(parents=True, exist_ok=True)
            statistics_logger = FileLogger(output_path=str(run_output_path / "results.json"))

            evolution = SimpleEvolution(
                Subpopulation(
                    creators=creator,
                    population_size=evolution_config.population_size,
                    evaluator=ECKITYEvaluatorAdapter(evaluator),
                    higher_is_better=True,
                    elitism_rate=0.0,
                    operators_sequence=[crossover_operator, mutation_operator],
                    selection_methods=[(selection_operator, 1)],
                ),
                breeder=SimpleBreeder(),
                max_generation=evolution_config.generations,
                max_workers=1,
                statistics=statistics_logger,
                random_seed=int(time.time()),
            )

            try:
                evolution.evolve()
            except Exception:
                exception_path = run_output_path / "exception_log.txt"
                with open(exception_path, "w", encoding="utf-8") as exception_file:
                    exception_file.write(traceback.format_exc())
                logger.exception("Experiment failed for %s run %s", instance_name, run_index)
            finally:
                statistics_logger.save_expr()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the artificial ant experiments from the DNM paper.")
    parser.add_argument(
        "--maps-dir",
        type=Path,
        default=None,
        help="Directory containing artificial ant map files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where run artifacts will be written.",
    )
    parser.add_argument("--generations", type=int, default=None, help="Number of evolutionary generations.")
    parser.add_argument("--population-size", type=int, default=None, help="Population size for each run.")
    parser.add_argument("--runs", type=int, default=None, help="Number of repeated runs per map.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    config = EvolutionConfig()
    if args.maps_dir is not None:
        config = replace(config, maps_dir=args.maps_dir)
    if args.output_dir is not None:
        config = replace(config, output_root=args.output_dir)
    if args.generations is not None:
        config = replace(config, generations=args.generations)
    if args.population_size is not None:
        config = replace(config, population_size=args.population_size)
    if args.runs is not None:
        config = replace(config, runs=args.runs)
    run_artificial_ant_experiments(config)


if __name__ == "__main__":
    main()
