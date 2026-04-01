import json
import logging
import time

import numpy as np
from eckity.statistics.statistics import Statistics


logger = logging.getLogger(__name__)


class FileLogger(Statistics):
    """Persist per-generation summary statistics to JSON."""

    def __init__(self, format_string=None, output_path=None, save_every_n_generations=50):
        if format_string is None:
            format_string = "best fitness {}\nworst fitness {}\naverage fitness {}\n"
        self.output_path = output_path or "./experiment/results.json"
        self.generation_stats = {
            "mean": [],
            "max": [],
            "min": [],
            "std": [],
            "time": [],
        }
        super().__init__(format_string)
        self.time_from_last_generation = time.time()
        self.save_every_n_generations = save_every_n_generations

    def write_statistics(self, sender, data_dict):
        logger.info("generation #%s", data_dict["generation_num"])
        for index, sub_population in enumerate(data_dict["population"].sub_populations):
            logger.info("subpopulation #%s", index)
            best_individual = sub_population.get_best_individual()
            all_fitness_values = np.array(
                [individual.get_pure_fitness() for individual in sub_population.individuals]
            )

            self.generation_stats["mean"].append(float(np.mean(all_fitness_values)))
            self.generation_stats["max"].append(float(np.max(all_fitness_values)))
            self.generation_stats["min"].append(float(np.min(all_fitness_values)))
            self.generation_stats["std"].append(float(np.std(all_fitness_values)))
            self.generation_stats["time"].append(time.time() - self.time_from_last_generation)
            self.time_from_last_generation = time.time()

            logger.info(
                self.format_string.format(
                    best_individual.get_pure_fitness(),
                    sub_population.get_worst_individual().get_pure_fitness(),
                    sub_population.get_average_fitness(),
                )
            )

        if data_dict["generation_num"] % self.save_every_n_generations == 0:
            self.save_expr()

    def save_expr(self):
        with open(self.output_path, "w", encoding="utf-8") as output_file:
            json.dump(self.generation_stats, output_file)
