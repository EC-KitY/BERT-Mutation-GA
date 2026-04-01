from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "experiments" / "artificial_ant" / "runs"
DEFAULT_MAPS_DIR = PROJECT_ROOT / "artifical_ant_maps"
DEFAULT_FROZEN_LAKE_RESULTS_ROOT = PROJECT_ROOT / "experiments" / "frozen_lake" / "runs"


@dataclass(frozen=True)
class MutationModelConfig:
    batch_size: int = 64
    learning_rate: float = 1e-3
    epsilon_greedy: float = 0.05
    mask_probability: float = 0.1
    word_embedding_dim: int = 60
    internal_size: int = 128
    n_layers: int = 3
    n_attention_heads: int = 3
    normalize_batches: bool = False
    best_ind_sample_size: int = 100
    full_trajectory_query: bool = False
    higher_is_better: bool = True


@dataclass(frozen=True)
class EvolutionConfig:
    generations: int = 1000
    population_size: int = 100
    mutation_probability: float = 0.4
    crossover_probability: float = 0.75
    crossover_n_points: int = 2
    tournament_size: int = 4
    runs: int = 1
    output_root: Path = DEFAULT_RESULTS_ROOT
    maps_dir: Path = DEFAULT_MAPS_DIR
    instances_to_run: dict[str, int] = field(
        default_factory=lambda: {
            "aux_map1.txt": 283,
            "aux_map2.txt": 286,
            "john_muir_trail.txt": 200,
            "los_altos.txt": 800,
            "santafe_trail.txt": 400,
        }
    )


@dataclass(frozen=True)
class FrozenLakeConfig:
    generations: int = 100
    population_size: int = 100
    mutation_probability: float = 0.4
    crossover_probability: float = 0.75
    crossover_n_points: int = 2
    tournament_size: int = 4
    runs: int = 1
    total_episodes: int = 2000
    output_root: Path = DEFAULT_FROZEN_LAKE_RESULTS_ROOT
