import gymnasium as gym
import numpy as np
from eckity.genetic_encodings.ga import IntVector

from dnm_paper.problems.frozen_lake_instances import DEFAULT_8X8_MAP


def choose_action(state: int, vector: np.ndarray) -> int:
    """Return the action encoded for a given state."""

    return int(vector[state])


class FrozenLakeEvaluator:
    def __init__(
            self,
            frozenlake_map=None,
            total_episodes=2000,
            is_slippery=True,
            normalize_score=True,
            fitness_dict=None,
            use_fitness_dict=False,
    ):
        if frozenlake_map is None:
            frozenlake_map = DEFAULT_8X8_MAP

        self.total_episodes = total_episodes
        self.normalize_score = normalize_score

        self.map = frozenlake_map
        self.env = gym.make(
            "FrozenLake-v1",
            desc=self.map,
            is_slippery=is_slippery,
        )
        self.map_size = len(frozenlake_map)
        self.holes = [i * self.map_size + j
                      for i in range(self.map_size)
                      for j in range(self.map_size)
                      if self.map[i][j] == 'H']

        if fitness_dict is None:
            fitness_dict = {}

        self.fitness_dict = fitness_dict
        self.use_fitness_dict = use_fitness_dict

    def evaluate_individual(self, individual, fitness_dict=None):
        if fitness_dict is None:
            fitness_dict = self.fitness_dict

        if not self.use_fitness_dict:
            fitness_dict.clear()

        if isinstance(individual, IntVector):
            individual = individual.vector

        individual_key = tuple(individual)
        if individual_key in fitness_dict:
            return fitness_dict[individual_key]

        vector = np.copy(individual)

        for hole in self.holes:
            vector[hole] = -1

        score = 0.0
        for _ in range(self.total_episodes):
            state = self.env.reset()[0]  # Reset the environment
            done = False
            total_rewards = 0

            while not done:
                action = choose_action(state=state, vector=vector)

                new_state, reward, terminated, truncated, _ = self.env.step(action)

                done = terminated or truncated

                total_rewards += reward
                state = new_state

            score += total_rewards

        if self.normalize_score:
            fitness = score / self.total_episodes
        else:
            fitness = score

        fitness_dict[individual_key] = fitness
        return fitness

    def terminate(self):
        self.env.close()
