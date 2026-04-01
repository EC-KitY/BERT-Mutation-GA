import numpy as np
import torch
from torch.optim import Adam
from transformers import BertConfig, BertForMaskedLM


class BertMutation:
    def __init__(
        self,
        max_int_val,
        get_fitness_func,
        batch_size=64,
        learning_rate=1e-3,
        adam_decay=0,
        epsilon_greedy=0.01,
        word_embedding_dim=64,
        context_size=1024,
        n_layers=4,
        n_attention_heads=4,
        internal_size=64,
        clip_grad_norm=1.0,
        full_trajectory_query=False,
        higher_is_better=True,
        mask_probability=0.1,
        normalize_batches=False,
        entropy_coefficient=0.0,
        scale_fitness_function=None,
        fitness_dict=None,
        best_ind_scale_factor=2.0,
        best_ind_sample_size=10,
    ):
        assert 0 <= mask_probability <= 1, "Mask probability must be between 0 and 1"
        assert n_layers > 0, "Number of layers must be greater than 0"
        assert n_attention_heads > 0, "Number of attention heads must be greater than 0"
        assert internal_size > 0, "Internal size must be greater than 0"
        assert word_embedding_dim > 0, "Word embedding dimension must be greater than 0"
        assert context_size > 0, "Context size must be greater than 0"
        assert batch_size > 0, "Batch size must be greater than 0"
        assert learning_rate > 0, "Learning rate must be greater than 0"
        assert max_int_val >= 1, "Max integer value must be at least 1"

        self.vocab_size = max_int_val + 2
        self.max_int_val = max_int_val
        self.mask_id = max_int_val + 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        bert_config = BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=word_embedding_dim,
            num_hidden_layers=n_layers,
            num_attention_heads=n_attention_heads,
            intermediate_size=internal_size,
            max_position_embeddings=context_size,
            pad_token_id=self.mask_id,
        )
        self.model = BertForMaskedLM(bert_config).to(self.device)
        self.fitness_dict = fitness_dict
        self.batch_size = batch_size
        self.get_fitness_func = get_fitness_func
        self.epsilon_greedy = epsilon_greedy
        self.clip_grad_norm = clip_grad_norm
        self.full_trajectory_query = full_trajectory_query
        self.higher_is_better = higher_is_better
        self.mask_probability = mask_probability
        self.normalize_batches = normalize_batches
        self.entropy_coefficient = entropy_coefficient
        self.scale_fitness_function = scale_fitness_function
        self.best_ind_scale_factor = best_ind_scale_factor
        self.best_ind_sample_size = best_ind_sample_size
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=adam_decay)
        self.trajectory_log_probabilities = []
        self.rewards = []
        self.entropy_list = []

    def mutate(self, individual_to_mutate, mask):
        unmasked_tokens = np.copy(individual_to_mutate)
        masked_individual = np.copy(unmasked_tokens)
        masked_individual[mask] = self.mask_id

        tokens_ids = torch.tensor(np.array([masked_individual]), dtype=torch.long, device=self.device)
        logits = self.model(tokens_ids, attention_mask=torch.ones_like(tokens_ids, device=self.device)).logits
        mask_indices = torch.where(tokens_ids == self.mask_id)[1]

        suggested_mutation, trajectory_action_probabilities, dist_entropy = self.masked_trajectory_generation(
            logits, mask_indices
        )

        for genome_index, suggested_value in zip(np.where(mask)[0], suggested_mutation):
            unmasked_tokens[genome_index] = suggested_value

        reward = self.get_fitness_func(unmasked_tokens)
        self.log_trajectory_to_memory(dist_entropy, reward, trajectory_action_probabilities)

        if self.fitness_dict is not None and self.fitness_dict:
            self.peek_to_best_individual(logits)

        self.run_epoch()
        return unmasked_tokens

    def log_trajectory_to_memory(self, dist_entropy, reward, trajectory_action_probabilities):
        trajectory_probability = torch.log(trajectory_action_probabilities).sum().unsqueeze(0).unsqueeze(0)
        self.rewards.append(torch.full_like(trajectory_probability, reward))
        self.trajectory_log_probabilities.append(trajectory_probability)
        self.entropy_list.append(dist_entropy.unsqueeze(0).unsqueeze(0))

    def get_mutation(self, individuals_to_mutate):
        mutated_individuals = []
        individual_masks = self._sample_masks(individuals_to_mutate)
        for individual_to_mutate, mask in zip(individuals_to_mutate, individual_masks):
            mutated_individuals.append(self.mutate(individual_to_mutate, mask))
        return mutated_individuals

    def _sample_masks(self, individuals):
        n_individuals = len(individuals)
        individual_length = len(individuals[0])
        return np.random.choice(
            [True, False],
            size=(n_individuals, individual_length),
            p=[self.mask_probability, 1 - self.mask_probability],
        )

    def masked_trajectory_generation(self, logits, mask_indices):
        n_masks = len(mask_indices)
        operators_proba = torch.softmax(logits[0, mask_indices][:, :-1], dim=-1).to(self.device)
        sampled_distribution = torch.distributions.Categorical(operators_proba)
        sampled_operators_dist = sampled_distribution.sample().to(self.device)
        dist_entropy = sampled_distribution.entropy().mean()

        epsilon_greedy_probas = torch.rand(n_masks, device=self.device)
        epsilon_greedy_masks = torch.where(epsilon_greedy_probas < self.epsilon_greedy)[0]
        sampled_operators_dist[epsilon_greedy_masks] = torch.randint(
            0, self.vocab_size - 1, (epsilon_greedy_masks.shape[0],), device=self.device
        )
        trajectory_action_probabilities = torch.gather(
            operators_proba, dim=1, index=sampled_operators_dist.unsqueeze(-1)
        )
        suggested_mutation = sampled_operators_dist.detach().cpu().numpy()
        return suggested_mutation, trajectory_action_probabilities, dist_entropy

    def peek_to_best_individual(self, logits):
        top_k_individuals = sorted(
            self.fitness_dict.items(),
            key=lambda item: item[1],
            reverse=self.higher_is_better,
        )[: self.best_ind_sample_size]
        best_individual = top_k_individuals[np.random.randint(0, len(top_k_individuals))][0]
        best_individual_fitness = self.fitness_dict[tuple(best_individual)]

        best_individual = torch.tensor(best_individual, dtype=torch.long, device=self.device).unsqueeze(0)
        operators_proba = torch.softmax(logits[0][:, :-1], dim=-1).to(self.device)
        dist_entropy = torch.distributions.Categorical(operators_proba).entropy().mean()
        trajectory_action_probabilities = torch.gather(operators_proba, dim=1, index=best_individual.T)
        self.log_trajectory_to_memory(
            dist_entropy,
            best_individual_fitness * self.best_ind_scale_factor,
            trajectory_action_probabilities,
        )

    def run_epoch(self, numerical_stability=1e-10):
        if self.get_batch_size() < self.batch_size:
            return

        all_traj_proba = torch.cat(self.trajectory_log_probabilities, dim=0).to(self.device)
        all_rewards = torch.cat(self.rewards, dim=0).to(self.device)
        all_entropy = torch.cat(self.entropy_list, dim=0).to(self.device)

        if self.scale_fitness_function is not None:
            all_rewards = self.scale_fitness_function(all_rewards)

        self.trajectory_log_probabilities.clear()
        self.rewards.clear()
        self.entropy_list.clear()
        self.optimizer.zero_grad()

        if self.normalize_batches:
            advantages = (all_rewards - torch.mean(all_rewards)) / (torch.std(all_rewards) + numerical_stability)
        else:
            advantages = all_rewards

        loss = -torch.mean(all_traj_proba * advantages) - self.entropy_coefficient * torch.mean(all_entropy)
        loss.backward()
        loss_value = loss.item()

        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

        self.optimizer.step()
        print(f"loss: {loss_value}, reward: {torch.mean(all_rewards)}")

    def get_batch_size(self):
        return sum(len(reward) for reward in self.rewards)
