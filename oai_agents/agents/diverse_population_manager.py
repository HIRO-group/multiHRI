import torch as th
import torch.nn.functional as F
from oai_agents.agents.diversity_rl_trainer import DiversityRLAgentTrainer
from scripts.utils.common import generate_name
from oai_agents.common.tags import Prefix
from oai_agents.common.multi_setup_trainer import generate_hdim_and_seed
from oai_agents.common.tags import TeamType
from oai_agents.common.curriculum import Curriculum
from oai_agents.common.learner import LearnerType

class DiversePopulationManager:
    def __init__(self, population_size, args, learner_type=LearnerType.ORIGINALER):
        self.population_size = population_size
        self.args = args
        self.population = []
        self.checkpoint_rate = args.pop_total_training_timesteps // args.num_of_ckpoints
        seeds, h_dims = generate_hdim_and_seed(
            for_evaluation=args.gen_pop_for_eval,
            total_ego_agents=population_size
        )

        primary_train_types = [TeamType.SELF_PLAY]
        primary_eval_types = {
            'generate': [TeamType.SELF_PLAY],
            'load': []
        }
        for i in range(population_size):
            curriculum = Curriculum(train_types=primary_train_types, is_random=True)
            name = generate_name(
                args=args,
                prefix=Prefix.SELF_PLAY,
                seed=seeds[i],
                h_dim=h_dims[i],
                train_types=curriculum.train_types,
                curriculum=curriculum
            )
            # Instantiate a DiversityRLAgentTrainer for each member of the population.
            trainer = DiversityRLAgentTrainer(
                args=args,
                name=name,
                teammates_collection={},
                curriculum=curriculum,
                hidden_dim=h_dims[i],
                seed=seeds[i],
                checkpoint_rate=self.checkpoint_rate,
                learner_type=learner_type,
                agent=,
                epoch_timesteps=self.args.epoch_timesteps,
                n_envs=,
                start_step=,
                start_timestep=,

                teammates_collection={},  # Adjust teammate sampling if needed.
                args=args,
                agent=None,              # RLAgentTrainer creates its own SB3 agent if agent is None.
                epoch_timesteps=args.epoch_timesteps,
                n_envs=args.n_envs,
                seed=args.seed + i,
                learner_type="your_learner_type"  # Replace with the appropriate learner type.
            )
            self.population.append(trainer)

    def get_other_policies(self, current_trainer):
        """
        Returns a list of policy networks for all trainers except the current one.
        """
        return [
            trainer.learning_agent.agent.policy
            for trainer in self.population
            if trainer != current_trainer
        ]

    def compute_entory_bonus(self, current_state, current_action, current_policy, other_policies, device="cpu"):
        """
        Computes a diversity bonus ("entory") for the current trainer.
        Here, we calculate the average KL divergence between the current policy’s action distribution
        and those of the other trainers at the given state.
        """
        # Ensure that current_state is a tensor.
        if not isinstance(current_state, th.Tensor):
            current_state_tensor = th.tensor(current_state, dtype=th.float32, device=device)
        else:
            current_state_tensor = current_state

        with th.no_grad():
            logits_current = current_policy(current_state_tensor)  # shape: [batch, num_actions]
            probs_current = th.softmax(logits_current, dim=-1)
            kl_divs = []
            for policy in other_policies:
                logits_other = policy(current_state_tensor)
                probs_other = th.softmax(logits_other, dim=-1)
                kl_div = F.kl_div(probs_other.log(), probs_current, reduction='batchmean')
                kl_divs.append(kl_div.item())
            bonus = sum(kl_divs) / len(kl_divs) if kl_divs else 0.0
        return bonus

    def bonus_getter_factory(self, current_trainer):
        """
        Returns a bonus_getter function for the current trainer.
        This function uses the current trainer's policy and compares it with the policies of the other trainers.
        """
        return lambda env_state, action: self.compute_entory_bonus(
            current_state=env_state,
            current_action=action,
            current_policy=current_trainer.learning_agent.agent.policy,
            other_policies=self.get_other_policies(current_trainer),
            device=self.args.device
        )

    def train_population(self, total_timesteps, timesteps_per_update):
        """
        The main training loop for the population.
        In each iteration, the bonus_getter for each trainer's training environment is updated,
        and then each trainer trains for a fixed number of timesteps.
        """
        while not self._reached_total_timesteps(total_timesteps):
            for trainer in self.population:
                # Update the bonus_getter in the trainer's training environment.
                bonus_getter = self.bonus_getter_factory(trainer)
                trainer.env.env_method("set_bonus_getter", bonus_getter)
                # Train the trainer for a fixed number of timesteps.
                trainer.train_agents(timesteps_per_update, tag_for_returning_agent="pop_update")
            print("Population training update complete.")

    def _reached_total_timesteps(self, total_timesteps):
        total = sum(trainer.learning_agent.num_timesteps for trainer in self.population)
        return total >= total_timesteps


# -------------------- Test Script for DiversePopulationManager --------------------
if __name__ == "__main__":
    from oai_agents.common.arguments import get_arguments
    args = get_arguments()  # Ensure your arguments include n_envs, epoch_timesteps, seed, etc.

    # Optionally, set default values here if not provided by your arguments.
    population_size = getattr(args, "population_size", 4)
    total_timesteps = getattr(args, "total_timesteps", int(1e7))
    timesteps_per_update = getattr(args, "timesteps_per_update", 100000)

    manager = DiversePopulationManager(population_size=population_size, args=args)
    manager.train_population(total_timesteps=total_timesteps, timesteps_per_update=timesteps_per_update)
