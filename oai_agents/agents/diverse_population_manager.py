import torch as th
import torch.nn.functional as F
import random
from oai_agents.agents.diversity_rl_trainer import DiversityRLAgentTrainer
from scripts.utils.common import generate_name
from oai_agents.common.tags import Prefix
from oai_agents.common.multi_setup_trainer import generate_hdim_and_seed
from oai_agents.common.tags import TeamType
from oai_agents.common.curriculum import Curriculum
from oai_agents.common.learner import LearnerType
# trainer = DiversityRLAgentTrainer(
#                 args=args,
#                 name=name,
#                 teammates_collection={},
#                 curriculum=curriculum,
#                 hidden_dim=h_dims[i],
#                 seed=seeds[i],
#                 checkpoint_rate=self.checkpoint_rate,
#                 learner_type=learner_type,
#                 agent=,
#                 epoch_timesteps=self.args.epoch_timesteps,
#                 n_envs=,
#                 start_step=,
#                 start_timestep=,

#                 teammates_collection={},  # Adjust teammate sampling if needed.
#                 args=args,
#                 agent=None,              # RLAgentTrainer creates its own SB3 agent if agent is None.
#                 epoch_timesteps=args.epoch_timesteps,
#                 n_envs=args.n_envs,
#                 seed=args.seed + i,
#                 learner_type="your_learner_type"  # Replace with the appropriate learner type.
#             )
import random
import torch as th
import torch.nn.functional as F
import wandb
from oai_agents.agents.diversity_rl_trainer import DiversityRLAgentTrainer

class DiversePopulationManager:
    def __init__(self, population_size, args):
        self.population_size = population_size
        self.args = args
        self.epoch_timesteps = args.epoch_timesteps  # Number of timesteps per training episode
        seeds, h_dims = generate_hdim_and_seed(
            for_evaluation=args.gen_pop_for_eval,
            total_ego_agents=population_size
        )

        self.population = []
        for i in range(population_size):
            primary_train_types = [TeamType.SELF_PLAY]
            primary_eval_types = {
                'generate': [TeamType.SELF_PLAY],
                'load': []
            }
            curriculum = Curriculum(train_types=primary_train_types, is_random=True)
            name = generate_name(
                args=args,
                prefix=Prefix.SELF_PLAY,
                seed=seeds[i],
                h_dim=h_dims[i],
                train_types=curriculum.train_types,
                curriculum=curriculum
            )
            trainer = DiversityRLAgentTrainer(
                args=args,
                name=name,
                teammates_collection={},
                curriculum=curriculum,
                hidden_dim=h_dims[i],
                seed=seeds[i],
                checkpoint_rate=self.checkpoint_rate,
                learner_type=LearnerType.ORIGINALER,
                agent=None,
                epoch_timesteps=args.epoch_timesteps,
                n_envs=args.n_envs,
                start_step=0,
                start_timestep=0,
            )
            self.population.append(trainer)
        self.timesteps = 0
        self.project_name = 'maximum_entropy_population'
        # # Evaluation and checkpoint intervals
        # self.eval_interval = getattr(args, "eval_interval", 50000)
        # self.checkpoint_interval = getattr(args, "checkpoint_interval", 100000)

    def get_other_policies(self, current_trainer):
        """Return a list of policy networks for all trainers except the current one."""
        return [
            trainer.learning_agent.agent.policy
            for trainer in self.population
            if trainer != current_trainer
        ]

    def compute_entropy_bonus(self, current_state, current_action, current_policy, other_policies, device="cpu"):
        if not isinstance(current_state, th.Tensor):
            current_state_tensor = th.tensor(current_state, dtype=th.float32, device=device)
        else:
            current_state_tensor = current_state

        with th.no_grad():
            logits_current = current_policy(current_state_tensor)
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
        return lambda env_state, action: self.compute_entropy_bonus(
            current_state=env_state,
            current_action=action,
            current_policy=current_trainer.learning_agent.agent.policy,
            other_policies=self.get_other_policies(current_trainer),
            device=self.args.device
        )

    def evaluate_population(self):
        evaluation_results = {}
        for layout in range(self.args.n_layouts):
            evaluation_results[layout] = {}
            for idx, trainer in enumerate(self.population):
                total_reward = 0.0
                if isinstance(trainer.eval_env, list) and len(trainer.eval_env) > layout:
                    env = trainer.eval_env[layout]
                else:
                    env = trainer.eval_env
                obs = env.reset()
                done = False
                while not done:
                    action = trainer.learning_agent.agent.predict(obs, deterministic=True)[0]
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                evaluation_results[layout][f"trainer_{idx}"] = total_reward
                print(f"Evaluation - Layout {layout}, Trainer {idx}: Reward = {total_reward}")
        return evaluation_results

    def train_population(self, total_timesteps=None, num_of_ckpoints=None):
        # Ensure that wandb is initialized at the start of training.
        if wandb.run is None:
            wandb.init(project=self.project_name, config=self.args)
        if total_timesteps is None:
            total_timesteps = self.args.pop_total_training_timesteps
        if num_of_ckpoints is None:
            num_of_ckpoints = self.args.num_of_ckpoints
        checkpoint_interval = total_timesteps // self.args.num_of_ckpoints

        next_eval = self.eval_interval
        next_checkpoint = checkpoint_interval

        while self.timesteps < total_timesteps:
            trainer = random.choice(self.population)
            trainer.learning_agent.learn(self.epoch_timesteps)
            self.timesteps += self.epoch_timesteps
            print(f"Trained one trainer for {self.epoch_timesteps} timesteps. Total steps: {self.timesteps}")
            wandb.log({"train/episode_steps": self.epoch_timesteps, "train/timesteps": self.timesteps}, step=self.timesteps)

            for t in self.population:
                bonus_getter = self.bonus_getter_factory(t)
                t.env.env_method("set_bonus_getter", bonus_getter)

            if self.timesteps >= next_eval:
                eval_results = self.evaluate_population()
                for layout, trainer_rewards in eval_results.items():
                    log_dict = {}
                    for trainer_key, reward in trainer_rewards.items():
                        log_dict[f"eval/layout_{layout}/{trainer_key}"] = reward
                    wandb.log(log_dict, step=self.timesteps)
                print("Evaluation results:", eval_results)
                next_eval += self.eval_interval

            if self.timesteps >= next_checkpoint:
                for t in self.population:
                    t.save_model()
                wandb.log({"checkpoint": self.timesteps}, step=self.timesteps)
                print(f"Checkpoint reached at {self.timesteps} timesteps, models saved.")
                next_checkpoint += checkpoint_interval

if __name__ == "__main__":
    from oai_agents.common.arguments import get_arguments
    args = get_arguments()
    population_size = getattr(args, "population_size", 4)
    total_timesteps = getattr(args, "total_timesteps", int(1e7))

    manager = DiversePopulationManager(population_size=population_size, args=args)
    manager.train_population(total_timesteps=total_timesteps)
