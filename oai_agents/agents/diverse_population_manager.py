import torch as th
import torch.nn.functional as F
import random, wandb
from typing import List
from oai_agents.agents.base_agent import OAITrainer
from oai_agents.agents.diversity_rl_trainer import DiversityRLAgentTrainer
from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.checked_model_name_handler import CheckedModelNameHandler
from oai_agents.common.curriculum import Curriculum
from oai_agents.common.learner import LearnerType
from oai_agents.common.multi_setup_trainer import generate_hdim_and_seed
from oai_agents.common.tags import TeamType, Prefix, KeyCheckpoints
from scripts.utils.common import generate_name

class DiversePopulationManager:
    def __init__(self, population_size, args):
        self.population_size = population_size
        self.args = args
        self.epoch_timesteps = args.epoch_timesteps  # Number of timesteps per training episode
        seeds, h_dims = generate_hdim_and_seed(
            for_evaluation=args.gen_pop_for_eval,
            total_ego_agents=population_size
        )

        self.population: List[RLAgentTrainer] = []
        for i in range(population_size):
            primary_train_types = [TeamType.SELF_PLAY]
            primary_eval_types = [TeamType.SELF_PLAY]
            curriculum = Curriculum(
                train_types=primary_train_types,
                eval_types=primary_eval_types,
                is_random=True)
            name = generate_name(
                args=args,
                prefix=Prefix.SELF_PLAY,
                seed=seeds[i],
                h_dim=h_dims[i],
                train_types=primary_train_types,
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
                train_types=primary_train_types,
                eval_types=primary_eval_types,
                start_step=0,
                start_timestep=0,
            )
            self.population.append(trainer)
        self.start_timestep = 0
        self.timesteps = self.start_timestep
        self.experiment_name = RLAgentTrainer.get_experiment_name(
            exp_folder=args.exp_dir, model_name="maximum_entropy_population")
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

    # def evaluate_population(self):
    #     evaluation_results = {}
    #     for layout in range(self.args.n_layouts):
    #         evaluation_results[layout] = {}
    #         for idx, trainer in enumerate(self.population):
    #             total_reward = 0.0
    #             if isinstance(trainer.eval_env, list) and len(trainer.eval_env) > layout:
    #                 env = trainer.eval_env[layout]
    #             else:
    #                 env = trainer.eval_env
    #             obs = env.reset()
    #             done = False
    #             while not done:
    #                 action = trainer.learning_agent.agent.predict(obs, deterministic=True)[0]
    #                 obs, reward, done, info = env.step(action)
    #                 total_reward += reward
    #             evaluation_results[layout][f"trainer_{idx}"] = total_reward
    #             print(f"Evaluation - Layout {layout}, Trainer {idx}: Reward = {total_reward}")
    #     return evaluation_results

    # def log_details(self, experiment_name, total_train_timesteps):
    #     print("Training a Maximum Entropy Population:")
    #     print("How Long: ", self.args.how_long)
    #     print(f"Epoch timesteps: {self.epoch_timesteps}")
    #     print(f"Total training timesteps: {total_train_timesteps}")
    #     print(f"Number of environments: {self.n_envs}")
    #     print(f"Hidden dimension: {self.hidden_dim}")
    #     print(f"Seed: {self.seed}")
    #     print(f"args.num_of_ckpoints: {self.args.num_of_ckpoints if self.checkpoint_rate else None}")
    #     print(f"args.checkpoint_rate: {self.checkpoint_rate}")
    #     print(f"Learner type: {self.learner_type}")
    #     print("Dynamic Reward: ", self.args.dynamic_reward)
    #     print("Final sparse reward ratio: ", self.args.final_sparse_r_ratio)
    #     print('args.custom_agent_ck_rate_generation: ', self.args.custom_agent_ck_rate_generation)
    #     print('args.num_steps_in_traj_for_dyn_adv: ', self.args.num_steps_in_traj_for_dyn_adv)
    #     print('args.num_static_advs_per_heatmap: ', self.args.num_static_advs_per_heatmap)
    #     print('args.num_dynamic_advs_per_heatmap: ', self.args.num_dynamic_advs_per_heatmap)
    #     print('args.use_val_func_for_heatmap_gen: ', self.args.use_val_func_for_heatmap_gen)

    def train_population(self, total_timesteps=None, num_of_ckpoints=None, eval_interval=None):
        # Ensure that wandb is initialized at the start of training.
        run = wandb.init(project=self.project_name, config=self.args)
        if total_timesteps is None:
            total_timesteps = self.args.pop_total_training_timesteps
        if num_of_ckpoints is None:
            num_of_ckpoints = self.args.num_of_ckpoints
        if eval_interval is None:
            eval_interval = 40 * self.epoch_timesteps
        checkpoint_interval = total_timesteps // self.args.num_of_ckpoints

        experiment_name = RLAgentTrainer.get_experiment_name(exp_folder=self.args.exp_dir, model_name="maximum_entropy_population")
        run = wandb.init(
            project="overcooked_ai",
            entity=self.args.wandb_ent,
            dir=str(self.args.base_dir / 'wandb'),
            reinit=True,
            name=experiment_name,
            mode=self.args.wandb_mode,
            resume="allow"
        )
        self.timesteps = self.start_timestep
        next_eval = self.eval_interval
        next_checkpoint = checkpoint_interval

        # All RLAgentTrainer(s) save their first checkpoints.
        if self.timesteps == 0:
            for t in self.population:
                t.save_init_model_and_cklist()
        ck_name_handler = CheckedModelNameHandler()

        while self.timesteps < total_timesteps:
            trainer = random.choice(self.population)
            trainer.set_new_teammates(curriculum=trainer.curriculum)
            trainer.learning_agent.learn(self.epoch_timesteps)
            self.timesteps += self.epoch_timesteps

            print(f"Trained one trainer for {self.epoch_timesteps} timesteps. Total steps: {self.timesteps}")
            wandb.log({"train/episode_steps": self.epoch_timesteps, "train/timesteps": self.timesteps}, step=self.timesteps)

            for t in self.population:
                bonus_getter = self.bonus_getter_factory(t)
                t.env.env_method("set_bonus_getter", bonus_getter)

            if self.timesteps >= next_eval or self.timesteps >= next_checkpoint:
                for t in self.population:
                    mean_reward, rew_per_layout, rew_per_layout_per_teamtype = t.evaluate(
                        eval_agent=t.learning_agent,
                        timestep=self.timesteps,
                        log_wandb=False,)
                    if self.timesteps >= next_checkpoint:
                        path = OAITrainer.get_model_path(
                            base_dir=t.args.base_dir,
                            exp_folder=t.args.exp_dir,
                            model_name=t.name
                        )
                        tag = ck_name_handler.generate_tag(
                            id=len(t.ck_list), mean_reward=mean_reward)
                        t.ck_list.append((rew_per_layout, path, tag))
                        t.save_agents(path=path, tag=tag)
                if self.timesteps >= next_eval:
                    next_eval += self.eval_interval
                if self.timesteps >= next_checkpoint:
                    next_checkpoint += checkpoint_interval
        for t in self.population:
            t.save_agents(tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL)
        run.finish()


if __name__ == "__main__":
    from oai_agents.common.arguments import get_arguments
    args = get_arguments()
    population_size = getattr(args, "population_size", 4)
    total_timesteps = getattr(args, "total_timesteps", int(1e7))

    manager = DiversePopulationManager(population_size=population_size, args=args)
    manager.train_population(total_timesteps=total_timesteps)
