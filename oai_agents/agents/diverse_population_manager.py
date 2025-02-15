import torch as th
import torch.nn.functional as F
import random, wandb
from typing import List
from oai_agents.agents.base_agent import OAITrainer, OAIAgent
from oai_agents.agents.diversity_rl_trainer import DiversityRLAgentTrainer
from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.checked_model_name_handler import CheckedModelNameHandler
from oai_agents.common.curriculum import Curriculum
from oai_agents.common.learner import LearnerType
from oai_agents.common.multi_setup_trainer import generate_hdim_and_seed
from oai_agents.common.tags import TeamType, Prefix, KeyCheckpoints
from scripts.utils.common import generate_name
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.distributions import CategoricalDistribution

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
                checkpoint_rate=None,
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

    def get_other_agents(self, ego_trainer: RLAgentTrainer) -> List[OAIAgent]:
        """Return a list of agents for all trainers except the current one."""
        return [
            trainer.learning_agent
            for trainer in self.population
            if trainer != ego_trainer
        ]

    def compute_entropy_bonus(self, obs, action, ego_agent: OAIAgent, other_agents: List[OAIAgent]):
        with th.no_grad():
            alpha = 0.01
            probs_on_action = []
            ego_logits = ego_agent.get_distribution(obs).distribution.logits
            ego_probs = th.softmax(ego_logits, dim=1)
            probs_on_action.append(ego_probs[0][action])
            for learning_agent in other_agents:
                logits = learning_agent.get_distribution(obs).distribution.logits  # CategoricalDistribution object
                probs = th.softmax(logits, dim=1)
                probs_on_action.append(probs[0][action])
            avg_prob = th.stack(probs_on_action).mean()
            log_avg_prob = th.log(avg_prob)
            bonus = -alpha*log_avg_prob.item()
            # print(f"probs_on_action: {probs_on_action}")
            # print(f"avg_prob: {avg_prob}")
            # print(f"bonus: {bonus}")
        return bonus

    def bonus_getter_factory(self, ego_trainer:RLAgentTrainer):
        return lambda obs, action: self.compute_entropy_bonus(
            obs=obs,
            action=action,
            ego_agent=ego_trainer.learning_agent,
            other_agents=self.get_other_agents(ego_trainer=ego_trainer)
        )

    def train_population(self, total_timesteps=None, num_of_ckpoints=None, eval_interval=None):
        # Ensure that wandb is initialized at the start of training.
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
        next_eval = eval_interval
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

            print(f"Trained trainer of seed {trainer.seed} for {self.epoch_timesteps} timesteps. current timesteps: {self.timesteps}")
            wandb.log({"train/episode_steps": int(self.epoch_timesteps), "train/timesteps": int(self.timesteps)}, step=int(self.timesteps))

            for t in self.population:
                bonus_getter = self.bonus_getter_factory(ego_trainer=t)
                t.env.env_method("set_bonus_getter", bonus_getter)

            if self.timesteps >= next_eval or self.timesteps >= next_checkpoint:
                for t in self.population:
                    mean_reward, rew_per_layout, rew_per_layout_per_teamtype = t.evaluate(
                        eval_agent=t.learning_agent,
                        timestep=self.timesteps,
                        log_wandb=False,
                    )
                    if mean_reward >= t.best_score:
                        best_path, best_tag = t.save_agents(tag=KeyCheckpoints.BEST_EVAL_REWARD)
                        t.best_score = mean_reward
                        print(f'Model of Seed {t.seed} Update: \nBest evaluation score of {mean_reward} reached, model saved to {best_path}/{best_tag}')
                    wandb.log({
                        f'eval/{t.name}/eval_mean_reward': mean_reward,
                        f'eval/{t.name}/timestep': int(self.timesteps)
                    })
                    for _, env in enumerate(t.eval_envs):
                        wandb.log({
                            f'eval/{t.name}/eval_mean_reward_{env.layout_name}': rew_per_layout[env.layout_name],
                            f'eval/{t.name}/timestep': int(self.timesteps),
                        })
                        for teamtype in rew_per_layout_per_teamtype[env.layout_name]:
                            wandb.log({
                                f'eval/{t.name}/eval_mean_reward_{env.layout_name}_teamtype_{teamtype}': rew_per_layout_per_teamtype[env.layout_name][teamtype],
                                f'eval/{t.name}/timestep': int(self.timesteps),
                            })

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
                    next_eval += eval_interval
                if self.timesteps >= next_checkpoint:
                    next_checkpoint += checkpoint_interval
        for t in self.population:
            t.save_agents(tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL)
        run.finish()


if __name__ == "__main__":
    from oai_agents.common.arguments import get_arguments
    from scripts.train_agents import set_input
    args = get_arguments()
    args.quick_test = False
    args.pop_force_training = False
    args.adversary_force_training = False
    args.primary_force_training = False
    args.teammates_len = 4

    if args.teammates_len == 1:
        args.how_long = 20
        args.num_of_ckpoints = 35
    elif args.teammates_len == 2:
        args.how_long = 25
        args.num_of_ckpoints = 40
    elif args.teammates_len == 4:
        args.how_long = 35
        args.num_of_ckpoints = 50

    set_input(args=args)

    args.total_ego_agents = 4

    manager = DiversePopulationManager(population_size=args.total_ego_agents, args=args)
    manager.train_population()
