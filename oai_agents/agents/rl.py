from oai_agents.agents.base_agent import SB3Wrapper, SB3LSTMWrapper, OAITrainer, PolicyClone, OAIAgent
from oai_agents.common.arguments import get_arguments
from oai_agents.common.networks import OAISinglePlayerFeatureExtractor
from oai_agents.common.state_encodings import ENCODING_SCHEMES
from oai_agents.common.tags import AgentPerformance, TeamType, TeammatesCollection, KeyCheckpoints
from oai_agents.agents.agent_utils import CustomAgent
from oai_agents.common.checked_model_name_handler import CheckedModelNameHandler

import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import RecurrentPPO, MaskablePPO
import wandb
import os
from typing import Optional

VEC_ENV_CLS = DummyVecEnv #

class RLAgentTrainer(OAITrainer):
    ''' Train an RL agent to play with a teammates_collection of agents.'''
    def __init__(
            self, teammates_collection, args,
            agent, epoch_timesteps, n_envs,
            seed, learner_type,
            train_types=[], eval_types=[],
            curriculum=None, num_layers=2, hidden_dim=256,
            checkpoint_rate=None, name=None, env=None, eval_envs=None,
            use_cnn=False, use_lstm=False, use_frame_stack=False,
            taper_layers=False, use_policy_clone=False, deterministic=False, start_step: int=0, start_timestep: int=0
        ):


        name = name or 'rl_agent'
        super(RLAgentTrainer, self).__init__(name, args, seed=seed)


        self.args = args
        self.device = args.device
        self.teammates_len = self.args.teammates_len
        self.num_players = self.args.num_players
        self.curriculum = curriculum

        self.epoch_timesteps = epoch_timesteps
        self.n_envs = n_envs

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.seed = seed
        self.checkpoint_rate = checkpoint_rate
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]

        self.use_lstm = use_lstm
        self.use_cnn = use_cnn
        self.taper_layers = taper_layers
        self.use_frame_stack = use_frame_stack
        self.use_policy_clone = use_policy_clone

        self.learner_type = learner_type
        self.env, self.eval_envs = self.get_envs(env, eval_envs, deterministic, learner_type, start_timestep)
        # Episode to start training from (usually 0 unless restarted)
        self.start_step = start_step
        self.steps = self.start_step
        # Cumm. timestep to start training from (usually 0 unless restarted)
        self.start_timestep = start_timestep
        self.learning_agent, self.agents = self.get_learning_agent(agent)
        self.teammates_collection, self.eval_teammates_collection = self.get_teammates_collection(
            _tms_clctn = teammates_collection,
            learning_agent = self.learning_agent,
            train_types = train_types,
            eval_types = eval_types
        )
        self.best_score, self.best_training_rew = -1, float('-inf')

    @classmethod
    def generate_randomly_initialized_agent(
            cls,
            args,
            learner_type:str,
            name:str,
            seed:int,
            hidden_dim:int,
            n_envs: int
        ) -> OAIAgent:
        '''
        Generate a randomly initialized learning agent using the RLAgentTrainer class
        This function does not perform any learning

        :param args: Parsed args object
        :param seed: Random seed
        :returns: An untrained, randomly inititalized RL agent
        '''
        trainer = cls(
            name=name,
            args=args,
            agent=None,
            teammates_collection={},
            epoch_timesteps=args.epoch_timesteps,
            n_envs=n_envs,
            seed=seed,
            hidden_dim=hidden_dim,
            learner_type=learner_type,
        )

        learning_agent, _ = trainer.get_learning_agent(None)
        return learning_agent

    def get_learning_agent(self, agent):
        if agent:
            learning_agent = agent
            learning_agent.agent.env = self.env
            learning_agent.agent.env.reset()
            agents = [learning_agent]
            return learning_agent, agents

        sb3_agent, agent_name = self.get_sb3_agent()
        learning_agent = self.wrap_agent(sb3_agent, agent_name)
        agents = [learning_agent]
        return learning_agent, agents


    def get_teammates_collection(self, _tms_clctn, learning_agent, train_types=[], eval_types=[]):
        '''
        Returns a dictionary of teammates_collection for training and evaluation
            dict
            teammates_collection = {
                'layout_name': {
                    'TeamType.HIGH_FIRST': [[agent1, agent2], ...],
                    'TeamType.MEDIUM_FIRST': [[agent3, agent4], ...],
                    'TeamType.LOW_FIRST': [[agent5, agent6], ..],
                    'TeamType.RANDOM': [[agent7, agent8], ...],
                },
            }
        '''
        if _tms_clctn == {}:
            _tms_clctn = {
                TeammatesCollection.TRAIN: {
                    layout_name:
                        {TeamType.SELF_PLAY: [[learning_agent for _ in range(self.teammates_len)]]}
                    for layout_name in self.args.layout_names
                },
                TeammatesCollection.EVAL: {
                    layout_name:
                        {TeamType.SELF_PLAY: [[learning_agent for _ in range(self.teammates_len)]]}
                    for layout_name in self.args.layout_names
                }
            }

        else:
            for layout in self.args.layout_names:
                for tt in _tms_clctn[TeammatesCollection.TRAIN][layout]:
                    if tt == TeamType.SELF_PLAY:
                        _tms_clctn[TeammatesCollection.TRAIN][layout][TeamType.SELF_PLAY] = [[learning_agent for _ in range(self.teammates_len)]]
                for tt in _tms_clctn[TeammatesCollection.EVAL][layout]:
                    if tt == TeamType.SELF_PLAY:
                        _tms_clctn[TeammatesCollection.EVAL][layout][TeamType.SELF_PLAY] = [[learning_agent for _ in range(self.teammates_len)]]

        train_teammates_collection = _tms_clctn[TeammatesCollection.TRAIN]
        eval_teammates_collection = _tms_clctn[TeammatesCollection.EVAL]

        if train_types:
            train_teammates_collection = {
                layout: {team_type: train_teammates_collection[layout][team_type] for team_type in train_types}
                for layout in train_teammates_collection
            }
        if eval_types:
            eval_teammates_collection = {
                layout: {team_type: eval_teammates_collection[layout][team_type] for team_type in eval_types}
                for layout in eval_teammates_collection
            }

        self.check_teammates_collection_structure(train_teammates_collection)
        self.check_teammates_collection_structure(eval_teammates_collection)
        return train_teammates_collection, eval_teammates_collection


    def print_tc_helper(self, teammates_collection, message=None):
        print("-------------------")
        if message:
            print(message)
        for layout_name in teammates_collection:
            for tag in teammates_collection[layout_name]:
                print(f'\t{tag}:')
                teammates_c = teammates_collection[layout_name][tag]
                for teammates in teammates_c:
                    for agent in teammates:
                        print(f'\t{agent.name}, score for layout {layout_name} is: {agent.layout_scores[layout_name]}, start_pos: {agent.get_start_position(layout_name, 0)}, len: {len(teammates)}')
        print("-------------------")


    def get_envs(self, _env, _eval_envs, deterministic, learner_type, start_timestep: int = 0):
        from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv
        if _env is None:
            env_kwargs = {'shape_rewards': True, 'full_init': False, 'stack_frames': self.use_frame_stack,
                        'deterministic': deterministic,'args': self.args, 'learner_type': learner_type, 'start_timestep': start_timestep}
            env = make_vec_env(OvercookedGymEnv, n_envs=self.args.n_envs, seed=self.seed, vec_env_cls=VEC_ENV_CLS, env_kwargs=env_kwargs)

            eval_envs_kwargs = {'is_eval_env': True, 'horizon': 400, 'stack_frames': self.use_frame_stack,
                                 'deterministic': deterministic, 'args': self.args, 'learner_type': learner_type}
            eval_envs = [OvercookedGymEnv(**{'env_index': i, **eval_envs_kwargs, 'unique_env_idx':self.args.n_envs+i}) for i in range(self.n_layouts)]
        else:
            env = _env
            eval_envs = _eval_envs

        for i in range(self.n_envs):
            env.env_method('set_env_layout', indices=i, env_index =i % self.n_layouts, unique_env_idx=i)
        return env, eval_envs


    def get_sb3_agent(self):
        layers = [self.hidden_dim // (2**i) for i in range(self.num_layers)] if self.taper_layers else [self.hidden_dim] * self.num_layers
        policy_kwargs = dict(net_arch=dict(pi=layers, vf=layers))

        if self.use_cnn:
            policy_kwargs.update(
                features_extractor_class=OAISinglePlayerFeatureExtractor,
                features_extractor_kwargs=dict(hidden_dim=self.hidden_dim))
        if self.use_lstm:
            policy_kwargs['n_lstm_layers'] = 2
            policy_kwargs['lstm_hidden_size'] = self.hidden_dim
            sb3_agent = RecurrentPPO('MultiInputLstmPolicy', self.env, seed=self.seed, policy_kwargs=policy_kwargs, verbose=1,
                                     n_steps=500, n_epochs=4, batch_size=500)
            agent_name = f'{self.name}_lstm'

        else:
            '''
            n_steps = n_steps is the number of experiences collected from a single environment
            number of updates = total_timesteps // (n_steps * n_envs)
            a batch for PPO is actually n_steps * n_envs BUT
            batch_size = minibatch size where you take some subset of your buffer (batch) with random shuffling.
            https://stackoverflow.com/a/76198343/9102696
            n_epochs = Number of epoch when optimizing the surrogate loss
            '''
            sb3_agent = PPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, seed=self.seed, verbose=self.args.sb_verbose, n_steps=500,
                            n_epochs=4, learning_rate=0.0003, batch_size=500, ent_coef=0.001, vf_coef=0.3,
                            gamma=0.99, gae_lambda=0.95, device=self.args.device)
            agent_name = f'{self.name}'
        return sb3_agent, agent_name


    def check_teammates_collection_structure(self, teammates_collection):
        '''
        teammates_collection = {
                'layout_name': {
                    'high_perf_first': [[agent1, agent2], ...],
                    'medium_perf_..':[[agent3, agent4], ...],
                    'low_...': [[agent5, agent6], ...],
                    'random': [[agent7, agent8], ...],
                },
            }
        '''
        for layout in teammates_collection:
            for team_type in teammates_collection[layout]:
                for teammates in teammates_collection[layout][team_type]:
                    assert len(teammates) == self.teammates_len,\
                            f"Teammates length in collection: {len(teammates)} must be equal to self.teammates_len: {self.teammates_len}"
                    for teammate in teammates:
                        assert type(teammate) in [SB3Wrapper, CustomAgent], f"All teammates must be of type SB3Wrapper, but got: {type(teammate)}"


    def _get_constructor_parameters(self):
        return dict(args=self.args, name=self.name, use_lstm=self.use_lstm,
                    use_frame_stack=self.use_frame_stack,
                    hidden_dim=self.hidden_dim, seed=self.seed)

    def wrap_agent(self, sb3_agent, name):
        if self.use_lstm:
            return SB3LSTMWrapper(sb3_agent, name, self.args)
        return SB3Wrapper(sb3_agent, name, self.args)


    def should_evaluate(self, steps):
        mean_training_rew = np.mean([ep_info["r"] for ep_info in self.learning_agent.agent.ep_info_buffer])
        self.best_training_rew *= 1
        steps_divisible_by_x = (steps + 1) % 40 == 0
        mean_rew_greater_than_best = mean_training_rew > self.best_training_rew and self.learning_agent.num_timesteps >= 5e6
        checkpoint_rate_reached = self.checkpoint_rate and (self.learning_agent.num_timesteps // self.checkpoint_rate) > (len(self.ck_list) - 1 + self.args.ck_list_offset)
        return steps_divisible_by_x or mean_rew_greater_than_best or checkpoint_rate_reached


    def log_details(self, experiment_name, total_train_timesteps):
        print("Training agent: " + self.name + ", for experiment: " + experiment_name)
        self.print_tc_helper(self.teammates_collection, "Train TC")
        self.print_tc_helper(self.eval_teammates_collection, "Eval TC")
        self.curriculum.print_curriculum()
        print("How Long: ", self.args.how_long)
        print(f"Epoch timesteps: {self.epoch_timesteps}")
        print(f"Total training timesteps: {total_train_timesteps}")
        print(f"Number of environments: {self.n_envs}")
        print(f"Hidden dimension: {self.hidden_dim}")
        print(f"Seed: {self.seed}")
        print(f"args.num_of_ckpoints: {self.args.num_of_ckpoints if self.checkpoint_rate else None}")
        print(f"args.checkpoint_rate: {self.checkpoint_rate}")
        print(f"Learner type: {self.learner_type}")
        for arg in vars(self.args):
            print(arg, getattr(self.args, arg))


    def save_init_model_and_cklist(self):
        self.ck_list = []
        path, tag = self.save_agents(tag=f'{KeyCheckpoints.CHECKED_MODEL_PREFIX}{0}')
        self.ck_list.append(({k: 0 for k in self.args.layout_names}, path, tag))

    def train_agents(self, total_train_timesteps, tag_for_returning_agent, resume_ck_list=None):
        experiment_name = RLAgentTrainer.get_experiment_name(exp_folder=self.args.exp_dir, model_name=self.name)
        run = wandb.init(project="overcooked_ai", entity=self.args.wandb_ent, dir=str(self.args.base_dir / 'wandb'),
                         reinit=True, name=experiment_name, mode=self.args.wandb_mode,
                         resume="allow")

        self.log_details(experiment_name, total_train_timesteps)

        if self.checkpoint_rate is not None:
            if self.args.resume:
                path = RLAgentTrainer.get_model_path(
                    base_dir=self.args.base_dir,
                    exp_folder=self.args.exp_dir,
                    model_name=self.name
                )
                if not path.exists():
                    print(f"Warning: The directory {path} does not exist.")
                    self.save_init_model_and_cklist()
                else:
                    ckpts = [name for name in os.listdir(path) if name.startswith(KeyCheckpoints.CHECKED_MODEL_PREFIX)]
                    if not ckpts:
                        print(f"Warning: No checkpoints found in {path} with prefix '{KeyCheckpoints.CHECKED_MODEL_PREFIX}'.")
                        self.save_init_model_and_cklist()
                    else:
                        ckpts_nums = [int(c.split('_')[1]) for c in ckpts]
                        sorted_idxs = np.argsort(ckpts_nums)
                        ckpts = [ckpts[i] for i in sorted_idxs]
                        self.ck_list = [(c[0], path, c[2]) for c in resume_ck_list] if resume_ck_list else [
                                ({k: 0 for k in self.args.layout_names}, path, ck) for ck in ckpts]
            else:
                self.save_init_model_and_cklist()


        best_path, best_tag = None, None

        self.steps = self.start_step
        self.learning_agent.num_timesteps = self.n_envs * self.start_timestep
        
        ck_name_handler = CheckedModelNameHandler()

        while self.learning_agent.num_timesteps < total_train_timesteps:
            self.curriculum.update(current_step=self.steps)
            self.set_new_teammates(curriculum=self.curriculum)

            # In each iteration the agent collects n_envs * n_steps experiences. This continues until self.learning_agent.num_timesteps > epoch_timesteps is reached.
            self.learning_agent.learn(self.epoch_timesteps)
            self.steps += 1

            if self.should_evaluate(steps=self.steps):
                mean_training_rew = np.mean([ep_info["r"] for ep_info in self.learning_agent.agent.ep_info_buffer])
                if mean_training_rew >= self.best_training_rew:
                    self.best_training_rew = mean_training_rew

                mean_reward, rew_per_layout, rew_per_layout_per_teamtype = self.evaluate(self.learning_agent, timestep=self.learning_agent.num_timesteps)

                if self.curriculum.prioritized_sampling:
                    # Use the results from the evaluation to dictate how teammates are sampled in the next round
                    self.curriculum.update_teamtype_performances(teamtype_performances=rew_per_layout_per_teamtype)

                if self.checkpoint_rate:
                    if self.learning_agent.num_timesteps // self.checkpoint_rate > (len(self.ck_list) - 1):
                        path = OAITrainer.get_model_path(
                            base_dir=self.args.base_dir,
                            exp_folder=self.args.exp_dir,
                            model_name=self.name
                        )
                        tag = ck_name_handler.generate_tag(id=len(self.ck_list), mean_reward=mean_reward)
                        self.ck_list.append((rew_per_layout, path, tag))
                        _, _ = self.save_agents(path=path, tag=tag)

                if mean_reward >= self.best_score:
                    best_path, best_tag = self.save_agents(tag=KeyCheckpoints.BEST_EVAL_REWARD)
                    print(f'New best evaluation score of {mean_reward} reached, model saved to {best_path}/{best_tag}')
                    self.best_score = mean_reward

        self.save_agents(tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL)
        self.agents, _, _ = RLAgentTrainer.load_agents(args=self.args, name=self.name, tag=tag_for_returning_agent)
        run.finish()

    @staticmethod
    def find_closest_score_path_tag(target_score, all_score_path_tag):
        closest_score = float('inf')
        closest_score_path_tag = None
        for score, path, tag in all_score_path_tag:
            if abs(score - target_score) < closest_score:
                closest_score = abs(score - target_score)
                closest_score_path_tag = (score, path, tag)
        return closest_score_path_tag

    @staticmethod
    def get_agents_and_set_score_and_perftag(args, layout_name, scores_path_tag, performance_tag, ck_list):
        score, path, tag = scores_path_tag
        all_agents, _, _ = RLAgentTrainer.load_agents(args, path=path, tag=tag)
        for agent in all_agents:
            agent.layout_scores[layout_name] = score
            agent.layout_performance_tags[layout_name] = performance_tag

        # set other layouts's scores. Can't set their performance tags because we don't know it but it doesn't matter, we don't use the perftag
        for agent in all_agents:
            for scores, ck_path, ck_tag in ck_list:
                if ck_path == path and ck_tag == tag:
                    for other_layout in args.layout_names:
                        if other_layout != layout_name:
                            agent.layout_scores[other_layout] = scores[other_layout]
        return all_agents

    @staticmethod
    def get_checkedpoints_agents(args, ck_list, layout_name):
        '''
        categorizes agents using performance tags based on the checkpoint list
            AgentPerformance.HIGH
            AgentPerformance.MEDIUM
            AgentPerformance.LOW
        It categorizes by setting their score and performance tag:
            OAIAgent.layout_scores
            OAIAgent.layout_performance_tags
        returns all_agents = [agent1, agent2, ...]
        '''
        if len(ck_list) < len(AgentPerformance.ALL):
            raise ValueError(f'Must have at least {len(AgentPerformance.ALL)} checkpoints saved. \
                             Currently is: {len(ck_list)}. Increase ck_rate or training length')

        all_score_path_tag_sorted = []
        for scores, path, tag in ck_list:
            all_score_path_tag_sorted.append((scores[layout_name], path, tag))
        all_score_path_tag_sorted.sort(key=lambda x: x[0], reverse=True)

        highest_score = all_score_path_tag_sorted[0][0]
        lowest_score = all_score_path_tag_sorted[-1][0]
        middle_score = (highest_score + lowest_score) // 2

        high_score_path_tag = all_score_path_tag_sorted[0]
        medium_score_path_tag = RLAgentTrainer.find_closest_score_path_tag(middle_score, all_score_path_tag_sorted)
        low_score_path_tag = all_score_path_tag_sorted[-1]

        H_agents = RLAgentTrainer.get_agents_and_set_score_and_perftag(args, layout_name, high_score_path_tag, AgentPerformance.HIGH, ck_list=ck_list)
        M_agents = RLAgentTrainer.get_agents_and_set_score_and_perftag(args, layout_name, medium_score_path_tag, AgentPerformance.MEDIUM, ck_list=ck_list)
        L_agents = RLAgentTrainer.get_agents_and_set_score_and_perftag(args, layout_name, low_score_path_tag, AgentPerformance.LOW, ck_list=ck_list)

        all_agents = H_agents + M_agents + L_agents
        return all_agents
