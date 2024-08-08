from oai_agents.agents.base_agent import SB3Wrapper, SB3LSTMWrapper, OAITrainer, PolicyClone, OAIAgent
from oai_agents.common.arguments import get_arguments
from oai_agents.common.networks import OAISinglePlayerFeatureExtractor
from oai_agents.common.state_encodings import ENCODING_SCHEMES
from oai_agents.common.tags import AgentPerformance, TeamType, TeammatesCollection
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv

import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import RecurrentPPO, MaskablePPO
import wandb

VEC_ENV_CLS = DummyVecEnv #

class RLAgentTrainer(OAITrainer):
    ''' Train an RL agent to play with a teammates_collection of agents.'''
    def __init__(self, teammates_collection, args, 
                agent, epoch_timesteps, n_envs,
                seed, train_types=[], eval_types=[],
                curriculum=None, num_layers=2, hidden_dim=256, 
                fcp_ck_rate=None, name=None, env=None, eval_envs=None,
                use_cnn=False, use_lstm=False, use_frame_stack=False,
                taper_layers=False, use_policy_clone=False, deterministic=False):
        
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
        self.fcp_ck_rate = fcp_ck_rate
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]

        self.use_lstm = use_lstm
        self.use_cnn = use_cnn
        self.taper_layers = taper_layers
        self.use_frame_stack = use_frame_stack
        self.use_policy_clone = use_policy_clone

        self.env, self.eval_envs = self.get_envs(env, eval_envs, deterministic)
        
        self.learning_agent, self.agents = self.get_learning_agent(agent)
        self.teammates_collection, self.eval_teammates_collection = self.get_teammates_collection(_tms_clctn = teammates_collection,
                                                                                                   learning_agent = self.learning_agent,
                                                                                                   train_types = train_types,
                                                                                                   eval_types = eval_types)
        self.best_score, self.best_training_rew = -1, float('-inf')

    @classmethod
    def generate_randomly_initialized_SP_agent(cls,
                                               args,
                                               seed:int=8080) -> OAIAgent:
        '''
        Generate a randomly initialized learning agent using the RLAgentTrainer class
        This function does not perform any learning

        :param args: Parsed args object
        :param seed: Random seed
        :returns: An untrained, randomly inititalized RL agent
        '''

        name = 'randomized_agent'

        sp_trainer = cls(name=name,
                        args=args,
                        agent=None,
                        teammates_collection={},
                        epoch_timesteps=args.epoch_timesteps,
                        n_envs=args.n_envs,
                        seed=seed)

        return sp_trainer.get_agents()[0]

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
            print("No teammates collection provided, using SELF_PLAY: teammates will be the agent itself.")
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
        if message:
            print(message)
        for layout_name in teammates_collection:
            for tag in teammates_collection[layout_name]:
                print(f'\t{tag}:')
                teammates_c = teammates_collection[layout_name][tag]
                for teammates in teammates_c:
                    for agent in teammates:
                        print(f'\t{agent.name}, score for layout {layout_name} is: {agent.layout_scores[layout_name]}, len: {len(teammates)}')


    def get_envs(self, _env, _eval_envs, deterministic):
        if _env is None:
            env_kwargs = {'shape_rewards': True, 'full_init': False, 'stack_frames': self.use_frame_stack,
                        'deterministic': deterministic,'args': self.args}
            env = make_vec_env(OvercookedGymEnv, n_envs=self.args.n_envs, seed=self.seed,
                                    vec_env_cls=VEC_ENV_CLS, env_kwargs=env_kwargs)
            eval_envs_kwargs = {'is_eval_env': True, 'horizon': 400, 'stack_frames': self.use_frame_stack,
                                 'deterministic': deterministic, 'args': self.args}
            eval_envs = [OvercookedGymEnv(**{'env_index': i, **eval_envs_kwargs}) for i in range(self.n_layouts)]
        else:
            env = _env
            eval_envs = _eval_envs

        for i in range(self.n_envs):
            env.env_method('set_env_layout', indices=i, env_index=i % self.n_layouts)
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
            sb3_agent = RecurrentPPO('MultiInputLstmPolicy', self.env, policy_kwargs=policy_kwargs, verbose=1,
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
            sb3_agent = PPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=self.args.sb_verbose, n_steps=500,
                            n_epochs=4, learning_rate=0.0003, batch_size=500, ent_coef=0.001, vf_coef=0.3,
                            gamma=0.99, gae_lambda=0.95)
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
                        assert type(teammate) == SB3Wrapper, f"All teammates must be of type SB3Wrapper, but got: {type(teammate)}"


    def _get_constructor_parameters(self):
        return dict(args=self.args, name=self.name, use_lstm=self.use_lstm, 
                    use_frame_stack=self.use_frame_stack,
                    hidden_dim=self.hidden_dim, seed=self.seed)

    def wrap_agent(self, sb3_agent, name):
        if self.use_lstm:
            return SB3LSTMWrapper(sb3_agent, name, self.args)
        return SB3Wrapper(sb3_agent, name, self.args)

    def get_experiment_name(self, exp_name):
        all_train_teamtypes = [tag for tags_dict in self.teammates_collection.values() for tag in tags_dict.keys()]
        return exp_name or 'train_' + '_'.join(all_train_teamtypes)


    def should_evaluate(self, steps):
        mean_training_rew = np.mean([ep_info["r"] for ep_info in self.learning_agent.agent.ep_info_buffer])
        self.best_training_rew *= 0.98

        steps_divisable_by_5 = (steps + 1) % 5 == 0
        mean_rew_greater_than_best = mean_training_rew > self.best_training_rew and self.learning_agent.num_timesteps >= 5e6
        fcp_ck_rate_reached = self.fcp_ck_rate and self.learning_agent.num_timesteps // self.fcp_ck_rate > (len(self.ck_list) - 1)
    
        return steps_divisable_by_5 or mean_rew_greater_than_best or fcp_ck_rate_reached
    

    def train_agents(self, total_train_timesteps, exp_name=None):       
        exp_name = self.get_experiment_name(exp_name)
        run = wandb.init(project="overcooked_ai", entity=self.args.wandb_ent, dir=str(self.args.base_dir / 'wandb'),
                         reinit=True, name= exp_name + '_' + self.name, mode=self.args.wandb_mode,
                         resume="allow")
        
        print("Training agent: " + self.name + ", for experiment: "+exp_name)

        self.print_tc_helper(self.teammates_collection, "Train TC")
        self.print_tc_helper(self.eval_teammates_collection, "Eval TC")

        if self.fcp_ck_rate is not None:
            self.ck_list = []
            path, tag = self.save_agents(tag=f'ck_{len(self.ck_list)}')
            self.ck_list.append(({k: 0 for k in self.args.layout_names}, path, tag))

        best_path, best_tag = None, None
        
        steps = 0
        curr_timesteps = 0
        prev_timesteps = self.learning_agent.num_timesteps

        while curr_timesteps < total_train_timesteps:
            self.curriculum.update(current_step=steps)
            self.set_new_teammates(curriculum=self.curriculum)

            # In each iteration the agent collects n_envs * n_steps experiences
            # This continues until self.learning_agent.num_timesteps > epoch_timesteps is reached.
            self.learning_agent.learn(self.epoch_timesteps)
            
            
            curr_timesteps += self.learning_agent.num_timesteps - prev_timesteps
            prev_timesteps = self.learning_agent.num_timesteps

            if self.should_evaluate(steps=steps):
                mean_training_rew = np.mean([ep_info["r"] for ep_info in self.learning_agent.agent.ep_info_buffer])                
                if mean_training_rew >= self.best_training_rew:
                    self.best_training_rew = mean_training_rew
                mean_reward, rew_per_layout = self.evaluate(self.learning_agent, timestep=self.learning_agent.num_timesteps)

                if self.fcp_ck_rate:
                    if self.learning_agent.num_timesteps // self.fcp_ck_rate > (len(self.ck_list) - 1):
                        path, tag = self.save_agents(tag=f'ck_{len(self.ck_list)}_rew_{mean_reward}')
                        self.ck_list.append((rew_per_layout, path, tag))

                if mean_reward >= self.best_score:
                    best_path, best_tag = self.save_agents(tag='best')
                    print(f'New best score of {mean_reward} reached, model saved to {best_path}/{best_tag}')
                    self.best_score = mean_reward

            steps += 1

        self.save_agents()
        self.agents = RLAgentTrainer.load_agents(self.args, self.name, best_path, best_tag)
        run.finish()


    def find_closest_score_path_tag(self, target_score, all_score_path_tag):
        closest_score = float('inf')
        closest_score_path_tag = None
        for score, path, tag in all_score_path_tag:
            if abs(score - target_score) < closest_score:
                closest_score = abs(score - target_score)
                closest_score_path_tag = (score, path, tag)
        return closest_score_path_tag
    
    def get_agents_and_set_score_and_perftag(self, layout_name, scores_path_tag, performance_tag):
        score, path, tag = scores_path_tag
        all_agents = RLAgentTrainer.load_agents(self.args, path=path, tag=tag)
        for agent in all_agents:
            agent.layout_scores[layout_name] = score
            agent.layout_performance_tags[layout_name] = performance_tag        
        return all_agents


    def get_fcp_agents(self, layout_name):
        '''
        categorizes agents using performance tags based on the checkpoint list
            AgentPerformance.HIGH
            AgentPerformance.HIGH_MEDIUM
            AgentPerformance.MEDIUM
            AgentPerformance.MEDIUM_LOW
            AgentPerformance.LOW    
        It categorizes by setting their score and performance tag:
            OAIAgent.layout_scores
            OAIAgent.layout_performance_tags
        returns all_agents = [agent1, agent2, ...]
        '''
        if len(self.ck_list) < len(AgentPerformance.ALL):
            raise ValueError(f'Must have at least {len(AgentPerformance.ALL)} checkpoints saved. \
                             Currently is: {len(self.ck_list)}. Increase fcp_ck_rate or training length')

        all_score_path_tag_sorted = []
        for scores, path, tag in self.ck_list:
            all_score_path_tag_sorted.append((scores[layout_name], path, tag))
        all_score_path_tag_sorted.sort(key=lambda x: x[0], reverse=True)

        highest_score = all_score_path_tag_sorted[0][0]
        lowest_score = all_score_path_tag_sorted[-1][0]
        middle_score = (highest_score + lowest_score) // 2
        high_middle_score = (highest_score + middle_score) //2
        middle_low_score = (middle_score + lowest_score) // 2
        
        high_score_path_tag = all_score_path_tag_sorted[0]
        high_score_medium_path_tag = self.find_closest_score_path_tag(high_middle_score, all_score_path_tag_sorted)
        medium_score_path_tag = self.find_closest_score_path_tag(middle_score, all_score_path_tag_sorted)
        medium_score_low_path_tag = self.find_closest_score_path_tag(middle_low_score, all_score_path_tag_sorted)
        low_score_path_tag = all_score_path_tag_sorted[-1]

        H_agents = self.get_agents_and_set_score_and_perftag(layout_name, high_score_path_tag, AgentPerformance.HIGH)
        HM_agents = self.get_agents_and_set_score_and_perftag(layout_name, high_score_medium_path_tag, AgentPerformance.HIGH_MEDIUM)
        M_agents = self.get_agents_and_set_score_and_perftag(layout_name, medium_score_path_tag, AgentPerformance.MEDIUM)
        ML_agents = self.get_agents_and_set_score_and_perftag(layout_name, medium_score_low_path_tag, AgentPerformance.MEDIUM_LOW)
        L_agents = self.get_agents_and_set_score_and_perftag(layout_name, low_score_path_tag, AgentPerformance.LOW)

        all_agents = H_agents + HM_agents + M_agents + ML_agents + L_agents
        return all_agents