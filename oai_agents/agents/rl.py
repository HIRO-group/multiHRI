from oai_agents.agents.base_agent import SB3Wrapper, SB3LSTMWrapper, OAITrainer, PolicyClone
from oai_agents.common.arguments import get_arguments
from oai_agents.common.networks import OAISinglePlayerFeatureExtractor
from oai_agents.common.state_encodings import ENCODING_SCHEMES
from oai_agents.common.population_tags import AgentPerformance, TeamType
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
    def __init__(self, teammates_collection, selfplay, args, 
                epoch_timesteps, n_envs,
                seed, num_layers=2, hidden_dim=256, 
                fcp_ck_rate=None, name=None, env=None, eval_envs=None,
                use_cnn=False, use_lstm=False, use_frame_stack=False,
                taper_layers=False, use_policy_clone=False, deterministic=False):
        
        name = name or 'rl_agent'
        super(RLAgentTrainer, self).__init__(name, args, seed=seed)
        
        self.args = args
        self.device = args.device
        self.teammates_len = self.args.teammates_len
        self.num_players = self.args.num_players
        
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
        
        self.learning_agent, self.agents = self.get_learning_agent()
        self.teammates_collection, self.eval_teammates_collection = self.get_teammates_collection(teammates_collection, selfplay, self.learning_agent)
        self.best_score, self.best_training_rew = -1, float('-inf')


    def get_learning_agent(self):
        sb3_agent, agent_name = self.get_sb3_agent()
        learning_agent = self.wrap_agent(sb3_agent, agent_name)
        agents = [learning_agent]
        return learning_agent, agents


    def get_teammates_collection(self, _tms_clctn, selfplay, learning_agent):
        if not _tms_clctn and not selfplay:
            raise ValueError('Either a teammates_collection with len > 0 must be passed in or selfplay must be true')

        if selfplay:
            '''
            list
            teammates_collection = 
            {
                'all_layouts': {
                    TeamType.SELF_PLAY: [agent1, agent1]
                }
            }
            '''
            teammates_collection = {}
            teammates_collection['all_layouts'] = {
                TeamType.SELF_PLAY: [learning_agent for _ in range(self.teammates_len)]
            }
        else:
            '''
            dict 
            teammates_collection = {
                'layout_name': {
                    'high': [agent1, agent2],
                    'medium': [agent3, agent4],
                    'low': [agent5, agent6],
                    'random': [agent7, agent8],
                },
            }
            '''
            teammates_collection = {}
            for layout in self.args.layout_names:
                teammates_collection[layout] = {
                            TeamType.HIGH_FIRST: _tms_clctn[layout][TeamType.HIGH_FIRST],
                            TeamType.LOW_FIRST: _tms_clctn[layout][TeamType.LOW_FIRST],
                            }
                if self.teammates_len >= 2:
                    teammates_collection[layout][TeamType.HIGH_LOW_RANDOM] = _tms_clctn[layout][TeamType.HIGH_LOW_RANDOM]

        self.check_teammates_collection_structure(teammates_collection)
        
        eval_teammates_collection = teammates_collection
        return teammates_collection, eval_teammates_collection


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
        policy_kwargs = dict(net_arch=[dict(pi=layers, vf=layers)])

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
        IF we use FCP:
        teammates_collection = {
                'layout_name': {
                    'high': [agent1, agent2],
                    'medium': [agent3, agent4],
                    'low': [agent5, agent6],
                    'random': [agent7, agent8],
                },
            }

        IF we use SP:
        teammates_collection = 
            {
                'all_layouts': {
                    TeamType.SELF_PLAY: [agent1, agent1]
                }
            }
        '''
 
        if type(teammates_collection) == dict:
            for layout in teammates_collection: 
                for team_type in teammates_collection[layout]:
                    assert len(teammates_collection[layout][team_type]) == self.teammates_len
        else:
            raise ValueError('teammates_collection must be a dict with layout names as keys and a list of agents as values')


    def _get_constructor_parameters(self):
        return dict(args=self.args, name=self.name, use_lstm=self.use_lstm, use_frame_stack=self.use_frame_stack,
                    hidden_dim=self.hidden_dim, seed=self.seed)

    def wrap_agent(self, sb3_agent, name):
        if self.use_lstm:
            agent = SB3LSTMWrapper(sb3_agent, name, self.args)
        else:
            agent = SB3Wrapper(sb3_agent, name, self.args)
        return agent


    def train_agents(self, total_train_timesteps, exp_name=None):
        print("Training agent: "+self.name)

        run = wandb.init(project="overcooked_ai", entity=self.args.wandb_ent, dir=str(self.args.base_dir / 'wandb'),
                         reinit=True, name= exp_name or self.args.exp_name + '_' + self.name, mode=self.args.wandb_mode,
                         resume="allow")

        if self.fcp_ck_rate is not None:
            self.ck_list = []
            path, tag = self.save_agents(tag=f'ck_{len(self.ck_list)}')
            self.ck_list.append(({k: 0 for k in self.args.layout_names}, path, tag))

        best_path, best_tag = None, None
        
        steps = 0
        curr_timesteps = 0
        prev_timesteps = self.learning_agent.num_timesteps

        while curr_timesteps < total_train_timesteps:
            self.set_new_teammates()
            self.learning_agent.learn(self.epoch_timesteps)

            # Learning_agent.num_timesteps = n_steps * n_envs
            curr_timesteps += self.learning_agent.num_timesteps - prev_timesteps
            prev_timesteps = self.learning_agent.num_timesteps

            # Evaluate
            mean_training_rew = np.mean([ep_info["r"] for ep_info in self.learning_agent.agent.ep_info_buffer])
            self.best_training_rew *= 0.98

            if (steps + 1) % 5 == 0 or (mean_training_rew > self.best_training_rew and self.learning_agent.num_timesteps >= 5e6) or \
                (self.fcp_ck_rate and self.learning_agent.num_timesteps // self.fcp_ck_rate > (len(self.ck_list) - 1)):
                
                if mean_training_rew >= self.best_training_rew:
                    self.best_training_rew = mean_training_rew
                mean_reward, rew_per_layout = self.evaluate(self.learning_agent, timestep=self.learning_agent.num_timesteps)

                # FCP pop checkpointing
                if self.fcp_ck_rate:
                    if self.learning_agent.num_timesteps // self.fcp_ck_rate > (len(self.ck_list) - 1):
                        path, tag = self.save_agents(tag=f'ck_{len(self.ck_list)}')
                        self.ck_list.append((rew_per_layout, path, tag))
                
                # Save best model
                if mean_reward >= self.best_score:
                    print(f'New best score of {mean_reward} reached, model saved to {best_path}/{best_tag}')
                    best_path, best_tag = self.save_agents(tag='best')
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
