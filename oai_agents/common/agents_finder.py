from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.tags import Prefix, KeyCheckpoints
from oai_agents.agents.base_agent import OAIAgent, OAITrainer
from oai_agents.common.arguments import get_arguments
from oai_agents.common.path_helper import get_experiment_models_dir
from oai_agents.common.learner import LearnerType
from oai_agents.common.overcooked_simulation import OvercookedSimulation
from oai_agents.common.teammates_collection import get_best_SP_agent
from typing import List


import os

from scripts.utils.layout_config import (
    complex_5_chefs_layouts
)

def get_layouts_from_cklist(ck_list):
    assert ck_list is not None
    scores, _, _ = ck_list[0]
    assert isinstance(scores, dict)
    return list(scores.keys())

class AgentCategory:
    """
    Represents the categories of agents with default uniform weights.
    """
    HIGH_PERFORMANCE = "high_performance"
    MEDIUM_PERFORMANCE = "medium_performance"
    LOW_PERFORMANCE = "low_performance"
    IDLE = "idle"
    FLEXIBLE = "flexible"
    SELFISH = "selfish"

    CATEGORY_NAMES = [
        HIGH_PERFORMANCE,
        MEDIUM_PERFORMANCE,
        LOW_PERFORMANCE,
        IDLE,
        FLEXIBLE,
        SELFISH
    ]

    @classmethod
    def default_weights(cls):
        """Returns default uniform weights for all categories."""
        weight = 1 / len(cls.CATEGORY_NAMES)
        return dict.fromkeys(cls.CATEGORY_NAMES, weight)

    @classmethod
    def pure_weight(cls, category):
        """Returns weights for a pure category (1 for the category, 0 for others)."""
        if category not in cls.CATEGORY_NAMES:
            raise ValueError(f"Invalid category: {category}")
        return {name: (1.0 if name == category else 0.0) for name in cls.CATEGORY_NAMES}


class AgentProfile:
    """
    Represents an individual agent profile with a model, target layouts,
    category weights, and feature weights.
    """
    def __init__(self, model: OAIAgent, layouts, category_weights=None, feature_weights=None):
        self.model = model  # Agent model
        self.layouts = set(layouts)  # Set of layouts the agent targets
        self.category_weights = category_weights or AgentCategory.default_weights()  # Default uniform weights
        self.category = ""
        self.feature_weights = feature_weights or {}  # Dictionary of feature weights

    def assign_category_weight(self, category):
        """
        Assigns pure weight for the specified category.
        Example: `assign_category_weight(AgentCategory.HIGH_PERFORMANCE)`
        """
        self.category_weights = AgentCategory.pure_weight(category)
        self.category = category

    def category_weights_to_list(self):
        """
        Converts category_weights dictionary to a list of values in the same order
        as CATEGORY_NAMES.
        """
        return [self.category_weights[name] for name in AgentCategory.CATEGORY_NAMES]

    def __repr__(self):
        def format_dict(d):
            return ",\n            ".join(f"{k}: {v:.2f}" for k, v in d.items())

        categories = format_dict({k: v * 100 for k, v in self.category_weights.items()})
        features = format_dict(self.feature_weights)

        return (
            f"AgentProfile(\n"
            f"    model={self.model},\n"
            f"    category_weights={{\n        {categories}\n    }},\n"
            f"    feature_weights={{\n        {features}\n    }},\n"
            f"    layouts={list(self.layouts)}\n"
            f")"
        )

class AgentsFinder:
    def __init__(self, args, folders=None):
        self.args = args
        self.folders = folders
        self.target_dir = get_experiment_models_dir(base_dir=self.args.base_dir, exp_folder=self.args.exp_dir)

    def directory_exists_and_nonempty(self):
        """Check if target_dir exists and contains at least one folder."""
        if not os.path.exists(self.target_dir) or not os.path.isdir(self.target_dir):
            print(f"Warning: Directory '{self.target_dir}' does not exist.")
            return False
        if not os.listdir(self.target_dir):  # Check if directory is empty
            print(f"Warning: Directory '{self.target_dir}' is empty.")
            return False
        return True

    def get_agentfolders_with_prefix(self, prefix):
        if self.directory_exists_and_nonempty():
            return [
                folder for folder in os.listdir(self.target_dir)
                if os.path.isdir(os.path.join(self.target_dir, folder)) and folder.startswith(prefix)
            ]
        else:
            return []

    def get_agentfolders_with_suffix(self, suffix):
        if self.directory_exists_and_nonempty():
            return [
                folder for folder in os.listdir(self.target_dir)
                if os.path.isdir(os.path.join(self.target_dir, folder)) and folder.endswith(suffix)
            ]
        else:
            return []

    def get_agentfolders_containing(self, substring):
        if self.directory_exists_and_nonempty():
            return [
                folder for folder in os.listdir(self.target_dir)
                if os.path.isdir(os.path.join(self.target_dir, folder)) and substring in folder
            ]
        else:
            return []

    def get_agents_infos(self, tag=None):
        all_agents = []
        env_infos = []
        training_infos = []
        if len(self.folders)>0:
            for folder in self.folders:
                if tag is not None:
                    agents, env_info, training_info = RLAgentTrainer.load_agents(
                        args=self.args,
                        name=folder,
                        tag=tag
                    )
                else:
                    last_ckpt = KeyCheckpoints.get_most_recent_checkpoint(
                        base_dir=self.args.base_dir,
                        exp_dir=self.args.exp_dir,
                        name=folder,
                    )
                    agents, env_info, training_info = RLAgentTrainer.load_agents(
                        args=self.args,
                        name=folder,
                        tag=last_ckpt
                    )
                all_agents.append(agents[0])
                env_infos.append(env_info)
                training_infos.append(training_info)

        return all_agents, env_infos, training_infos

    def get_agents(self, tag= None):
        agents, _, _ = self.get_agents_infos(tag=tag)
        return agents

class SelfPlayAgentsFinder(AgentsFinder):
    def get_agents_infos(self, tag=None):
        self.folders = self.get_agentfolders_with_prefix(prefix=Prefix.SELF_PLAY)
        return super().get_agents_infos(tag=tag)

class AdversaryAgentsFinder(AgentsFinder):
    def get_agents_infos(self, tag=None):
        self.folders = self.get_agentfolders_with_suffix(suffix=LearnerType.SELFISHER)
        return super().get_agents_infos(tag=tag)

class FCPAgentsFinder(AgentsFinder):
    def get_agents_infos(self, tag=None):
        self.folders = self.get_agentfolders_with_suffix(suffix=f"tr[SPH_SPM_SPL]_ran_{LearnerType.ORIGINALER}")
        return super().get_agents_infos(tag=tag)

class AgentsFinderByKey(AgentsFinder):
    def get_agents(self, key, tag=None):
        agents, _, _ = self.get_agents_infos(key=key, tag=tag)
        return agents

class AgentsFinderBySuffix(AgentsFinderByKey):
    def get_agents_infos(self, key, tag=None):
        self.folders = self.get_agentfolders_with_suffix(suffix=key)
        return super().get_agents_infos(tag=tag)

class AMMAS23AgentsFinder(AgentsFinder):
    def get_agents_infos(self, tag=None):
        return self.get_agents(tag=tag)
    def get_agents(self, tag):
        all_agents = []
        assert len(self.folders)>0
        for folder in self.folders:
            if tag is not None:
                agents = RLAgentTrainer.only_load_agents(
                    args=self.args,
                    name=folder,
                    tag=tag
                )
            else:
                last_ckpt = KeyCheckpoints.get_most_recent_checkpoint(
                    base_dir=self.args.base_dir,
                    exp_dir=self.args.exp_dir,
                    name=folder,
                )
                agents = RLAgentTrainer.only_load_agents(
                    args=self.args,
                    name=folder,
                    tag=last_ckpt
                )
            for agent in agents:
                all_agents.append(agent)

        return all_agents

class AMMAS23AgentsFinderBySuffix(AMMAS23AgentsFinder):
    def get_agents(self, key, tag=None):
        self.folders = self.get_agentfolders_with_suffix(suffix=key)
        return super().get_agents(tag=tag)

class HMLProfileCollection:
    """
    Stores and organizes a collection of agent profiles, providing utilities for querying by layout.
    """
    def __init__(self, args, agents_finder: AgentsFinder):
        self.agent_profiles: List[AgentProfile] = []  # List to store all agent profiles
        self.layout_map = {}  # Maps layouts to lists of agent profiles
        self.args = args
        self._population = {}
        self.agents_finder: AgentsFinder = agents_finder
        self.add_multiple_agents()

    def add_agent(self, agent_profile):
        self.agent_profiles.append(agent_profile)
        for layout in agent_profile.layouts:
            if layout not in self.layout_map:
                self.layout_map[layout] = []
            self.layout_map[layout].append(agent_profile)

    def add_performance_agent(self, model, layout, category:AgentCategory):
        agent = AgentProfile(model=model, layouts=[layout])
        agent.assign_category_weight(category=category)
        self.add_agent(agent)

    def add_multiple_agents(self):
        agents, env_infos, training_infos = self.agents_finder.get_agents_infos()
        for training_info in training_infos:
            ck_list = training_info["ck_list"]
            layouts = get_layouts_from_cklist(ck_list=ck_list)
            for layout in layouts:
                if layout not in self._population:
                    self._population[layout] = []
                h_agents, m_agents, l_agents = RLAgentTrainer.get_HML_agents_by_layout(
                    args=self.args, ck_list=ck_list, layout_name=layout,
                )
                assert len(h_agents) == 1
                self.add_performance_agent(model=h_agents[0], layout=layout, category=AgentCategory.HIGH_PERFORMANCE)
                self._population[layout].append(h_agents[0])
                assert len(m_agents) == 1
                self.add_performance_agent(model=m_agents[0], layout=layout, category=AgentCategory.MEDIUM_PERFORMANCE)
                self._population[layout].append(m_agents[0])
                assert len(l_agents) == 1
                self.add_performance_agent(model=l_agents[0], layout=layout, category=AgentCategory.LOW_PERFORMANCE)
                self._population[layout].append(l_agents[0])


    def get_population(self) -> dict:
        return self._population

    def save_population(self):
        for layout in self._population.keys():
            pop = OAITrainer(
                name=f"pop_{layout}",
                args=self.args,
            )
            pop.agents = self._population[layout]
            pop.save_agents(tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL)

    def get_agentprofiles_by_layout(self, layout):
        """
        Returns all agent profiles associated with a specific layout.
        """
        return self.layout_map.get(layout, [])

    def __repr__(self):
        return f"BasicProfileCollection({len(self.agent_profiles)} agent profiles stored)"

if __name__ == '__main__':
    args = get_arguments()
    args.exp_dir = 'Complex/5'
    args.layout_names = complex_5_chefs_layouts
    args.num_players = 5
    agents_finder = SelfPlayAgentsFinder(args=args)
    hml_profiles = HMLProfileCollection(args=args, agents_finder=agents_finder)
    hml_profiles.save_population()
    agent = get_best_SP_agent(args=args, population=hml_profiles.get_population())
    print(f"agent.name: {agent.name}")
    score = 0
    for layout in hml_profiles.get_population().keys():
        score += agent.layout_scores[layout]
        print(f"agent.layout_scores[{layout}]: {agent.layout_scores[layout]}")
    print(f"average score: {score/len(hml_profiles.get_population().keys())}")
    print("\n\n")

    for layout in hml_profiles.get_population().keys():
        agent_profiles = hml_profiles.layout_map[layout]
        for agent_profile in agent_profiles:
            agent: OAIAgent = agent_profile.model
            simulation = OvercookedSimulation(args=args, agent=agent, teammates=[agent for i in range(args.num_players-1)], layout_name=layout, p_idx=0, horizon=400)
            trajectories = simulation.run_simulation(how_many_times=1)
            print(f"agent.name: {agent.name}")
            print(f"agent category: {agent_profile.category}")
            score = 0
            for l in hml_profiles.get_population().keys():
                score += agent.layout_scores[l]
                print(f"agent.layout_scores[{l}]: {agent.layout_scores[l]}")
            print(f"average score: {score/len(hml_profiles.get_population().keys())}")
            print(f"simulation score in {layout}: {sum(trajectories[0]['rewards'])}\n")
