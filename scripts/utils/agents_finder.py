from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.tags import AgentPerformance, Prefix, KeyCheckpoints
from oai_agents.common.cklist_helper import get_layouts_from_cklist
from oai_agents.common.arguments import get_arguments
from oai_agents.common.path_helper import get_experiment_models_dir
from oai_agents.common.learner import LearnerType
from oai_agents.common.overcooked_simulation import OvercookedSimulation
from pathlib import Path


import os

from scripts.utils.layout_config import (
    two_chefs_dec_layouts,
    three_chefs_dec_layouts,
    five_chefs_dec_layouts,
    two_chefs_aamas24_layouts,
    three_chefs_aamas24_layouts,
    four_chefs_aamas24_layouts,
    five_chefs_aamas24_layouts,
    classic_layouts
)

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
        return {name: weight for name in cls.CATEGORY_NAMES}

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
    def __init__(self, model, layouts, category_weights=None, feature_weights=None):
        self.model = model  # Agent model
        self.layouts = set(layouts)  # Set of layouts the agent targets
        self.category_weights = category_weights or AgentCategory.default_weights()  # Default uniform weights
        self.feature_weights = feature_weights or {}  # Dictionary of feature weights

    def assign_category_weight(self, category):
        """
        Assigns pure weight for the specified category.
        Example: `assign_category_weight(AgentCategory.HIGH_PERFORMANCE)`
        """
        self.category_weights = AgentCategory.pure_weight(category)

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


class BasicProfileCollection:
    """
    Stores and organizes a collection of agent profiles, providing utilities for querying by layout.
    """
    def __init__(self, args):
        self.agent_profiles = []  # List to store all agent profiles
        self.layout_map = {}  # Maps layouts to lists of agent profiles
        self.args = args
        self.add_sp_agents()
        # self.add_dummy_agents()
        # self.add_fcp_agents()
        # self.add_advp_agents()

    def add_agent(self, agent_profile):
        """
        Adds an agent profile to the collection and updates the mappings.
        """
        self.agent_profiles.append(agent_profile)

        # Update layout_map
        for layout in agent_profile.layouts:
            if layout not in self.layout_map:
                self.layout_map[layout] = []
            self.layout_map[layout].append(agent_profile)

    def add_performance_agent(self, model, layout, category):
        """
        Adds an agent to the collection with a specific performance category.
        """
        agent = AgentProfile(model=model, layouts=[layout])
        agent.assign_category_weight(category=category)
        self.add_agent(agent)

    def add_sp_agents(self):
        agent_finder = SelfPlayAgentsFinder(args=self.args)
        agents, env_infos, training_infos = agent_finder.get_agents_infos()
        for training_info in training_infos:
            ck_list = training_info["ck_list"]
            layouts = get_layouts_from_cklist(ck_list=ck_list)
            for layout in layouts:
                print(f"layout: {layout}")
                h_agents, m_agents, l_agents = RLAgentTrainer.get_HML_agents_by_layout(
                    args=self.args, ck_list=ck_list, layout_name=layout,
                )
                assert len(h_agents) == 1
                self.add_performance_agent(model=h_agents[0], layout=layout, category=AgentCategory.HIGH_PERFORMANCE)
                assert len(m_agents) == 1
                self.add_performance_agent(model=m_agents[0], layout=layout, category=AgentCategory.MEDIUM_PERFORMANCE)
                assert len(l_agents) == 1
                self.add_performance_agent(model=l_agents[0], layout=layout, category=AgentCategory.LOW_PERFORMANCE)

    def get_agents_by_layout(self, layout):
        """
        Returns all agent profiles associated with a specific layout.
        """
        return self.layout_map.get(layout, [])

    def __repr__(self):
        return f"BasicProfileCollection({len(self.agent_profiles)} agent profiles stored)"

class AgentsFinder:
    def __init__(self, args, folders=None):
        self.args = args
        self.folders = folders
        self.target_dir = get_experiment_models_dir(base_dir=self.args.base_dir, exp_folder=self.args.exp_dir)

    def get_agentfolders_with_prefix(self, prefix):
        return [
            folder for folder in os.listdir(self.target_dir)
            if os.path.isdir(os.path.join(self.target_dir, folder)) and folder.startswith(prefix)
        ]

    def get_agentfolders_with_suffix(self, suffix):
        return [
            folder for folder in os.listdir(self.target_dir)
            if os.path.isdir(os.path.join(self.target_dir, folder)) and folder.endswith(suffix)
        ]

    def get_agentfolders_containing(self, substring):
        return [
            folder for folder in os.listdir(self.target_dir)
            if os.path.isdir(os.path.join(self.target_dir, folder)) and substring in folder
        ]

    def get_agents_infos(self, tag=None):
        all_agents = []
        env_infos = []
        training_infos = []
        assert len(self.folders)>0
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

class AdaptiveAgentsFinder(AgentsFinder):
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

if __name__ == '__main__':
    args = get_arguments()
    args.exp_dir = 'Selected/2'
    basic_profile = BasicProfileCollection(args=args)
    layouts = two_chefs_aamas24_layouts
    # for layout in two_chefs_aamas24_layouts:
    #     agents = basic_profile.layout_map[layout]
    #     simulation = OvercookedSimulation(args=args, agent=agents[0], teammates=[agents[1]], layout_name=layout, p_idx=p_idx, horizon=400)
    #     trajectories = simulation.run_simulation(how_many_times=1)