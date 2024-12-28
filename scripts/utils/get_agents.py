from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.tags import AgentPerformance, Prefix, KeyCheckpoints
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

def get_layoutnames_from_cklist(ck_list):
    assert ck_list is not None
    scores, _, _ = ck_list[0]
    assert isinstance(scores, dict)
    return list(ck_list.keys())

def get_agents_by_layout(ck_list):
    layout_names = get_layoutnames_from_cklist(ck_list=ck_list)
    H_agents = []
    M_agents = []
    L_agents = []

    for layout_name in layout_names:
        H_agent, M_agent, L_agent = RLAgentTrainer.get_HML_agents_by_layout(
            ck_list=ck_list,
            layout_name=layout_name
        )


class AgentCategory:
    HIGH_PERFORMANCE = "high_performance"
    MEDIUM_PERFORMANCE = "medium_performance"
    LOW_PERFORMANCE = "low_performance"
    IDLE = "idle"
    FLEXIBLE = "flexible"
    SELFISH = "selfish"

class AgentProfile:
    """
    Represents an individual agent profile with a model, target layouts,
    category weights, and feature weights.
    """
    def __init__(self, model, layouts, category_weights, feature_weights):
        self.model = model  # Agent model
        self.layouts = set(layouts)  # Set of layouts the agent targets
        self.category_weights = category_weights  # Dictionary of category weights
        self.feature_weights = feature_weights  # Dictionary of feature weights

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
    def __init__(self):
        self.agent_profiles = []  # List to store all agent profiles
        self.layout_map = {}  # Maps layouts to lists of agent profiles

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

    def get_agents_by_layout(self, layout):
        """
        Returns all agent profiles associated with a specific layout.
        """
        return self.layout_map.get(layout, [])

    def __repr__(self):
        return f"BasicProfileCollection({len(self.agent_profiles)} agent profiles stored)"

class AgentsZoo:
    def __init__(self, args):
        self.layout_names = classic_layouts
        self.agent_dict = {layout: [] for layout in self.layout_names}
        self.args = args
        self.fill_SP_agents()
        self.fill_dummy_agents()
        self.fill_FCP_agents()

    def fill_SP_agents(self):
        folders = Prefix.find_folders_with_prefix(
            base_dir=self.args.base_dir,
            exp_dir=self.args.exp_dir,
            prefix=Prefix.SELF_PLAY
        )

        for folder in folders:
            last_ckpt = KeyCheckpoints.get_most_recent_checkpoint(
                base_dir=self.args.base_dir,
                exp_dir=self.args.exp_dir,
                name=folder,
            )
            _, _, training_info = RLAgentTrainer.load_agents(
                args=self.args,
                name=folder,
                tag=last_ckpt
            )
            ck_list = training_info["ck_list"]
