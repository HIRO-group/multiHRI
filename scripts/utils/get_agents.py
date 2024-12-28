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
