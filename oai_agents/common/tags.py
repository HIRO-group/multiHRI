from enum import Enum
import os
from oai_agents.common.path_helper import get_model_path

class AgentPerformance:
    '''
    Agent performance refers to the reward an agent receives after playing in
    self-play scenarios. For example, consider an agent, X, with
    AgentPerformance.HIGH.This means X has participated in an Overcooked game
    with multiple copies of itself, and the self-play team achieved a total
    reward categorized as high performance.
    '''
    HIGH = 'H'
    MEDIUM = 'M'
    LOW = 'L'

    ALL = [HIGH, MEDIUM, LOW]
    NOTSET = 'NS'


class TeamType:
    '''
    Team type refers to the type of agents in a team
    For example if teammates_len is 2, and the team type is HIGH_PRIORITY
    Then the list of agents are sorted based on score in a descending order
    and the first 2 agents are selected.

    SP: All agents are the same agent
    SPL: N-1 agents are the same agent, 1 agent is a low performing agent
    SPM: ...
    '''

    HIGH_FIRST = 'H'
    MEDIUM_FIRST = 'M'
    MIDDLE_FIRST = 'MID'
    LOW_FIRST = 'L'
    RANDOM = 'R'
    HIGH_MEDIUM = 'HM'
    HIGH_LOW = 'HL'
    MEDIUM_LOW = 'ML'
    HIGH_LOW_RANDOM = 'HLR'

    # Used to create a list of all possible permutations of agents from the teammate population
    # TODO: eventually, teammates_collection should be turned into its own class with 'select'
    # and 'update' functions that can be leveraged during training so the teammates_collection
    # doesn't need to be created before training begins, once that happens we can remove the AMX
    # type
    ALL_MIX = 'AMX'

    ALL_TYPES_BESIDES_SP = [HIGH_FIRST, MEDIUM_FIRST, MIDDLE_FIRST, LOW_FIRST, RANDOM, HIGH_MEDIUM, HIGH_LOW, MEDIUM_LOW, HIGH_LOW_RANDOM, ALL_MIX]

    SELF_PLAY = 'SP'
    SELF_PLAY_LOW = 'SPL'
    SELF_PLAY_MEDIUM = 'SPM'
    SELF_PLAY_MIDDLE = 'SPMID'
    SELF_PLAY_HIGH = 'SPH'
    SELF_PLAY_ADVERSARY = 'SPADV'

    SELF_PLAY_STATIC_ADV = 'SPSA' # Sits still
    SELF_PLAY_DYNAMIC_ADV = 'SPDA' # Moves around

    SELF_PLAY_X_TYPES = [SELF_PLAY_LOW, SELF_PLAY_MEDIUM, SELF_PLAY_MIDDLE, SELF_PLAY_HIGH, SELF_PLAY_ADVERSARY, SELF_PLAY_STATIC_ADV, SELF_PLAY_DYNAMIC_ADV]

    def map_to_index(teamtype):
        tt_map = {
            TeamType.LOW_FIRST: 0,
            TeamType.MIDDLE_FIRST: 1,
            TeamType.MEDIUM_FIRST: 2,
            TeamType.HIGH_FIRST: 3,
            TeamType.RANDOM: 4,
            TeamType.HIGH_MEDIUM: 5,
            TeamType.HIGH_LOW: 6,
            TeamType.MEDIUM_LOW: 7,
            TeamType.HIGH_LOW_RANDOM: 8,
            TeamType.SELF_PLAY: 9,
            TeamType.SELF_PLAY_LOW: 10,
            TeamType.SELF_PLAY_MEDIUM: 11,
            TeamType.SELF_PLAY_MIDDLE: 11.5,
            TeamType.SELF_PLAY_HIGH: 12,
            TeamType.SELF_PLAY_ADVERSARY: 13,
            TeamType.ALL_MIX: 14,
            TeamType.SELF_PLAY_STATIC_ADV: 15,
            TeamType.SELF_PLAY_DYNAMIC_ADV: 16
        }
        return tt_map[teamtype]

class TeammatesCollection:
    TRAIN = 'train'
    EVAL = 'eval'

class KeyCheckpoints: # Tags to identify the type of model checkpoint to save/load
    BEST_EVAL_REWARD = 'best' # Use only for evaluation
    MOST_RECENT_TRAINED_MODEL = 'last' # Use only for training
    CHECKED_MODEL_PREFIX = 'ck_'
    REWARD_SUBSTR = '_rew_'

    @staticmethod
    def get_most_recent_checkpoint(base_dir, exp_dir, name: str) -> str:
        path = get_model_path(
            base_dir=base_dir,
            exp_folder=exp_dir,
            model_name=name
        )
        if not path.exists():
            print(f"Warning: The directory {path} does not exist.")
            return None
        ckpts = [name for name in os.listdir(path) if name.startswith(KeyCheckpoints.CHECKED_MODEL_PREFIX)]
        if not ckpts:
            print(f"Warning: No checkpoints found in {path} with prefix '{KeyCheckpoints.CHECKED_MODEL_PREFIX}'.")
            return None
        ckpts_nums = [int(c.split('_')[1]) for c in ckpts]
        last_ckpt_num = max(ckpts_nums)
        return [c for c in ckpts if c.startswith(f"{KeyCheckpoints.CHECKED_MODEL_PREFIX}{last_ckpt_num}")][0]



class Prefix:
    SELF_PLAY = 'SP'
    FICTITIOUS_CO_PLAY = 'FCP'
    ADVERSARY = 'adv'
    ADVERSARY_PLAY = 'pwadv'

    @classmethod
    def find_folders_with_prefix(cls, base_dir, exp_dir, prefix):
        """
        Finds folders in a directory starting with the given prefix.

        Args:
            base_dir (str): Base directory path.
            exp_dir (str): Experiment directory path.
            prefix (str): Prefix to filter folders.

        Returns:
            list: List of matching folder names.
        """
        target_dir = os.path.join(base_dir, exp_dir)
        if not os.path.exists(target_dir):
            raise ValueError(f"The directory {target_dir} does not exist.")

        # List all items in the directory and filter for those starting with the prefix
        matching_folders = [
            folder for folder in os.listdir(target_dir)
            if os.path.isdir(os.path.join(target_dir, folder)) and folder.startswith(prefix)
        ]

        return matching_folders

class AdversaryPlayConfig:
    MAP = 'MultiAdversaryPlay' # adapts to a list of adversary [adv0, adv1, adv2]
    SAP = 'SingleAdversaryPlay' # adapts to the latest trained adversary
