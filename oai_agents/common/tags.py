from enum import Enum
class AgentPerformance:
    '''
    Agent performance refers to the reward an agent receives after playing in 
    self-play scenarios. For example, consider an agent, X, with 
    AgentPerformance.HIGH.This means X has participated in an Overcooked game
    with multiple copies of itself, and the self-play team achieved a total 
    reward categorized as high performance.
    '''
    HIGH = 'H'
    HIGH_MEDIUM = 'HM'
    MEDIUM = 'M'
    MEDIUM_LOW = 'ML'
    LOW = 'L'

    ALL = [HIGH, HIGH_MEDIUM, MEDIUM, MEDIUM_LOW, LOW]
    
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

    ALL_TYPES_BESIDES_SP = [HIGH_FIRST, MEDIUM_FIRST, MIDDLE_FIRST, LOW_FIRST, RANDOM, HIGH_MEDIUM, HIGH_LOW, MEDIUM_LOW, HIGH_LOW_RANDOM]

    SELF_PLAY = 'SP'
    SELF_PLAY_LOW = 'SPL'
    SELF_PLAY_MEDIUM = 'SPM'
    SELF_PLAY_HIGH = 'SPH'
    SELF_PLAY_SABOTEUR = 'SPSAB'

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
            TeamType.SELF_PLAY_HIGH: 12,
            TeamType.SELF_PLAY_SABOTEUR: 13
        }
        return tt_map[teamtype]

class TeammatesCollection:
    TRAIN = 'train'
    EVAL = 'eval'

class CheckedPoints(Enum):
    BEST_EVAL_REWARD = "best"
    WORST_EVAL_REWARD ='worst'
    BEST_TRAIN_REWARD = "best_train_reward"
    WORST_TRAIN_REWARD ='worst_train_reward'
    FINAL_TRAINED_MODEL = "last"
