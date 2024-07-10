class AgentPerformance:
    '''
    Agent performance refers to the reward of the agent 
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

class TeammatesCollection:
    TRAIN = 't'
    EVAL = 'e'