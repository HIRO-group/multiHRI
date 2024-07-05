class AgentPerformance:
    '''
    Agent performance refers to the reward of the agent 
    '''
    HIGH = 'H'
    HIGH_MEDIUM = 'HM'
    MEDIUM = 'M'
    MEDIUM_LOW = 'ML'
    LOW = 'L'
    NOTSET = 'NS'
    ALL = [HIGH, HIGH_MEDIUM, MEDIUM, MEDIUM_LOW, LOW]


class TeamType:
    '''
    Team type refers to the type of agents in a team
    For example if teammates_len is 2, and the team type is HIGH_PRIORITY
    Then the list of agents are sorted based on score in a descending order
    and the first 2 agents are selected
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
    SELF_PLAY = 'SP'