from oai_agents.common.tags import TeamType

class EvalMembersToBeLoaded:
    def __init__(self, load_from_pop_structure, names, team_type, tags, layout_name):
        self.load_from_pop_structure = load_from_pop_structure
        self.names = names
        self.team_type = team_type
        self.tags = tags
        self.layout_name = layout_name

        if load_from_pop_structure:
            assert len(names) == 1, 'Only one name should be provided if reading from pop structure'
            assert len(tags) == 1, 'Only one tag should be provided if reading from pop structure'
        
        assert len(names) == len(tags), 'Number of names and tags should be the same'


def get_eval_types_to_load():
    '''
    If load_from_pop_structure is False, it means that we are reading independent agents from files.
    '''
    t1 = EvalMembersToBeLoaded(
        load_from_pop_structure = False,
        names = ['eval/2_chefs/fcp_hd256_seed26', 'eval/2_chefs/fcp_hd256_seed39'],
        team_type = TeamType.HIGH_FIRST,
        tags = ['best', 'best'],
        layout_name = '3_chefs_small_kitchen',
    )

    '''
    Pop structure holds the population of agent used for FCP training. 
    '''
    t2 = EvalMembersToBeLoaded(
        load_from_pop_structure = True,
        names = ['eval/2_chefs/fcp_pop_3_chefs_small_kitchen'],
        team_type = TeamType.HIGH_FIRST,
        tags = ['aamas25'],
        layout_name = '3_chefs_small_kitchen',
    )
    return [t1, t2]


