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
    Instructions:

    If load_from_pop_structure is False, it means that we are reading independent agents from files.
    t1 = EvalMembersToBeLoaded(
        load_from_pop_structure = False,
        names = ['eval/3_chefs/fcp_hd256_seed26', 'eval/3_chefs/fcp_hd256_seed39'],
        team_type = TeamType.HIGH_FIRST,
        tags = ['best', 'best'],
        layout_name = '3_chefs_small_kitchen',
    )

    Pop structure holds the population of agent used for FCP training.
    t2 = EvalMembersToBeLoaded(
        load_from_pop_structure = True,
        names = ['eval/3_chefs/fcp_pop_3_chefs_small_kitchen'],
        team_type = TeamType.HIGH_FIRST,
        tags = ['last'],
        layout_name = '3_chefs_small_kitchen',
    )
    '''

    t_pop_h = EvalMembersToBeLoaded(
            load_from_pop_structure = True,
            names = ['eval/3_chefs/fcp_pop_3_chefs_small_kitchen'],
            team_type = TeamType.HIGH_FIRST,
            tags = ['last'],
            layout_name = '3_chefs_small_kitchen',
        )

    t_pop_l = EvalMembersToBeLoaded(
        load_from_pop_structure = True,
        names = ['eval/3_chefs/fcp_pop_3_chefs_small_kitchen'],
        team_type = TeamType.LOW_FIRST,
        tags = ['last'],
        layout_name = '3_chefs_small_kitchen',
    )

    t_pop_m = EvalMembersToBeLoaded(
        load_from_pop_structure = True,
        names = ['eval/3_chefs/fcp_pop_3_chefs_small_kitchen'],
        team_type = TeamType.MEDIUM_FIRST,
        tags = ['aamas25'],
        layout_name = '3_chefs_small_kitchen',
    )

    t_pop_hm = EvalMembersToBeLoaded(
        load_from_pop_structure = True,
        names = ['eval/3_chefs/fcp_pop_3_chefs_small_kitchen'],
        team_type = TeamType.HIGH_MEDIUM,
        tags = ['last'],
        layout_name = '3_chefs_small_kitchen',
    )

    t_pop_hl = EvalMembersToBeLoaded(
        load_from_pop_structure = True,
        names = ['eval/3_chefs/fcp_pop_3_chefs_small_kitchen'],
        team_type = TeamType.HIGH_LOW,
        tags = ['last'],
        layout_name = '3_chefs_small_kitchen',
    )


    t_fcp_h = EvalMembersToBeLoaded(
        load_from_pop_structure = False,
        names = ['eval/3_chefs/fcp_H', 'eval/3_chefs/fcp_H'],
        team_type = "FCP_HH", # We can choose arbitrary name here for wandb plots purposes
        tags = ['best', 'best'],
        layout_name = '3_chefs_small_kitchen',
    )

    t_fcp_hl = EvalMembersToBeLoaded(
        load_from_pop_structure = False,
        names = ['eval/3_chefs/fcp_HL', 'eval/3_chefs/fcp_HL'],
        team_type = "FCP_HL",
        tags = ['best', 'best'],
        layout_name = '3_chefs_small_kitchen',
    )

    t_fcp_hm = EvalMembersToBeLoaded(
        load_from_pop_structure = False,
        names = ['eval/3_chefs/fcp_HM', 'eval/3_chefs/fcp_HM'],
        team_type = "FCP_HM",
        tags = ['best', 'best'],
        layout_name = '3_chefs_small_kitchen',
    )

    t_fcp_h_l = EvalMembersToBeLoaded(
        load_from_pop_structure = False,
        names = ['eval/3_chefs/fcp_H_L', 'eval/3_chefs/fcp_H_L'],
        team_type = "FCP_H_L",
        tags = ['best', 'best'],
        layout_name = '3_chefs_small_kitchen',
    )

    t_fcp_h_mid = EvalMembersToBeLoaded(
        load_from_pop_structure = False,
        names = ['eval/3_chefs/fcp_H_MID', 'eval/3_chefs/fcp_H_MID'],
        team_type = "FCP_H_MID",
        tags = ['best', 'best'],
        layout_name = '3_chefs_small_kitchen',
    )

    t_sp_h = EvalMembersToBeLoaded(
        load_from_pop_structure = False,
        names = ['eval/3_chefs/fcp_hd64_seed13', 'eval/3_chefs/fcp_hd64_seed13'],
        team_type = "SP_H",
        tags = ['best', 'best'],
        layout_name = '3_chefs_small_kitchen')

    t_fcp_hl_hm = EvalMembersToBeLoaded(
        load_from_pop_structure = False,
        names = ['eval/3_chefs/fcp_HL_HM', 'eval/3_chefs/fcp_HL_HM'],
        team_type = "FCP_HL_HM",
        tags = ['best', 'best'],
        layout_name = '3_chefs_small_kitchen',
    )

    t_fcp_h_mid_l = EvalMembersToBeLoaded(
        load_from_pop_structure = False,
        names = ['eval/3_chefs/fcp_H_MID_L', 'eval/3_chefs/fcp_H_MID_L'],
        team_type = "FCP_H_MID_L",
        tags = ['best', 'best'],
        layout_name = '3_chefs_small_kitchen',
    )
    t_fcp_ml = EvalMembersToBeLoaded(
        load_from_pop_structure = False,
        names = ['eval/3_chefs/fcp_ML', 'eval/3_chefs/fcp_ML'],
        team_type = "FCP_ML",
        tags = ['best', 'best'],
        layout_name = '3_chefs_small_kitchen',
    )
    return [t_sp_h, t_pop_m, t_fcp_h, t_fcp_h_mid, t_fcp_h_l, t_fcp_hl, t_fcp_hl_hm, t_fcp_h_mid_l, t_fcp_ml]
