import multiprocessing as mp
mp.set_start_method('spawn', force=True) # should be called before any other module imports

from oai_agents.common.arguments import get_arguments
from oai_agents.common.tags import TeamType, AdversaryPlayConfig, KeyCheckpoints
from oai_agents.common.curriculum import Curriculum
from oai_agents.common.agents_finder import HMLProfileCollection, SelfPlayAgentsFinder
from oai_agents.agents.mep_population_manager import MEPPopulationManager


from scripts.utils import (
    get_SP_agents,
    get_FCP_agent_w_pop,
    get_N_X_FCP_agents,
    get_N_X_SP_agents,
    get_best_EGO_agents,
)

def MEP_POPULATION(args):
    # AgentsFinder can find agent under the directory, f"{args.exp_dir}/{args.num_players}",
    # by its method get_agents_infos
    agents_finder = SelfPlayAgentsFinder(args=args)
    _, _, training_infos = agents_finder.get_agents_infos()
    if len(training_infos)==0:
        manager = MEPPopulationManager(population_size=args.total_sp_agents, args=args)
        manager.train_population(
            total_timesteps=args.pop_total_training_timesteps,
            num_of_ckpoints=args.num_of_ckpoints,
            eval_interval = args.eval_steps_interval * args.epoch_timesteps
        )
    # HMLProfileCollection use agents_finder to find information of multiple agents and
    # call save_population to save pop files under args.layout_names
    hml_profiles = HMLProfileCollection(args=args, agents_finder=agents_finder)
    hml_profiles.save_population()

def SP(args):
    primary_train_types = [TeamType.SELF_PLAY]
    primary_eval_types = {
        'generate': [TeamType.SELF_PLAY],
        'load': []
    }
    curriculum = Curriculum(train_types=primary_train_types, is_random=True)

    get_SP_agents(
        args=args,
        train_types=curriculum.train_types,
        eval_types=primary_eval_types,
        curriculum=curriculum,
        tag_for_returning_agent=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL
    )


def SPN_1ADV(args) -> None:
    '''
    In N-agents games, a randomly initialized agent will be trained with either one of two conditions:
    (a)N-1 copies of itself and 1 unseen adversary teammate.
    (b)N copies of itself

    e.g.
    when N is 4, the team can be composed by [SP, SP, SP, SP] or [SP, SP, SP, ADV] in a 4-chef layout.
    '''
    attack_rounds = 3
    unseen_teammates_len = 1
    adversary_play_config = AdversaryPlayConfig.MAP
    primary_train_types = [TeamType.SELF_PLAY, TeamType.SELF_PLAY_ADVERSARY]

    primary_eval_types = {
        'generate': [
            TeamType.SELF_PLAY_HIGH,
            TeamType.SELF_PLAY_LOW,
            TeamType.SELF_PLAY_ADVERSARY
        ],
        'load': []
    }

    curriculum = Curriculum(
        train_types = primary_train_types, is_random = True)
    get_N_X_SP_agents(
        args,
        n_x_sp_train_types=curriculum.train_types,
        n_x_sp_eval_types=primary_eval_types,
        curriculum=curriculum,
        unseen_teammates_len=unseen_teammates_len,
        adversary_play_config=adversary_play_config,
        attack_rounds=attack_rounds
    )


def SPN_1ADV_XSPCKP(args) -> None:
    '''
    In N-agents games, a randomly initialized agent will be trained with N-X copies of itself and X unseen teammates.
    X unseen teammates can be composed by either one of the two conditions:
    (a) 1 adversary and X-1 self-play checkpoints.
    (b) X self-play checkpoints.
    e.g.
    when N is 4 and X is 1, the team can be composed by [SP, SP, SP, ADV] or [SP, SP, SP, H] or [SP, SP, SP, M] or [SP, SP, SP, L] in a 4-chef layout.
    when N is 4 and X is 2, the team can be composed
    [SP, SP, ADV, H] or [SP, SP, ADV, M] or [SP, SP, ADV, L] or
    [SP, SP, H, H] or [SP, SP, M, M] or [SP, SP, L, L] in a 4-chef layout.

    - X is the number of unseen teammate.
    - X is assigned by the variable, unseen_teammates_len, in the funciton.
    '''
    attack_rounds = 3
    unseen_teammates_len = 1
    adversary_play_config = AdversaryPlayConfig.MAP
    primary_train_types = [
        TeamType.SELF_PLAY_HIGH,
        TeamType.SELF_PLAY_MEDIUM,
        TeamType.SELF_PLAY_ADVERSARY
    ]

    primary_eval_types = {
        'generate': [
            TeamType.SELF_PLAY_HIGH,
            TeamType.SELF_PLAY_MEDIUM,
            TeamType.SELF_PLAY_ADVERSARY
        ],
        'load': []
    }

    if args.prioritized_sampling:
        curriculum = Curriculum(
            train_types = primary_train_types,
            eval_types=primary_eval_types,
            prioritized_sampling=True,
        )

    else:
        curriculum = Curriculum(
            train_types = primary_train_types,
            is_random=False,
            prioritized_sampling=True,
            total_steps = args.n_x_sp_total_training_timesteps//args.epoch_timesteps,
            training_phases_durations_in_order={
                (TeamType.SELF_PLAY_ADVERSARY): 0.5,
            },
            rest_of_the_training_probabilities={
                TeamType.SELF_PLAY_MEDIUM: 0.3,
                TeamType.SELF_PLAY_HIGH: 0.3,
                TeamType.SELF_PLAY_ADVERSARY: 0.4,
            },
            probabilities_decay_over_time=0
        )
    get_N_X_SP_agents(
        args,
        n_x_sp_train_types=curriculum.train_types,
        n_x_sp_eval_types=primary_eval_types,
        curriculum=curriculum,
        unseen_teammates_len=unseen_teammates_len,
        adversary_play_config=adversary_play_config,
        attack_rounds=attack_rounds
    )

def FCP_mhri(args):
    '''
    There are two types of FCP, one is the traditional FCP that uses random teammates (i.e. ALL_MIX),
    one is our own version that uses certain types HIGH_FIRST, MEDIUM_FIRST, etc.
    The reason we have our version is that when we used the traditional FCP it got ~0 reward so we
    decided to add different types for teammates_collection.
    '''
    primary_train_types = [TeamType.LOW_FIRST, TeamType.MEDIUM_FIRST, TeamType.HIGH_FIRST]
    primary_eval_types = {'generate' : [TeamType.HIGH_FIRST],
                          'load': []}

    fcp_curriculum = Curriculum(
        train_types = primary_train_types,
        is_random=False,
        total_steps = args.fcp_total_training_timesteps//args.epoch_timesteps,
        training_phases_durations_in_order={
            (TeamType.LOW_FIRST): 0.5,
            (TeamType.MEDIUM_FIRST): 0.125,
            (TeamType.HIGH_FIRST): 0.125,
        },
        rest_of_the_training_probabilities={
            TeamType.LOW_FIRST: 0.4,
            TeamType.MEDIUM_FIRST: 0.3,
            TeamType.HIGH_FIRST: 0.3,
        },
        probabilities_decay_over_time=0
    )

    _, _ = get_FCP_agent_w_pop(
        args,
        fcp_train_types = fcp_curriculum.train_types,
        fcp_eval_types=primary_eval_types,
        fcp_curriculum=fcp_curriculum
    )


def N_1_FCP(args):
    unseen_teammates_len = 1 # This is the X in FCP_X_SP

    fcp_train_types = [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, TeamType.LOW_FIRST]
    fcp_eval_types = {'generate' : [], 'load': []}
    fcp_curriculum = Curriculum(train_types=fcp_train_types, is_random=True)

    primary_train_types = [
        TeamType.SELF_PLAY_LOW,
        TeamType.SELF_PLAY_MEDIUM,
        TeamType.SELF_PLAY_HIGH
    ]
    primary_eval_types = {
        'generate': [
            TeamType.SELF_PLAY_LOW,
            TeamType.SELF_PLAY_MEDIUM,
            TeamType.SELF_PLAY_HIGH
        ],
        'load': []
    }
    n_1_fcp_curriculum = Curriculum(train_types=primary_train_types, is_random=True)

    get_N_X_FCP_agents(
        args=args,
        fcp_train_types=fcp_curriculum.train_types,
        fcp_eval_types=fcp_eval_types,
        n_1_fcp_train_types=n_1_fcp_curriculum.train_types,
        n_1_fcp_eval_types=primary_eval_types,
        fcp_curriculum=fcp_curriculum,
        n_1_fcp_curriculum=n_1_fcp_curriculum,
        unseen_teammates_len=unseen_teammates_len
    )



def FCP_traditional(args):
    '''
    The ALL_MIX TeamType enables truly random teammates when training (like in the original FCP
    implementation)
    '''
    primary_train_types = [TeamType.ALL_MIX]
    primary_eval_types = {
        'generate' : [TeamType.HIGH_FIRST, TeamType.LOW_FIRST],
        'load': []
    }
    fcp_curriculum = Curriculum(train_types=primary_train_types, is_random=True)
    _, _ = get_FCP_agent_w_pop(
        args,
        fcp_train_types=fcp_curriculum.train_types,
        fcp_eval_types=primary_eval_types,
        fcp_curriculum=fcp_curriculum,
    )


def SPN_XSPCKP(args) -> None:
    '''
    In N-agents games, a randomly initialized agent will be trained with N-X copies of itself
    and X homogeneous unseen teammates, which are checkpoints saved during a previous self-play process.
    when N is 4 and X is 1, the team can be composed by [SP, SP, SP, H], [SP, SP, SP, M], [SP, SP, SP, L] in a 4-chef layout.
    when N is 4 and X is 2, the team can be composed [SP, SP, H, H], [SP, SP, M, M], [SP, SP, L, L] in a 4-chef layout.
    - X is the number of unseen teammate.
    - X is assigned by the variable, unseen_teammates_len, in the funciton.
    '''
    unseen_teammates_len = 1
    primary_train_types = [
        TeamType.SELF_PLAY_HIGH,
        TeamType.SELF_PLAY_MEDIUM,
        TeamType.SELF_PLAY_LOW,
        # TeamType.SELF_PLAY_DYNAMIC_ADV, # TODO: read from command line arg
        # TeamType.SELF_PLAY_STATIC_ADV,
    ]
    primary_eval_types = {
        'generate': [
                    TeamType.SELF_PLAY_HIGH,
                     TeamType.SELF_PLAY_LOW,
                    #  TeamType.SELF_PLAY_DYNAMIC_ADV,
                    #  TeamType.SELF_PLAY_STATIC_ADV,
                    ],
        'load': []
    }
    if args.prioritized_sampling:
        curriculum = Curriculum(train_types=primary_train_types,
                                eval_types=primary_eval_types,
                                is_random=False,
                                prioritized_sampling=True,
                                priority_scaling=2.0)
    else:
        curriculum = Curriculum(train_types=primary_train_types, is_random=True)

    get_N_X_SP_agents(
        args,
        n_x_sp_train_types = curriculum.train_types,
        n_x_sp_eval_types=primary_eval_types,
        curriculum=curriculum,
        unseen_teammates_len=unseen_teammates_len,
    )


def best_EGO(args, add_adv=False) -> None:
    '''for a very specifric experimetn: only for 2 players:: ignore this'''
    primary_train_types = [
        TeamType.SELF_PLAY_HIGH,
        TeamType.SELF_PLAY_MEDIUM,
        TeamType.SELF_PLAY_LOW,
    ]
    primary_eval_types = {
        'generate': [
                    TeamType.SELF_PLAY_HIGH,
                    TeamType.SELF_PLAY_LOW,
                    ],
        'load': []
    }
    if args.prioritized_sampling:
        curriculum = Curriculum(train_types=primary_train_types, 
                                eval_types=primary_eval_types, 
                                is_random=False, 
                                prioritized_sampling=True,
                                priority_scaling=2.0)
    else:
        curriculum = Curriculum(train_types=primary_train_types, is_random=True)

    get_best_EGO_agents(
        args,
        curriculum=curriculum,
        primary_eval_types=primary_eval_types,
        primary_train_types=curriculum.train_types,
        add_adv=add_adv
    )


if __name__ == '__main__':
    args = get_arguments()

    if args.algo_name == 'SP':
        SP(args=args)

    elif args.algo_name == 'SPN_XSPCKP':
        SPN_XSPCKP(args=args)

    elif args.algo_name == 'MEP':
        MEP_POPULATION(args=args)

    elif args.algo_name == 'FCP_traditional':
        FCP_traditional(args=args)

    # elif args.algo_name == 'best_EGO':
    #     best_EGO(args=args, add_adv=False)

    # elif args.algo_name == 'FCP_mhri':
    #     FCP_mhri(args=args)

    # elif args.algo_name == 'SPN_1ADV':
    #     SPN_1ADV(args=args)

    # elif args.algo_name == 'N_1_FCP':
    #     N_1_FCP(args=args)

    # elif args.algo_name == 'SPN_1ADV_XSPCKP':
    #     SPN_1ADV_XSPCKP(args=args)

