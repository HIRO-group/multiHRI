import multiprocessing as mp
mp.set_start_method('spawn', force=True) # should be called before any other module imports

from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.arguments import get_arguments
from oai_agents.common.tags import TeamType, TeammatesCollection
from scripts.utils import get_fcp_population


def train_FCP(args, name, teammates_collection, train_types, total_training_timesteps):
    fcp_trainer = RLAgentTrainer(
        name=name,
        args=args,
        agent=None,
        teammates_collection=teammates_collection,
        epoch_timesteps=args.epoch_timesteps,
        n_envs=args.n_envs,
        train_types=train_types,
        seed=2602,
    )
    fcp_trainer.train_agents(total_train_timesteps=total_training_timesteps)


def set_input(args, quick_test=False):
    args.layout_names = ['3_chefs_small_kitchen']
    args.teammates_len = 2
    args.num_players = args.teammates_len + 1  # 3 players = 1 agent + 2 teammates
    args.exp_dir = f'eval/{args.teammates_len+1}_chefs'
    
    if not quick_test: 
        args.n_envs = 50
        args.epoch_timesteps = 1e5
        args.pop_total_training_timesteps = 5e6
        args.fcp_total_training_timesteps = 5e6
        args.num_SPs_to_train = 5

    else: # Used for doing quick tests
        args.sb_verbose = 1
        args.wandb_mode = 'disabled'
        args.n_envs = 2
        args.epoch_timesteps = 2
        args.pop_total_training_timesteps = 3500
        args.fcp_total_training_timesteps = 3500
        args.num_SPs_to_train = 4


if __name__ == "__main__":
    args = get_arguments()
    quick_test = False
    parallel = True
    pop_force_training = True
    primary_force_training = True
    set_input(args=args, quick_test=quick_test)

    all_FCP_train_types = [
        [TeamType.HIGH_FIRST],
        [TeamType.HIGH_FIRST, TeamType.MIDDLE_FIRST],
        [TeamType.HIGH_FIRST, TeamType.LOW_FIRST],
        [TeamType.HIGH_FIRST, TeamType.MIDDLE_FIRST, TeamType.LOW_FIRST],
        [TeamType.HIGH_LOW],
        [TeamType.HIGH_MEDIUM],
        [TeamType.MEDIUM_LOW],
        [TeamType.HIGH_LOW, TeamType.HIGH_MEDIUM],
        [TeamType.RANDOM],
    ]

    teammates_collection = get_fcp_population(args,
                                              ck_rate=args.pop_total_training_timesteps // 5,
                                              train_types = TeamType.ALL_TYPES_BESIDES_SP,
                                              eval_types_to_generate = [],
                                              eval_types_to_load_from_file = [],
                                              num_SPs_to_train=args.num_SPs_to_train,
                                              total_training_timesteps = args.pop_total_training_timesteps,
                                              force_training=pop_force_training,
                                              parallel=parallel,
                                              )

    teammates_collection[TeammatesCollection.EVAL] = teammates_collection[TeammatesCollection.TRAIN]
    
    # TODO: run this in parallel
    for fcp_train_types in all_FCP_train_types:
        vb = '_'.join(fcp_train_types)
        train_FCP(args=args,
                  name='fcp_{vb}',
                  teammates_collection=teammates_collection,
                  train_types=fcp_train_types,
                  total_training_timesteps=args.fcp_total_training_timesteps,
                  )
