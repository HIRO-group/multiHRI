import multiprocessing as mp
mp.set_start_method('spawn', force=True) # should be called before any other module imports

from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.arguments import get_arguments
from oai_agents.common.population_tags import TeamType, TeammatesCollection

from scripts.utils import get_fcp_population, update_tms_clction_with_selfplay_types, load_agents, print_teammates_collection
from scripts.train_agents import get_input


def train_FCP(args, name, teammates_collection, train_types):
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

    fcp_trainer.train_agents(total_train_timesteps=fcp_total_training_timesteps)


if __name__ == "__main__":
    args = get_arguments()
    quick_test = True
    parallel = False
    pop_force_training = False
    fcp_force_training = True
    
    pop_total_training_timesteps, fcp_total_training_timesteps, fcp_w_sp_total_training_timesteps = get_input(args=args,
                                                                                                              quick_test=quick_test)
    
    save_path_prefix = f'eval/{args.teammates_len}_chefs'

    num_SP_agents_to_train = 1
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
                                              ck_rate=pop_total_training_timesteps // 5,
                                              train_types = TeamType.ALL_TYPES_BESIDES_SP,
                                              eval_types_to_generate = [],
                                              eval_types_to_load_from_file = [],
                                              num_self_play_agents_to_train=num_SP_agents_to_train,
                                              total_training_timesteps = pop_total_training_timesteps,
                                              force_training=pop_force_training,
                                              parallel=parallel,
                                              save_path_prefix = save_path_prefix,
                                              )
    # print_teammates_collection(teammates_collection[TeammatesCollection.TRAIN])

    for fcp_train_types in all_FCP_train_types:
        vb = '_'.join(fcp_train_types)
        train_FCP(args=args,
                  name=f"{save_path_prefix}/fcp_{vb}" if save_path_prefix else f'fcp_{vb}',
                  teammates_collection=teammates_collection,
                  train_types=fcp_train_types,
                  )
    


