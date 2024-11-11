from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.tags import TeamType
from oai_agents.common.population import get_categorized_population, generate_hdim_and_seed
from oai_agents.common.teammates_collection import generate_TC, get_best_SP_agent, generate_TC_for_ADV_agent, update_TC_w_ADV_teammates
from oai_agents.common.curriculum import Curriculum
from .common import load_agents, generate_name
from oai_agents.common.tags import Prefix
from oai_agents.common.tags import KeyCheckpoints


def get_SP_agent(args, train_types, eval_types, curriculum, tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL):
    name = generate_name(args,
                         prefix=Prefix.SELF_PLAY,
                         seed=args.SP_seed,
                         h_dim=args.SP_h_dim,
                         train_types=train_types,
                         has_curriculum= not curriculum.is_random)

    agents = load_agents(args, name=name, tag=tag, force_training=args.pop_force_training)
    if agents:
        return agents[0]

    selfplay_trainer = RLAgentTrainer(
        name=name,
        args=args,
        agent=None,
        teammates_collection={},
        epoch_timesteps=args.epoch_timesteps,
        n_envs=args.n_envs,
        curriculum=curriculum,
        seed=args.SP_seed,
        hidden_dim=args.SP_h_dim,
        learner_type=args.primary_learner_type,
        checkpoint_rate=args.pop_total_training_timesteps // args.num_of_ckpoints,
    )

    selfplay_trainer.train_agents(total_train_timesteps=args.pop_total_training_timesteps, tag=tag)
    return selfplay_trainer.get_agents()[0]


def get_N_X_SP_agents(args,
                        unseen_teammates_len:int,
                        n_x_sp_train_types:list,
                        n_x_sp_eval_types:list,
                        curriculum:Curriculum,
                        tag:str=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL,
                        attack_rounds:int=-1,
                        adversary_play_config:str=None) -> tuple:

    curriculum.validate_curriculum_types(expected_types = [TeamType.SELF_PLAY_HIGH,
                                                           TeamType.SELF_PLAY_MEDIUM,
                                                           TeamType.SELF_PLAY_LOW,
                                                           TeamType.SELF_PLAY,
                                                           TeamType.SELF_PLAY_ADVERSARY],
                                         unallowed_types = TeamType.ALL_TYPES_BESIDES_SP)


    if TeamType.SELF_PLAY_ADVERSARY in n_x_sp_train_types:
        prefix = 'PWADV' + '-N-' + str(unseen_teammates_len) + '-SP'
        suffix = args.primary_learner_type + f'_attack{attack_rounds-1}'
    else:
        prefix = 'N-' + str(unseen_teammates_len) + '-SP'
        suffix = args.primary_learner_type

    name = generate_name(args,
                         prefix = prefix,
                         seed = args.N_X_SP_seed,
                         h_dim = args.N_X_SP_h_dim,
                         train_types = n_x_sp_train_types,
                         has_curriculum = not curriculum.is_random,
                         suffix=suffix,
                         )
    agents = load_agents(args, name=name, tag=tag, force_training=args.primary_force_training)
    if agents:
        return agents[0]

    population = get_categorized_population(
        args=args,
        ck_rate=args.pop_total_training_timesteps // args.num_of_ckpoints,
        total_training_timesteps=args.pop_total_training_timesteps,
        train_types=n_x_sp_train_types,
        eval_types=n_x_sp_eval_types['generate'],
        unseen_teammates_len = unseen_teammates_len,
        num_SPs_to_train=args.num_SPs_to_train,
        force_training=args.pop_force_training,
        tag=tag
    )


    if TeamType.SELF_PLAY_ADVERSARY in n_x_sp_train_types:
        joint_ADV_N_X_SP(args=args,
                      population=population,
                      curriculum=curriculum,
                      unseen_teammates_len=unseen_teammates_len,
                      adversary_play_config=adversary_play_config,
                      attack_rounds=attack_rounds,
                      n_x_sp_eval_types=n_x_sp_eval_types
                      )
    else:
        no_ADV_N_X_SP(args=args,
                           population=population,
                           curriculum=curriculum,
                           unseen_teammates_len=unseen_teammates_len,
                           n_x_sp_eval_types=n_x_sp_eval_types
                           )


def joint_ADV_N_X_SP(args, population, curriculum, unseen_teammates_len, adversary_play_config, attack_rounds, n_x_sp_eval_types, tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL):
    assert TeamType.SELF_PLAY_ADVERSARY in curriculum.train_types

    agent_to_be_attacked = get_best_SP_agent(args=args, population=population)

    adversary_agents = []
    for attack_round in range(attack_rounds):
        adversary_agent = get_adversary_agent(args=args,
                                              agent_to_be_attacked=agent_to_be_attacked,
                                              attack_round=attack_round)
        adversary_agents.append(adversary_agent)

        name = generate_name(args,
                            prefix = f'PWADV-N-{unseen_teammates_len}-SP',
                            seed = args.N_X_SP_seed,
                            h_dim = args.N_X_SP_h_dim,
                            train_types = curriculum.train_types,
                            has_curriculum = not curriculum.is_random,
                            suffix=args.primary_learner_type + '_attack' + str(attack_round),
                            )

        agents = load_agents(args, name=name, tag=tag, force_training=args.primary_force_training)
        if agents:
            agent_to_be_attacked = agents[0]
            continue

        random_init_agent = RLAgentTrainer.generate_randomly_initialized_agent(args=args,
                                                                               name=name,
                                                                               learner_type=args.primary_learner_type,
                                                                               hidden_dim=args.N_X_SP_h_dim,
                                                                               seed=args.N_X_SP_seed)

        teammates_collection = generate_TC(args=args,
                                            population=population,
                                            agent=random_init_agent,
                                            train_types=curriculum.train_types,
                                            eval_types_to_generate=n_x_sp_eval_types['generate'],
                                            eval_types_to_read_from_file=n_x_sp_eval_types['load'],
                                            unseen_teammates_len=unseen_teammates_len)

        teammates_collection = update_TC_w_ADV_teammates(args=args,
                                                    teammates_collection=teammates_collection,
                                                    primary_agent=random_init_agent,
                                                    adversaries=adversary_agents,
                                                    adversary_play_config=adversary_play_config)

        if attack_round == attack_rounds-1:
            total_train_timesteps = 4*args.n_x_sp_total_training_timesteps
        else:
            total_train_timesteps = args.n_x_sp_total_training_timesteps

        n_x_sp_types_trainer = RLAgentTrainer(name=name,
                                            args=args,
                                            agent=random_init_agent,
                                            teammates_collection=teammates_collection,
                                            epoch_timesteps=args.epoch_timesteps,
                                            n_envs=args.n_envs,
                                            curriculum=curriculum,
                                            seed=args.N_X_SP_seed,
                                            hidden_dim=args.N_X_SP_h_dim,
                                            learner_type=args.primary_learner_type,
                                            checkpoint_rate=total_train_timesteps // args.num_of_ckpoints,
                                            )

        n_x_sp_types_trainer.train_agents(total_train_timesteps=total_train_timesteps, tag=tag)
        agent_to_be_attacked = n_x_sp_types_trainer.get_agents()[0]


def no_ADV_N_X_SP(args, population, curriculum, unseen_teammates_len, n_x_sp_eval_types, tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL):
    assert TeamType.SELF_PLAY_ADVERSARY not in curriculum.train_types

    name = generate_name(args,
                         prefix = f'N-{unseen_teammates_len}-SP',
                         seed = args.N_X_SP_seed,
                         h_dim = args.N_X_SP_h_dim,
                         train_types = curriculum.train_types,
                         has_curriculum = not curriculum.is_random,
                         suffix=args.primary_learner_type,
                        )

    agents = load_agents(args, name=name, tag=tag, force_training=args.primary_force_training)
    if agents:
        return agents[0]

    random_init_agent = RLAgentTrainer.generate_randomly_initialized_agent(args=args,
                                                                           name=name,
                                                                           learner_type=args.primary_learner_type,
                                                                           hidden_dim=args.N_X_SP_h_dim,
                                                                           seed=args.N_X_SP_seed)

    teammates_collection = generate_TC(args=args,
                                        population=population,
                                        agent=random_init_agent,
                                        train_types=curriculum.train_types,
                                        eval_types_to_generate=n_x_sp_eval_types['generate'],
                                        eval_types_to_read_from_file=n_x_sp_eval_types['load'],
                                        unseen_teammates_len=unseen_teammates_len)

    n_x_sp_types_trainer = RLAgentTrainer(name=name,
                                        args=args,
                                        agent=random_init_agent,
                                        teammates_collection=teammates_collection,
                                        epoch_timesteps=args.epoch_timesteps,
                                        n_envs=args.n_envs,
                                        curriculum=curriculum,
                                        seed=args.N_X_SP_seed,
                                        hidden_dim=args.N_X_SP_h_dim,
                                        learner_type=args.primary_learner_type,
                                        checkpoint_rate=args.n_x_sp_total_training_timesteps // args.num_of_ckpoints,
                                        )
    n_x_sp_types_trainer.train_agents(total_train_timesteps=args.n_x_sp_total_training_timesteps, tag=tag)



def get_adversary_agent(args, agent_to_be_attacked, attack_round, tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL):
    # It doesn't matter what we set the variable, adversary_teammates_teamtype,
    # the purpose of it is to maintain consistent naming and correct TC/curriculum creation
    adversary_teammates_teamtype = TeamType.HIGH_FIRST

    teammates_collection = generate_TC_for_ADV_agent(args=args,
                                                    agent_to_be_attacked=agent_to_be_attacked,
                                                    teamtype=adversary_teammates_teamtype)

    name = generate_name(args,
                        prefix='ADV',
                        seed=args.ADV_seed,
                        h_dim=args.ADV_h_dim,
                        train_types=[adversary_teammates_teamtype],
                        has_curriculum=False,
                        suffix=args.adversary_learner_type +'_attack'+ str(attack_round))

    agents = load_agents(args, name=name, tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL, force_training=args.adversary_force_training)
    if agents:
        return agents[0]

    adversary_trainer = RLAgentTrainer(name=name,
                                        args=args,
                                        agent=None,
                                        teammates_collection=teammates_collection,
                                        epoch_timesteps=args.epoch_timesteps,
                                        n_envs=args.n_envs,
                                        curriculum=Curriculum(train_types=[adversary_teammates_teamtype], is_random=True),
                                        seed=args.ADV_seed,
                                        hidden_dim=args.ADV_h_dim,
                                        learner_type=args.adversary_learner_type,
                                        checkpoint_rate=args.adversary_total_training_timesteps // args.num_of_ckpoints)
    adversary_trainer.train_agents(total_train_timesteps=args.adversary_total_training_timesteps, tag=tag)
    return adversary_trainer.get_agents()[0]


def get_FCP_agent_w_pop(args,
                        fcp_train_types,
                        fcp_eval_types,
                        fcp_curriculum,
                        tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL):

    name = generate_name(args,
                         prefix=Prefix.FICTITIOUS_CO_PLAY,
                         seed=args.FCP_seed,
                         h_dim=args.FCP_h_dim,
                         train_types=fcp_train_types,
                         has_curriculum = not fcp_curriculum.is_random)

    population = get_categorized_population(
        args=args,
        ck_rate=args.pop_total_training_timesteps // args.num_of_ckpoints,
        total_training_timesteps=args.pop_total_training_timesteps,
        train_types=fcp_train_types,
        eval_types=fcp_eval_types['generate'],
        num_SPs_to_train=args.num_SPs_to_train,
        force_training=args.pop_force_training,
        tag=tag
    )

    teammates_collection = generate_TC(args=args,
                                        population=population,
                                        train_types=fcp_train_types,
                                        eval_types_to_generate=fcp_eval_types['generate'],
                                        eval_types_to_read_from_file=fcp_eval_types['load'])

    agents = load_agents(args, name=name, tag=tag, force_training=args.primary_force_training)
    if agents:
        return agents[0], population

    fcp_trainer = RLAgentTrainer(
        name=name,
        args=args,
        agent=None,
        teammates_collection=teammates_collection,
        epoch_timesteps=args.epoch_timesteps,
        n_envs=args.n_envs,
        seed=args.FCP_seed,
        hidden_dim=args.FCP_h_dim,
        curriculum=fcp_curriculum,
        learner_type=args.primary_learner_type,
        checkpoint_rate=args.fcp_total_training_timesteps // args.num_of_ckpoints,
    )

    fcp_trainer.train_agents(total_train_timesteps=args.fcp_total_training_timesteps, tag=tag)
    return fcp_trainer.get_agents()[0], population



def get_N_X_FCP_agents(args,
                        fcp_train_types,
                        fcp_eval_types,
                        n_1_fcp_train_types,
                        n_1_fcp_eval_types,
                        fcp_curriculum,
                        n_1_fcp_curriculum,
                        unseen_teammates_len,
                        tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL):

    n_1_fcp_curriculum.validate_curriculum_types(expected_types = [TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_LOW],
                                                  unallowed_types= TeamType.ALL_TYPES_BESIDES_SP)

    name = generate_name(args,
                         prefix=f'N-{unseen_teammates_len}-FCP',
                         seed=args.N_X_FCP_seed,
                         h_dim=args.N_X_FCP_h_dim,
                         train_types=n_1_fcp_curriculum.train_types,
                         has_curriculum = not fcp_curriculum.is_random)

    agents = load_agents(args, name=name, tag=tag, force_training=args.primary_force_training)
    if agents:
        return agents[0]

    fcp_agent, population = get_FCP_agent_w_pop(args,
                                                fcp_train_types=fcp_train_types,
                                                fcp_eval_types=fcp_eval_types,
                                                fcp_curriculum=fcp_curriculum)

    teammates_collection = generate_TC(args=args,
                                        population=population,
                                        agent=fcp_agent,
                                        train_types=n_1_fcp_train_types,
                                        eval_types_to_generate=n_1_fcp_eval_types['generate'],
                                        eval_types_to_read_from_file=n_1_fcp_eval_types['load'],
                                        unseen_teammates_len=unseen_teammates_len)

    fcp_trainer = RLAgentTrainer(
        name=name,
        args=args,
        agent=fcp_agent,
        teammates_collection=teammates_collection,
        epoch_timesteps=args.epoch_timesteps,
        n_envs=args.n_envs,
        seed=args.N_X_FCP_seed,
        hidden_dim=args.N_X_FCP_h_dim,
        curriculum=n_1_fcp_curriculum,
        learner_type=args.primary_learner_type,
        checkpoint_rate=args.n_x_fcp_total_training_timesteps // args.num_of_ckpoints,
    )

    fcp_trainer.train_agents(total_train_timesteps=args.n_x_fcp_total_training_timesteps, tag=tag)
    return fcp_trainer.get_agents()[0], teammates_collection
