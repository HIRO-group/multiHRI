from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.tags import TeamType
from oai_agents.common.population import get_performance_based_population_by_layouts, generate_hdim_and_seed
from oai_agents.common.teammates_collection import generate_TC, get_best_SP_agent, generate_TC_for_ADV_agent, update_TC_w_ADV_teammates, update_TC_w_dynamic_and_static_ADV_teammates
from oai_agents.common.curriculum import Curriculum
from oai_agents.common.heatmap import generate_adversaries_based_on_heatmap
from oai_agents.agents.agent_utils import CustomAgent
from .common import load_agents, generate_name
from oai_agents.common.tags import Prefix, KeyCheckpoints
from oai_agents.common.multi_setup_trainer import MultiSetupSPTrainer


def get_SP_agents(args, train_types, eval_types, curriculum, tag_for_returning_agent):
    sp_trainer = MultiSetupSPTrainer(
        args=args,
        train_types=train_types,
        eval_types=eval_types,
        curriculum=curriculum,
        tag_for_returning_agent=tag_for_returning_agent,
    )
    return sp_trainer.get_multiple_trained_agents()


def get_N_X_SP_agents(
        args,
        unseen_teammates_len:int,
        n_x_sp_train_types:list,
        n_x_sp_eval_types:dict,
        curriculum:Curriculum,
        tag:str=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL,
        attack_rounds:int=-1,
        adversary_play_config:str=None
    ):

    curriculum.validate_curriculum_types(
        expected_types = [
            TeamType.SELF_PLAY_HIGH,
            TeamType.SELF_PLAY_MEDIUM,
            TeamType.SELF_PLAY_LOW,
            TeamType.SELF_PLAY,
            TeamType.SELF_PLAY_ADVERSARY,
            TeamType.SELF_PLAY_STATIC_ADV,
            TeamType.SELF_PLAY_DYNAMIC_ADV
        ],
        unallowed_types = TeamType.ALL_TYPES_BESIDES_SP
    )


    if TeamType.SELF_PLAY_ADVERSARY in n_x_sp_train_types:
        prefix = 'PWADV' + '-N-' + str(unseen_teammates_len) + '-SP'
        suffix = args.primary_learner_type + f'_attack{attack_rounds-1}'
    else:
        prefix = 'N-' + str(unseen_teammates_len) + '-SP'
        suffix = args.primary_learner_type

    name = generate_name(
        args,
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

    population = get_performance_based_population_by_layouts(
        args=args,
        ck_rate=args.pop_total_training_timesteps // args.num_of_ckpoints,
        total_training_timesteps=args.pop_total_training_timesteps,
        train_types=n_x_sp_train_types,
        eval_types=n_x_sp_eval_types['generate'],
        unseen_teammates_len = unseen_teammates_len,
        total_ego_agents=args.total_ego_agents,
        force_training=args.pop_force_training,
        tag=tag
    )

    if TeamType.SELF_PLAY_ADVERSARY in n_x_sp_train_types:
        # Trains the adversaries
        train_ADV_and_N_X_SP(
            args=args,
            population=population,
            curriculum=curriculum,
            unseen_teammates_len=unseen_teammates_len,
            adversary_play_config=adversary_play_config,
            attack_rounds=attack_rounds,
            n_x_sp_eval_types=n_x_sp_eval_types
        )
    elif (TeamType.SELF_PLAY_STATIC_ADV in n_x_sp_train_types) or (TeamType.SELF_PLAY_DYNAMIC_ADV in n_x_sp_train_types):
        # Adversaries are not trained, are generated using a heatmap
        gen_ADV_train_N_X_SP(
            args=args,
            population=population,
            curriculum=curriculum,
            unseen_teammates_len=unseen_teammates_len,
            n_x_sp_eval_types=n_x_sp_eval_types
        )
    else:
        N_X_SP(
            args=args,
            population=population,
            curriculum=curriculum,
            unseen_teammates_len=unseen_teammates_len,
            n_x_sp_eval_types=n_x_sp_eval_types
        )


def gen_ADV_train_N_X_SP(args, population, curriculum, unseen_teammates_len, n_x_sp_eval_types, tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL):
    name = generate_name(args,
                        prefix = f'N-{unseen_teammates_len}-SP',
                        seed = args.N_X_SP_seed,
                        h_dim = args.N_X_SP_h_dim,
                        train_types = curriculum.train_types,
                        has_curriculum = not curriculum.is_random,
                        suffix=args.primary_learner_type + '_attack' + str(args.custom_agent_ck_rate_generation))

    agents = load_agents(args, name=name, tag=tag, force_training=args.primary_force_training)
    if agents:
        return agents[0]

    heatmap_source = get_best_SP_agent(args=args, population=population)
    init_agent = heatmap_source
    teammates_collection = generate_TC(args=args,
                                        population=population,
                                        agent=init_agent,
                                        train_types=curriculum.train_types,
                                        eval_types_to_generate=n_x_sp_eval_types['generate'],
                                        eval_types_to_read_from_file=n_x_sp_eval_types['load'],
                                        unseen_teammates_len=unseen_teammates_len,
                                        use_entire_population_for_train_types_teammates=True)

    adversaries = generate_adversaries_based_on_heatmap(args=args, heatmap_source=heatmap_source, current_adversaries={}, teammates_collection=teammates_collection, train_types=curriculum.train_types)

    total_train_timesteps = (args.n_x_sp_total_training_timesteps) // args.custom_agent_ck_rate_generation
    ck_rate = ((args.n_x_sp_total_training_timesteps + args.pop_total_training_timesteps) // args.num_of_ckpoints) // args.custom_agent_ck_rate_generation

    for round in range(args.custom_agent_ck_rate_generation):
        name = generate_name(args,
                        prefix = f'N-{unseen_teammates_len}-SP',
                        seed = args.N_X_SP_seed,
                        h_dim = args.N_X_SP_h_dim,
                        train_types = curriculum.train_types,
                        has_curriculum = not curriculum.is_random,
                        suffix=args.primary_learner_type + '_attack' + str(round))
        agents = load_agents(args, name=name, tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL, force_training=args.primary_force_training)
        if agents:
            init_agent = agents[0]
            continue

        teammates_collection = update_TC_w_dynamic_and_static_ADV_teammates(args=args,
                                                                            train_types=curriculum.train_types,
                                                                            eval_types=n_x_sp_eval_types['generate'],
                                                                            teammates_collection=teammates_collection,
                                                                            primary_agent=init_agent,
                                                                            adversaries=adversaries)
        init_agent.name = name
        n_x_sp_types_trainer = RLAgentTrainer(name=name,
                                                args=args,
                                                agent=init_agent,
                                                teammates_collection=teammates_collection,
                                                epoch_timesteps=args.epoch_timesteps,
                                                n_envs=args.n_envs,
                                                curriculum=curriculum,
                                                seed=args.N_X_SP_seed,
                                                hidden_dim=args.N_X_SP_h_dim,
                                                learner_type=args.primary_learner_type,
                                                checkpoint_rate= ck_rate * (round + 1),
                                                )

        n_x_sp_types_trainer.train_agents(total_train_timesteps = total_train_timesteps*(round + 1) + args.pop_total_training_timesteps,
                                                    tag_for_returning_agent=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL)
        init_agent = n_x_sp_types_trainer.agents[0]
        new_adversaries = generate_adversaries_based_on_heatmap(args=args, heatmap_source=init_agent, current_adversaries=adversaries, teammates_collection=teammates_collection, train_types=curriculum.train_types)
        adversaries = {key: adversaries.get(key, []) + new_adversaries.get(key, []) for key in set(adversaries) | set(new_adversaries)}
    return init_agent


def train_ADV_and_N_X_SP(args, population, curriculum, unseen_teammates_len, adversary_play_config, attack_rounds, n_x_sp_eval_types, tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL):
    assert TeamType.SELF_PLAY_ADVERSARY in curriculum.train_types

    agent_to_be_attacked = get_best_SP_agent(args=args, population=population)

    adversary_agents = []
    for attack_round in range(attack_rounds):
        adversary_agent = get_adversary_agent(
            args=args,
            agent_to_be_attacked=agent_to_be_attacked,
            attack_round=attack_round
        )
        adversary_agents.append(adversary_agent)

        name = generate_name(
            args,
            prefix = f'PWADV-N-{unseen_teammates_len}-SP',
            seed = args.N_X_SP_seed,
            h_dim = args.N_X_SP_h_dim,
            train_types = curriculum.train_types,
            has_curriculum = not curriculum.is_random,
            suffix=args.primary_learner_type + '_attack' + str(attack_round),
        )

        agents = load_agents(
            args,
            name=name,
            tag=tag,
            force_training=args.primary_force_training
        )
        if agents:
            agent_to_be_attacked = agents[0]
            continue

        random_init_agent = RLAgentTrainer.generate_randomly_initialized_agent(
            args=args,
            name=name,
            learner_type=args.primary_learner_type,
            hidden_dim=args.N_X_SP_h_dim,
            seed=args.N_X_SP_seed,
            n_envs=args.n_envs
        )

        teammates_collection = generate_TC(
            args=args,
            population=population,
            agent=random_init_agent,
            train_types=curriculum.train_types,
            eval_types_to_generate=n_x_sp_eval_types['generate'],
            eval_types_to_read_from_file=n_x_sp_eval_types['load'],
            unseen_teammates_len=unseen_teammates_len,
            use_entire_population_for_train_types_teammates=True
        )

        teammates_collection = update_TC_w_ADV_teammates(
            args=args,
            teammates_collection=teammates_collection,
            primary_agent=random_init_agent,
            adversaries=adversary_agents,
            adversary_play_config=adversary_play_config
        )

        if attack_round == attack_rounds-1:
            total_train_timesteps = 4*args.n_x_sp_total_training_timesteps
        else:
            total_train_timesteps = args.n_x_sp_total_training_timesteps

        n_x_sp_types_trainer = RLAgentTrainer(
            name=name,
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

        n_x_sp_types_trainer.train_agents(total_train_timesteps=total_train_timesteps, tag_for_returning_agent=tag)
        agent_to_be_attacked = n_x_sp_types_trainer.get_agents()[0]


def N_X_SP(args, population, curriculum, unseen_teammates_len, n_x_sp_eval_types, tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL):
    assert TeamType.SELF_PLAY_ADVERSARY not in curriculum.train_types

    name = generate_name(
        args,
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

    random_init_agent = RLAgentTrainer.generate_randomly_initialized_agent(
        args=args,
        name=name,
        learner_type=args.primary_learner_type,
        hidden_dim=args.N_X_SP_h_dim,
        seed=args.N_X_SP_seed,
        n_envs=args.n_envs
    )

    teammates_collection = generate_TC(
        args=args,
        population=population,
        agent=random_init_agent,
        train_types=curriculum.train_types,
        eval_types_to_generate=n_x_sp_eval_types['generate'],
        eval_types_to_read_from_file=n_x_sp_eval_types['load'],
        unseen_teammates_len=unseen_teammates_len,
        use_entire_population_for_train_types_teammates=True
    )

    n_x_sp_types_trainer = RLAgentTrainer(
        name=name,
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
    n_x_sp_types_trainer.train_agents(
        total_train_timesteps=args.n_x_sp_total_training_timesteps,
        tag_for_returning_agent=tag
    )



def get_adversary_agent(
        args,
        agent_to_be_attacked,
        attack_round,
        tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL
    ):
    # It doesn't matter what we set the variable, adversary_teammates_teamtype,
    # the purpose of it is to maintain consistent naming and correct TC/curriculum creation
    adversary_teammates_teamtype = TeamType.HIGH_FIRST

    teammates_collection = generate_TC_for_ADV_agent(
        args=args,
        agent_to_be_attacked=agent_to_be_attacked,
        teamtype=adversary_teammates_teamtype
    )

    name = generate_name(
        args,
        prefix='ADV',
        seed=args.ADV_seed,
        h_dim=args.ADV_h_dim,
        train_types=[adversary_teammates_teamtype],
        has_curriculum=False,
        suffix=args.adversary_learner_type +'_attack'+ str(attack_round)
    )

    agents = load_agents(
        args,
        name=name,
        tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL,
        force_training=args.adversary_force_training
    )
    if agents:
        return agents[0]

    adversary_trainer = RLAgentTrainer(
        name=name,
        args=args,
        agent=None,
        teammates_collection=teammates_collection,
        epoch_timesteps=args.epoch_timesteps,
        n_envs=args.n_envs,
        curriculum=Curriculum(train_types=[adversary_teammates_teamtype], is_random=True),
        seed=args.ADV_seed,
        hidden_dim=args.ADV_h_dim,
        learner_type=args.adversary_learner_type,
        checkpoint_rate=args.adversary_total_training_timesteps // args.num_of_ckpoints
    )
    adversary_trainer.train_agents(
        total_train_timesteps=args.adversary_total_training_timesteps,
        tag_for_returning_agent=tag
    )
    return adversary_trainer.get_agents()[0]


def get_FCP_agent_w_pop(
        args,
        fcp_train_types,
        fcp_eval_types,
        fcp_curriculum,
        tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL
    ):

    name = generate_name(
        args,
        prefix=Prefix.FICTITIOUS_CO_PLAY,
        seed=args.FCP_seed,
        h_dim=args.FCP_h_dim,
        train_types=fcp_train_types,
        has_curriculum = not fcp_curriculum.is_random
    )

    population = get_performance_based_population_by_layouts(
        args=args,
        ck_rate=args.pop_total_training_timesteps // args.num_of_ckpoints,
        total_training_timesteps=args.pop_total_training_timesteps,
        train_types=fcp_train_types,
        eval_types=fcp_eval_types['generate'],
        total_ego_agents=args.total_ego_agents,
        force_training=args.pop_force_training,
        tag=tag
    )

    teammates_collection = generate_TC(
        args=args,
        population=population,
        train_types=fcp_train_types,
        eval_types_to_generate=fcp_eval_types['generate'],
        eval_types_to_read_from_file=fcp_eval_types['load'],
        use_entire_population_for_train_types_teammates=False
    )

    agents = load_agents(
        args,
        name=name,
        tag=tag,
        force_training=args.primary_force_training
    )
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

    fcp_trainer.train_agents(
        total_train_timesteps=args.fcp_total_training_timesteps,
        tag_for_returning_agent=tag
    )
    return fcp_trainer.get_agents()[0], population



def get_N_X_FCP_agents(
        args,
        fcp_train_types,
        fcp_eval_types,
        n_1_fcp_train_types,
        n_1_fcp_eval_types,
        fcp_curriculum,
        n_1_fcp_curriculum,
        unseen_teammates_len,
        tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL
    ):

    n_1_fcp_curriculum.validate_curriculum_types(
        expected_types = [
            TeamType.SELF_PLAY_HIGH,
            TeamType.SELF_PLAY_MEDIUM,
            TeamType.SELF_PLAY_LOW
        ],
        unallowed_types= TeamType.ALL_TYPES_BESIDES_SP
    )

    name = generate_name(
        args,
        prefix=f'N-{unseen_teammates_len}-FCP',
        seed=args.N_X_FCP_seed,
        h_dim=args.N_X_FCP_h_dim,
        train_types=n_1_fcp_curriculum.train_types,
        has_curriculum = not fcp_curriculum.is_random
    )

    agents = load_agents(
        args,
        name=name,
        tag=tag,
        force_training=args.primary_force_training
    )
    if agents:
        return agents[0]

    fcp_agent, population = get_FCP_agent_w_pop(
        args,
        fcp_train_types=fcp_train_types,
        fcp_eval_types=fcp_eval_types,
        fcp_curriculum=fcp_curriculum
    )

    teammates_collection = generate_TC(
        args=args,
        population=population,
        agent=fcp_agent,
        train_types=n_1_fcp_train_types,
        eval_types_to_generate=n_1_fcp_eval_types['generate'],
        eval_types_to_read_from_file=n_1_fcp_eval_types['load'],
        unseen_teammates_len=unseen_teammates_len,
        use_entire_population_for_train_types_teammates=False
    )

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

    fcp_trainer.train_agents(
        total_train_timesteps=args.n_x_fcp_total_training_timesteps,
        tag_for_returning_agent=tag
    )
    return fcp_trainer.get_agents()[0], teammates_collection
