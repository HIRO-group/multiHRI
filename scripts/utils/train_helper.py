from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.tags import TeamType
from oai_agents.common.population import get_population
from oai_agents.common.teammates_collection import generate_TC, get_best_SP_agent, update_TC_w_adversary, generate_TC_for_Adversary, generate_TC_for_AdversarysPlay
from oai_agents.common.curriculum import Curriculum
from .common import load_agents, generate_name

from oai_agents.common.tags import CheckedPoints

from oai_agents.agents.agent_utils import load_agent
from pathlib import Path

def get_SP_agent(args, total_training_timesteps, train_types, eval_types, curriculum, tag=None, force_training=False):
    name = generate_name(args,
                         prefix='SP',
                         seed=args.SP_seed,
                         h_dim=args.SP_h_dim,
                         train_types=train_types,
                         has_curriculum= not curriculum.is_random)

    agents = load_agents(args, name=name, tag=tag, force_training=force_training)

    if agents:
        return agents[0]

    tc = generate_TC(args=args,
                    population={layout: [] for layout in args.layout_names},
                    train_types=curriculum.train_types,
                    eval_types_to_generate=eval_types['generate'],
                    eval_types_to_read_from_file=eval_types['load'])

    selfplay_trainer = RLAgentTrainer(
        name=name,
        args=args,
        agent=None,
        teammates_collection=tc,
        epoch_timesteps=args.epoch_timesteps,
        n_envs=args.n_envs,
        curriculum=curriculum,
        seed=args.SP_seed,
        hidden_dim=args.SP_h_dim,
    )

    selfplay_trainer.train_agents(total_train_timesteps=total_training_timesteps)
    return selfplay_trainer.get_agents()[0], tc


def get_N_X_SP_agents(args,
                        n_x_sp_train_types:list,
                        n_x_sp_eval_types:list,
                        curriculum:Curriculum,
                        tag:str=None) -> tuple:

    curriculum.validate_curriculum_types(expected_types = [TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_MEDIUM,
                                                           TeamType.SELF_PLAY_LOW,
                                                           TeamType.SELF_PLAY, TeamType.SELF_PLAY_ADVERSARY],
                                         unallowed_types = TeamType.ALL_TYPES_BESIDES_SP)


    if TeamType.SELF_PLAY_ADVERSARY in n_x_sp_train_types:
        prefix = 'PWADV' + '-N-' + str(args.unseen_teammates_len) + '-SP'
        suffix = args.primary_learner_type + f'_attack{args.attack_rounds-1}'
    else:
        prefix = 'N-' + str(args.unseen_teammates_len) + '-SP'
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

    population = get_population(
        args=args,
        ck_rate=args.pop_total_training_timesteps // 20,
        total_training_timesteps=args.pop_total_training_timesteps,
        train_types=n_x_sp_train_types,
        eval_types=n_x_sp_eval_types['generate'],
        unseen_teammates_len = args.unseen_teammates_len,
        num_SPs_to_train=args.num_SPs_to_train,
        force_training=args.pop_force_training,
        tag = tag
    )

    if TeamType.SELF_PLAY_ADVERSARY in n_x_sp_train_types:
        attack_N_X_SP(args=args,
                      population=population,
                      curriculum=curriculum)
    else:
        dont_attack_N_X_SP(args=args,
                           population=population,
                           curriculum=curriculum)


def attack_N_X_SP(args, population, curriculum):
    assert TeamType.SELF_PLAY_ADVERSARY in args.primary_train_types
    assert TeamType.SELF_PLAY_ADVERSARY in curriculum.train_types

    # agent_to_be_attacked = get_best_SP_agent(args=args, population=population)
    agent_to_be_attacked = RLAgentTrainer.load_agents(args, name='SP_hd64_seed14', tag='best')[0]

    adversary_agents = []
    for attack_round in range(args.attack_rounds):
        adversary_agent = get_adversary_agent(args=args,
                                              agent_to_be_attacked=agent_to_be_attacked,
                                              attack_round=attack_round)
        adversary_agents.append(adversary_agent)

        name = generate_name(args,
                            prefix = f'PWADV-N-{args.unseen_teammates_len}-SP',
                            seed = args.N_X_SP_seed,
                            h_dim = args.N_X_SP_h_dim,
                            train_types = args.primary_train_types,
                            has_curriculum = not curriculum.is_random,
                            suffix=args.primary_learner_type + '_attack' + str(attack_round),
                            )

        agents = load_agents(args, name=name, tag=CheckedPoints.FINAL_TRAINED_MODEL, force_training=args.adversary_force_training)
        print(f"name: {name}, agents: {agents}")
        if agents:
            agent_to_be_attacked = agents[0]
            continue

        random_init_agent = RLAgentTrainer.generate_randomly_initialized_agent(args=args, learner_type=args.primary_learner_type,
                                                                            hidden_dim=args.N_X_SP_h_dim, seed=args.N_X_SP_seed)

        teammates_collection = generate_TC(args=args,
                                        population=population,
                                        agent=random_init_agent,
                                        train_types=curriculum.train_types,
                                        eval_types_to_generate=args.primary_eval_types['generate'],
                                        eval_types_to_read_from_file=args.primary_eval_types['load'],
                                        unseen_teammates_len=args.unseen_teammates_len)

        teammates_collection = update_TC_w_adversary(args=args,
                                                        teammates_collection=teammates_collection,
                                                        primary_agent=random_init_agent,
                                                        adversaries=adversary_agents)

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
                                            fcp_ck_rate=args.n_x_sp_total_training_timesteps // 20,
                                            use_lstm=True,
                                            )
        n_x_sp_types_trainer.train_agents(total_train_timesteps=args.n_x_sp_total_training_timesteps)
        agent_to_be_attacked = n_x_sp_types_trainer.get_agents()[0]


def dont_attack_N_X_SP(args, population, curriculum):
    assert TeamType.SELF_PLAY_ADVERSARY not in args.primary_train_types
    assert TeamType.SELF_PLAY_ADVERSARY not in args.primary_eval_types
    assert TeamType.SELF_PLAY_ADVERSARY not in curriculum.train_types

    name = generate_name(args,
                         prefix = f'N-{args.unseen_teammates_len}-SP',
                         seed = args.N_X_SP_seed,
                         h_dim = args.N_X_SP_h_dim,
                         train_types = curriculum.train_types,
                         has_curriculum = not curriculum.is_random,
                         suffix=args.primary_learner_type,
                         )

    agents = load_agents(args, name=name, tag=CheckedPoints.FINAL_TRAINED_MODEL, force_training=args.primary_force_training)
    if agents:
        return agents[0]


    random_init_agent = RLAgentTrainer.generate_randomly_initialized_agent(args=args, h_dim=args.N_X_SP_h_dim, seed=args.N_X_SP_seed)

    teammates_collection = generate_TC(args=args,
                                        population=population,
                                        agent=random_init_agent,
                                        train_types=curriculum.train_types,
                                        eval_types_to_generate=args.primary_eval_types['generate'],
                                        eval_types_to_read_from_file=args.primary_eval_types['load'],
                                        unseen_teammates_len=args.unseen_teammates_len)

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
                                        fcp_ck_rate=args.n_x_sp_total_training_timesteps // 20,
                                        )
    n_x_sp_types_trainer.train_agents(total_train_timesteps=args.n_x_sp_total_training_timesteps)



def get_adversary_agent(args, agent_to_be_attacked, attack_round, tag=None):
    teammates_collection = generate_TC_for_Adversary(args=args,
                                                    agent=agent_to_be_attacked)

    name = generate_name(args,
                        prefix='ADV',
                        seed=args.ADV_seed,
                        h_dim=args.ADV_h_dim,
                        train_types=[TeamType.HIGH_FIRST],
                        has_curriculum=False,
                        suffix=args.adversary_learner_type +'_attack'+ str(attack_round))

    agents = load_agents(args, name=name, tag=tag, force_training=False)
    if agents:
        return agents[0]

    adversary_trainer = RLAgentTrainer(name=name,
                                        args=args,
                                        agent=None,
                                        teammates_collection=teammates_collection,
                                        epoch_timesteps=args.epoch_timesteps,
                                        n_envs=args.n_envs,
                                        curriculum=Curriculum(train_types=[TeamType.HIGH_FIRST], is_random=True),
                                        seed=args.ADV_seed,
                                        hidden_dim=args.ADV_h_dim,
                                        learner_type=args.adversary_learner_type,
                                        fcp_ck_rate=args.adversary_total_training_timesteps // 20)
    adversary_trainer.train_agents(total_train_timesteps=args.adversary_total_training_timesteps)
    return adversary_trainer.get_agents()[0]



def get_FCP_agent_w_pop(args,
                        pop_total_training_timesteps,
                        fcp_total_training_timesteps,
                        fcp_train_types,
                        fcp_eval_types,
                        fcp_curriculum,
                        pop_force_training,
                        primary_force_training,
                        num_SPs_to_train=2,
                        tag=None):

    name = generate_name(args,
                         prefix='FCP',
                         seed=args.FCP_seed,
                         h_dim=args.FCP_h_dim,
                         train_types=fcp_train_types,
                         has_curriculum = not fcp_curriculum.is_random)

    population = get_population(
        args=args,
        ck_rate=pop_total_training_timesteps // 20,
        total_training_timesteps=pop_total_training_timesteps,
        train_types=fcp_train_types,
        eval_types=fcp_eval_types['generate'],
        num_SPs_to_train=num_SPs_to_train,
        force_training=pop_force_training,
        tag = tag
    )

    teammates_collection = generate_TC(args=args,
                                        population=population,
                                        train_types=fcp_train_types,
                                        eval_types_to_generate=fcp_eval_types['generate'],
                                        eval_types_to_read_from_file=fcp_eval_types['load'])

    agents = load_agents(args, name=name, tag=tag, force_training=primary_force_training)
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
    )

    fcp_trainer.train_agents(total_train_timesteps=fcp_total_training_timesteps)
    return fcp_trainer.get_agents()[0], population



def get_N_X_FCP_agents(args,
                        pop_total_training_timesteps,
                        fcp_total_training_timesteps,
                        n_x_fcp_total_training_timesteps,

                        pop_force_training,
                        fcp_force_training,
                        primary_force_training,

                        fcp_train_types,
                        fcp_eval_types,
                        n_1_fcp_train_types,
                        n_1_fcp_eval_types,

                        fcp_curriculum,
                        n_1_fcp_curriculum,
                        num_SPs_to_train=2,
                        tag=None):

    n_1_fcp_curriculum.validate_curriculum_types(expected_types = [TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_LOW],
                                                  unallowed_types= TeamType.ALL_TYPES_BESIDES_SP)

    name = generate_name(args,
                         prefix=f'N-{args.unseen_teammates_len}-FCP',
                         seed=args.N_X_FCP_seed,
                         h_dim=args.N_X_FCP_h_dim,
                         train_types=n_1_fcp_curriculum.train_types,
                         has_curriculum = not fcp_curriculum.is_random)

    agents = load_agents(args, name=name, tag=tag, force_training=primary_force_training)
    if agents:
        return agents[0]

    fcp_agent, population = get_FCP_agent_w_pop(args,
                                                pop_total_training_timesteps=pop_total_training_timesteps,
                                                fcp_total_training_timesteps=fcp_total_training_timesteps,
                                                fcp_train_types=fcp_train_types,
                                                fcp_eval_types=fcp_eval_types,
                                                pop_force_training=pop_force_training,
                                                primary_force_training=fcp_force_training,
                                                num_SPs_to_train=num_SPs_to_train,
                                                fcp_curriculum=fcp_curriculum,
                                                 )

    teammates_collection = generate_TC(args=args,
                                        population=population,
                                        agent=fcp_agent,
                                        train_types=n_1_fcp_train_types,
                                        eval_types_to_generate=n_1_fcp_eval_types['generate'],
                                        eval_types_to_read_from_file=n_1_fcp_eval_types['load'],
                                        unseen_teammates_len=args.unseen_teammates_len)

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
    )

    fcp_trainer.train_agents(total_train_timesteps=n_x_fcp_total_training_timesteps)
    return fcp_trainer.get_agents()[0], teammates_collection

def get_adversary(args, total_training_timesteps, train_types, eval_types, curriculum, agent_path):
    name = generate_name(args,
                         prefix='adv',
                         seed=args.ADV_seed,
                         h_dim=args.ADV_h_dim,
                         train_types=train_types,
                         has_curriculum= not curriculum.is_random)

    adversary = load_agents(args, name=name, tag=CheckedPoints.FINAL_TRAINED_MODEL, force_training=False)

    tc = generate_TC_for_Adversary(args,
                                  agent=agent,
                                  train_types=train_types,
                                  eval_types_to_generate=eval_types['generate'],
                                  eval_types_to_read_from_file=eval_types['load'])

    if adversary:
        return adversary, tc, name

    adversary_trainer = RLAgentTrainer(
        name=name,
        args=args,
        agent=None,
        teammates_collection=tc,
        epoch_timesteps=args.epoch_timesteps,
        n_envs=args.n_envs,
        curriculum=curriculum,
        seed=args.ADV_seed,
        hidden_dim=args.ADV_h_dim,
        fcp_ck_rate=total_training_timesteps // 20,
    )

    adversary_trainer.train_agents(total_train_timesteps=total_training_timesteps)
    return adversary_trainer.get_agents()[0], tc, name


def get_agent_play_w_adversarys(args, train_types, eval_types, total_training_timesteps, curriculum, agent_path, adv_paths, check_whether_exist):
    name = generate_name(args,
                         prefix='pwadv',
                         seed=args.PwADV_seed,
                         h_dim=args.PwADV_h_dim,
                         train_types=train_types,
                         has_curriculum= not curriculum.is_random)
    latest_agent = load_agents(args, name=name, tag=CheckedPoints.FINAL_TRAINED_MODEL, force_training=False)
    agent = load_agent(Path(agent_path), args)
    adversarys = [load_agent(Path(adv_path), args) for adv_path in adv_paths]

    tc = generate_TC_for_AdversarysPlay(args,
                                  agent=agent,
                                  adversarys=adversarys,
                                  train_types=train_types,
                                  eval_types_to_generate=eval_types['generate'],
                                  eval_types_to_read_from_file=eval_types['load'])
    if latest_agent and check_whether_exist:
        return latest_agent, tc, name

    agent_trainer = RLAgentTrainer(
        name=name,
        args=args,
        agent=agent,
        teammates_collection=tc,
        epoch_timesteps=args.epoch_timesteps,
        n_envs=args.n_envs,
        curriculum=curriculum,
        seed=args.PwADV_seed,
        hidden_dim=args.PwADV_h_dim,
        fcp_ck_rate=total_training_timesteps // 20,
    )

    agent_trainer.train_agents(total_train_timesteps=total_training_timesteps)
    return agent_trainer.get_agents()[0], tc, name

def get_randomly_initialized_agent(args, n_env=200, h_dim=256, seed=13):
    return RLAgentTrainer.generate_randomly_initialized_agent(args=args, n_env=n_env, h_dim=h_dim, seed=seed)