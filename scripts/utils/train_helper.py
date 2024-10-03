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
                        pop_total_training_timesteps:int,
                        pop_force_training:bool,
                        n_x_sp_train_types:list,
                        n_x_sp_eval_types:list,
                        n_x_sp_force_training:list,
                        n_x_sp_total_training_timesteps:int,
                        curriculum:Curriculum,
                        tag:str=None,
                        num_SPs_to_train=2) -> tuple:

    curriculum.validate_curriculum_types(expected_types = [TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_MEDIUM,
                                                           TeamType.SELF_PLAY_LOW,
                                                           TeamType.SELF_PLAY, TeamType.SELF_PLAY_ADVERSARY],
                                         unallowed_types = TeamType.ALL_TYPES_BESIDES_SP)

    name = generate_name(args,
                         prefix = f'N-{args.unseen_teammates_len}-SP',
                         seed = args.N_X_SP_seed,
                         h_dim = args.N_X_SP_h_dim,
                         train_types = n_x_sp_train_types,
                         has_curriculum = not curriculum.is_random,
                         has_adversary = TeamType.SELF_PLAY_ADVERSARY in n_x_sp_train_types,
                         suffix=args.primary_learner_type,
                         )

    agents = load_agents(args, name=name, tag=tag, force_training=n_x_sp_force_training)
    if agents:
        return agents[0]

    population = get_population(
        args=args,
        ck_rate=pop_total_training_timesteps // 20,
        total_training_timesteps=pop_total_training_timesteps,
        train_types=n_x_sp_train_types,
        eval_types=n_x_sp_eval_types['generate'],
        unseen_teammates_len = args.unseen_teammates_len,
        num_SPs_to_train=num_SPs_to_train,
        force_training=pop_force_training,
        tag = tag
    )

    best_SP_agent = get_best_SP_agent(args, population)

    teammates_collection = generate_TC(args=args,
                                        population=population,
                                        agent=best_SP_agent,
                                        train_types=n_x_sp_train_types,
                                        eval_types_to_generate=n_x_sp_eval_types['generate'],
                                        eval_types_to_read_from_file=n_x_sp_eval_types['load'],
                                        unseen_teammates_len=args.unseen_teammates_len)
    
    if TeamType.SELF_PLAY_ADVERSARY in n_x_sp_train_types:
        attack_N_X_SP(args=args,
                        init_agent=best_SP_agent,
                        teammates_collection=teammates_collection,
                        curriculum=curriculum,
                        n_x_sp_total_training_timesteps=n_x_sp_total_training_timesteps)
    else:
        dont_attack_N_X_SP(args=args,
                         init_agent=best_SP_agent,
                         teammates_collection=teammates_collection,
                         curriculum=curriculum,
                          n_x_sp_total_training_timesteps=n_x_sp_total_training_timesteps)


def attack_N_X_SP(args, init_agent, teammates_collection, curriculum, n_x_sp_total_training_timesteps):
    assert TeamType.SELF_PLAY_ADVERSARY in args.primary_train_types
    assert TeamType.SELF_PLAY_ADVERSARY in curriculum.train_types

    primary_agent = init_agent
    adversary_agents = []
    for attack_round in range(args.attack_rounds):
        adversary_agent = get_adversary_agent(args=args,
                                              primary_agent=primary_agent,
                                              attack_round=attack_round)
        adversary_agents.append(adversary_agent)

        teammates_collection = update_TC_w_adversary(args=args,
                                                        teammates_collection=teammates_collection,
                                                        primary_agent=primary_agent,
                                                        adversaries=adversary_agents)

        name = generate_name(args,
                            prefix = f'PWADV-N-{args.unseen_teammates_len}-SP',
                            seed = args.N_X_SP_seed,
                            h_dim = args.N_X_SP_h_dim,
                            train_types = args.primary_train_types,
                            has_curriculum = not curriculum.is_random,
                            has_adversary = True,
                            suffix=args.primary_learner_type + '_attack' + str(attack_round),
                            )

        agent = load_agents(args, name=name, tag=CheckedPoints.FINAL_TRAINED_MODEL, force_training=args.adversary_force_training)
        if agent:
            primary_agent = agent[0]
            continue

        n_x_sp_types_trainer = RLAgentTrainer(name=name,
                                            args=args,
                                            agent=primary_agent,
                                            teammates_collection=teammates_collection,
                                            epoch_timesteps=args.epoch_timesteps,
                                            n_envs=args.n_envs,
                                            curriculum=curriculum,
                                            seed=args.N_X_SP_seed,
                                            hidden_dim=args.N_X_SP_h_dim,
                                            learner_type=args.primary_learner_type)
        n_x_sp_types_trainer.train_agents(total_train_timesteps=n_x_sp_total_training_timesteps)
        primary_agent = n_x_sp_types_trainer.get_agents()[0]


def dont_attack_N_X_SP(args, init_agent, teammates_collection, curriculum, n_x_sp_train_types, n_x_sp_total_training_timesteps, primary_force_training):
    assert TeamType.SELF_PLAY_ADVERSARY not in args.primary_train_types
    assert TeamType.SELF_PLAY_ADVERSARY not in args.primary_eval_types
    assert TeamType.SELF_PLAY_ADVERSARY not in curriculum.train_types
    
    name = generate_name(args,
                         prefix = f'N-{args.unseen_teammates_len}-SP',
                         seed = args.N_X_SP_seed,
                         h_dim = args.N_X_SP_h_dim,
                         train_types = n_x_sp_train_types,
                         has_curriculum = not curriculum.is_random,
                         has_adversary = TeamType.SELF_PLAY_ADVERSARY in n_x_sp_train_types,
                         suffix=args.primary_learner_type,
                         )
    
    agents = load_agents(args, name=name, tag=CheckedPoints.FINAL_TRAINED_MODEL, force_training=primary_force_training)
    if agents:
        return agents[0]

    n_x_sp_types_trainer = RLAgentTrainer(name=name,
                                        args=args,
                                        agent=init_agent,
                                        teammates_collection=teammates_collection,
                                        epoch_timesteps=args.epoch_timesteps,
                                        n_envs=args.n_envs,
                                        curriculum=curriculum,
                                        seed=args.N_X_SP_seed,
                                        hidden_dim=args.N_X_SP_h_dim,
                                        learner_type=args.primary_learner_type)
    n_x_sp_types_trainer.train_agents(total_train_timesteps=n_x_sp_total_training_timesteps)



def get_adversary_agent(args, primary_agent, attack_round, tag=None):
    teammates_collection = generate_TC_for_Adversary(args=args,
                                                    agent=primary_agent)

    name = generate_name(args,
                        prefix='adv',
                        seed=args.ADV_seed,
                        h_dim=args.ADV_h_dim,
                        train_types=[TeamType.SELF_PLAY_HIGH],
                        has_curriculum=False,
                        has_adversary=True,
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
                                        learner_type=args.adversary_learner_type)
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