from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.tags import TeamType
from oai_agents.common.population import get_population
from oai_agents.common.teammates_collection import generate_TC, generate_TC_for_Adversary, generate_TC_for_AdversarysPlay
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
                        parallel:bool=True,
                        num_SPs_to_train=2) -> tuple:
    

    
    curriculum.validate_curriculum_types(expected_types = [TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_LOW],
                                         unallowed_types = TeamType.ALL_TYPES_BESIDES_SP)

    name = generate_name(args,
                         prefix = f'N-{args.unseen_teammates_len}-SP',
                         seed = args.N_X_SP_seed,
                         h_dim = args.N_X_SP_h_dim, 
                         train_types = n_x_sp_train_types,
                         has_curriculum = not curriculum.is_random)
    
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
        parallel=parallel,
        force_training=pop_force_training,
        tag = tag
    )

    randomly_init_agent = RLAgentTrainer.generate_randomly_initialized_agent(args=args, seed=args.N_X_SP_seed)
    teammates_collection = generate_TC(args=args,
                                        population=population,
                                        agent=randomly_init_agent,
                                        train_types=n_x_sp_train_types,
                                        eval_types_to_generate=n_x_sp_eval_types['generate'],
                                        eval_types_to_read_from_file=n_x_sp_eval_types['load'],
                                        unseen_teammates_len=args.unseen_teammates_len)

    n_x_sp_types_trainer = RLAgentTrainer(name=name,
                                           args=args,
                                           agent=randomly_init_agent,
                                           teammates_collection=teammates_collection,
                                           epoch_timesteps=args.epoch_timesteps,
                                           n_envs=args.n_envs,
                                           curriculum=curriculum,
                                           seed=args.N_X_SP_seed,
                                           hidden_dim=args.N_X_SP_h_dim)

    n_x_sp_types_trainer.train_agents(total_train_timesteps=n_x_sp_total_training_timesteps)
    return n_x_sp_types_trainer.get_agents()[0], teammates_collection




def get_FCP_agent_w_pop(args, 
                        pop_total_training_timesteps,
                        fcp_total_training_timesteps,
                        fcp_train_types,
                        fcp_eval_types,
                        fcp_curriculum,
                        pop_force_training,
                        primary_force_training,
                        num_SPs_to_train=2,
                        tag=None,
                        parallel=True):

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
        parallel=parallel,
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
                        parallel=True,
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
                                                parallel=parallel)

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
    agent = load_agent(Path(agent_path), args)
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
    )
    
    agent_trainer.train_agents(total_train_timesteps=total_training_timesteps)
    return agent_trainer.get_agents()[0], tc, name
