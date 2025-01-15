from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.tags import KeyCheckpoints
from oai_agents.common.curriculum import Curriculum


def load_agents(args, name, tag, path=None, force_training=False):
    if force_training:
        return []
    try:
        agents, _, _ = RLAgentTrainer.load_agents(args, name=name, path=path, tag=tag)
        return agents
    except FileNotFoundError as e:
        print(f'Could not find saved {name} agent \nFull Error: {e}')
        return []


def generate_name(args, prefix, seed, h_dim, train_types:list=None, curriculum:Curriculum=None, suffix=None):
    
    assert ((train_types is not None) or (curriculum is not None)), "Must provide either train_types or curriculum to generate name for model"

    if (train_types is None):
        train_types = curriculum.train_types

    fname = prefix + '_s' + str(seed) + '_h' + str(h_dim) +'_tr['+'_'.join(train_types)+']'
    if (not curriculum) or (curriculum.is_random):
        curriculum_type = '_ran'
    elif curriculum.prioritized_sampling:
        curriculum_type = '_ps'
    else:
        curriculum_type = '_cur'
    fname = fname + curriculum_type
    if suffix:
        fname = fname + '_'+ suffix
    return fname
