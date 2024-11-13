from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.tags import KeyCheckpoints


def load_agents(args, name, tag, path=None, force_training=False):
    if force_training:
        return []
    try:
        agents, _, _ = RLAgentTrainer.load_agents(args, name=name, path=path, tag=tag)
        return agents
    except FileNotFoundError as e:
        print(f'Could not find saved {name} agent \nFull Error: {e}')
        return []


def generate_name(args, prefix, seed, h_dim, train_types, has_curriculum, suffix=None):
    fname = prefix + '_s' + str(seed) + '_h' + str(h_dim) +'_tr['+'_'.join(train_types)+']'
    fname = fname + '_cur' if has_curriculum else fname + '_ran'
    if suffix:
        fname = fname + '_'+ suffix
    return fname
