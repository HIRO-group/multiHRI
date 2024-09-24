from oai_agents.agents.rl import RLAgentTrainer


def load_agents(args, name, tag, path=None, force_training=False):
    if force_training:
        return []
    try:
        agents = RLAgentTrainer.load_agents(args, name=name, path=path, tag=tag or 'best')
        return agents
    except FileNotFoundError as e:
        print(f'Could not find saved {name} agent \nFull Error: {e}')
        return []


def generate_name(args, prefix, seed, h_dim, train_types, has_curriculum):
    fname = prefix + '_s' + str(seed) + '_h' + str(h_dim) +'_tr('+'_'.join(train_types)+')'
    fname = fname + '_cur' if has_curriculum else fname + '_ran'
    return fname
