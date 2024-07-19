from oai_agents.agents.rl import RLAgentTrainer

def load_agents(args, name, tag, force_training=False):
    if force_training:
        return []
    try:
        agents = RLAgentTrainer.load_agents(args, name=name, tag=tag or 'best')
        return agents
    except FileNotFoundError as e:
        print(f'Could not find saved {name} agent \nFull Error: {e}')
        return []