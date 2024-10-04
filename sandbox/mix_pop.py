from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.arguments import get_arguments
from oai_agents.common.tags import TeamType
from oai_agents.common.learner import LearnerType

three_chefs_layouts = [
        'selected_3_chefs_coordination_ring',
        'selected_3_chefs_counter_circuit',
        'selected_3_chefs_cramped_room'
]

population_old = {layout_name: [] for layout_name in three_chefs_layouts}
args = get_arguments()
args.teammates_len = 2
args.num_players = args.teammates_len + 1 
args.layout_names = three_chefs_layouts
args.exp_dir = 'old_3'

for layout_name in args.layout_names:
    name = f'pop_{layout_name}'
    population_old[layout_name] = RLAgentTrainer.load_agents(args, name=name, tag='aamas25')
    print(f'Loaded pop with {len(population_old[layout_name])} agents.')

population_new = {layout_name: [] for layout_name in three_chefs_layouts}
args.exp_dir = 'Final/3'

for layout_name in args.layout_names:
    name = f'pop_{layout_name}'
    population_new[layout_name] = RLAgentTrainer.load_agents(args, name=name, tag='aamas25')
    print(f'Loaded pop with {len(population_new[layout_name])} agents.')


population_mixed = {layout_name: [] for layout_name in three_chefs_layouts}
for layout_name in args.layout_names:
    population_mixed[layout_name] = population_old[layout_name] + population_new[layout_name]
    print(f'Created mixed pop with {len(population_mixed[layout_name])} agents.')


args.n_envs = 200
args.epoch_timesteps = 1e5

args.primary_learner_type = LearnerType.SUPPORTER
args.adversary_learner_type = LearnerType.SELFISHER
args.pop_learner_type = LearnerType.ORIGINALER
args.attack_rounds = 3

how_long = 4.0
args.pop_total_training_timesteps = int(5e6 * how_long)
args.n_x_sp_total_training_timesteps = int(5e6 * how_long)
args.adversary_total_training_timesteps = int(5e6 * how_long)

args.fcp_total_training_timesteps = int(5e6 * how_long)
args.n_x_fcp_total_training_timesteps = int(2 * args.fcp_total_training_timesteps * how_long)

args.SP_seed, args.SP_h_dim = 68, 256
args.N_X_SP_seed, args.N_X_SP_h_dim = 1010, 256
args.FCP_seed, args.FCP_h_dim = 2020, 256
args.N_X_FCP_seed, args.N_X_FCP_h_dim = 2602, 256
args.ADV_seed, args.ADV_h_dim = 68, 512

# Save the mixed population
args.exp_dir = 'corrected_wrong'
args.n_envs = 200
for layout_name in args.layout_names:
    name = f'pop_{layout_name}'
    rt = RLAgentTrainer(
        name=f'{name}',
        args=args,
        agent=None,
        teammates_collection={},
        train_types=[TeamType.SELF_PLAY],
        eval_types=[TeamType.SELF_PLAY],
        epoch_timesteps=args.epoch_timesteps,
        n_envs=args.n_envs,
        learner_type=args.pop_learner_type,
        seed=None,
    )
    rt.agents = population_mixed[layout_name]
    rt.save_agents(tag='aamas25')
    print(f'Saved pop with {len(population_mixed[layout_name])} agents to {args.exp_dir}/{name}.')
    # print(f'Saved pop with {len(population_mixed[layout_name])} agents to {args.exp_dir}/{name}.pkl')