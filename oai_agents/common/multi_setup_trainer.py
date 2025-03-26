import concurrent.futures
from scripts.utils.common import generate_name
from oai_agents.common.tags import Prefix
from oai_agents.agents.rl import RLAgentTrainer
import dill


class MultiSetupTrainer:
    def __init__(
            self,
            args,
            train_types,
            eval_types,
            curriculum,
            tag_for_returning_agent
        ):
        self.args = args
        self.train_types = train_types
        self.eval_types = eval_types
        self.curriculum = curriculum
        self.tag_for_returning_agent = tag_for_returning_agent

        self.parallel = args.parallel
        self.total_ego_agents = args.total_ego_agents
        self.for_evaluation = args.gen_pop_for_eval

    def get_trained_agent(self, seed, h_dim):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_multiple_trained_agents(self):
        agents = []

        seeds, hdims = generate_hdim_and_seed(
            for_evaluation=self.for_evaluation, total_ego_agents=self.total_ego_agents)
        inputs = [
            (seeds[i], hdims[i])
            for i in range(self.total_ego_agents)
        ]

        if self.args.parallel:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.args.max_concurrent_jobs) as executor:
                arg_lists = list(zip(*inputs))
                dilled_results = list(executor.map(self.get_trained_agent, *arg_lists))
            for dilled_res in dilled_results:
                checkpoints_list = dill.loads(dilled_res)
        else:
            for inp in inputs:
                checkpoints_list = self.get_trained_agent(seed=inp[0], h_dim=inp[1])

    def get_reinforcement_agent(
            self,
            name,
            teammates_collection,
            curriculum,
            h_dim,
            seed,
            learner_type,
            checkpoint_rate,
            total_train_timesteps,
        ):
        agent_ckpt = None
        start_step = 0
        start_timestep = 0
        ck_list = None
        n_envs=self.args.n_envs
        if self.args.resume:
            last_ckpt = RLAgentTrainer.get_most_recent_checkpoint(args=self.args, name=name)
            if last_ckpt:
                agent_ckpt_info, env_info, training_info = RLAgentTrainer.load_agents(args=self.args, name=name, tag=last_ckpt)
                agent_ckpt = agent_ckpt_info[0]
                start_step = env_info["step_count"]
                start_timestep = env_info["timestep_count"]
                ck_list = training_info["ck_list"]
                n_envs = training_info["n_envs"]
                print(f"The model with {seed} Restarting training from step: {start_step} (timestep: {n_envs*start_timestep})")

        rlat = RLAgentTrainer(
            args=self.args,
            name=name,
            teammates_collection=teammates_collection,
            curriculum=curriculum,
            hidden_dim=h_dim,
            seed=seed,
            checkpoint_rate=checkpoint_rate,
            learner_type=learner_type,
            agent=agent_ckpt,
            epoch_timesteps=self.args.epoch_timesteps,
            n_envs=n_envs,
            start_step=start_step,
            start_timestep=start_timestep
        )

        rlat.train_agents(
            total_train_timesteps=total_train_timesteps,
            tag_for_returning_agent=self.tag_for_returning_agent,
            resume_ck_list=ck_list
        )

        agent = rlat.get_agents()[0]
        checkpoint_list = rlat.ck_list

        if self.parallel:
            return dill.dumps(checkpoint_list)

        return checkpoint_list


class MultiSetupSPTrainer(MultiSetupTrainer):
    def get_trained_agent(self, seed, h_dim):
        name = generate_name(
            args=self.args,
            prefix=Prefix.SELF_PLAY,
            seed=seed,
            h_dim=h_dim,
            train_types=self.train_types,
            curriculum=self.curriculum
        )

        return self.get_reinforcement_agent(
            name=name,
            teammates_collection={},
            curriculum=self.curriculum,
            h_dim=h_dim,
            seed=seed,
            learner_type=self.args.primary_learner_type,
            checkpoint_rate=self.args.pop_total_training_timesteps // self.args.num_of_ckpoints,
            total_train_timesteps=self.args.pop_total_training_timesteps,
        )

def generate_hdim_and_seed(for_evaluation: bool, total_ego_agents: int):
    evaluation_seeds = [3031, 4041, 5051, 3708, 3809, 3910, 4607, 5506]
    evaluation_hdims = [256] * len(evaluation_seeds)

    training_seeds = [
        1010, 2020, 2602, 13,
        68, 2907, 105, 128
    ]
    training_hdims = [256] * len(training_seeds)

    if for_evaluation:
        assert total_ego_agents <= len(evaluation_seeds), (
            f"Total ego agents ({total_ego_agents}) cannot exceed the number of evaluation seeds ({len(evaluation_seeds)}). "
            "Please either increase the number of evaluation seeds in the `generate_hdim_and_seed` function or decrease "
            f"`self.total_ego_agents` (currently set to {total_ego_agents}, based on `args.total_ego_agents`)."
        )
        seeds = evaluation_seeds
        hdims = evaluation_hdims
    else:
        assert total_ego_agents <= len(training_seeds), (
            f"Total ego agents ({total_ego_agents}) cannot exceed the number of training seeds ({len(training_seeds)}). "
            "Please either increase the number of training seeds in the `generate_hdim_and_seed` function or decrease "
            f"`self.total_ego_agents` (currently set to {total_ego_agents}, based on `args.total_ego_agents`)."
        )
        seeds = training_seeds
        hdims = training_hdims

    selected_seeds = seeds[:total_ego_agents]
    selected_hdims = hdims[:total_ego_agents]

    return selected_seeds, selected_hdims
