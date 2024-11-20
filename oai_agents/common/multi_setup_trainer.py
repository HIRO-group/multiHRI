import concurrent.futures
import random
from scripts.utils.common import generate_name
from oai_agents.common.learner import LearnerType
from oai_agents.common.tags import Prefix, KeyCheckpoints
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
        self.num_of_training_variants = args.num_of_training_variants
        self.for_evaluation = args.for_evaluation

    def generate_hdim_and_seed(self):
        training_seeds = [1010, 2020, 2602, 13, 68, 2907, 105, 128]
        training_hdims = [256] * len(training_seeds)

        evaluation_seeds = [3031, 4041, 5051, 3708, 3809, 3910, 4607, 5506]
        evaluation_hdims = [256] * len(evaluation_seeds)

        if self.for_evaluation:
            seeds = evaluation_seeds
            hdims = evaluation_hdims
            min_seed, max_seed = 3000, 5999
        else:
            seeds = training_seeds
            hdims = training_hdims
            min_seed, max_seed = 0, 2999

        selected_seeds = []
        selected_hdims = []

        if self.num_of_training_variants <= len(seeds):
            selected_seeds = seeds[:self.num_of_training_variants]
            selected_hdims = hdims[:self.num_of_training_variants]
        else:
            selected_seeds = seeds[:]
            selected_hdims = hdims[:]

            remaining = self.num_of_training_variants - len(seeds)
            available_seeds = list(set(range(min_seed, max_seed + 1)) - set(selected_seeds))
            random_seeds = random.sample(available_seeds, remaining)
            random_hdims = [256] * remaining

            selected_seeds += random_seeds
            selected_hdims += random_hdims

        return selected_seeds, selected_hdims

    def get_trained_agent(self, seed, h_dim):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_multiple_trained_agents(self):
        agents = []

        seeds, hdims = self.generate_hdim_and_seed()
        inputs = [
            (seeds[i], hdims[i])
            for i in range(self.num_of_training_variants)
        ]

        if self.args.parallel:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.args.max_concurrent_jobs) as executor:
                arg_lists = list(zip(*inputs))
                executor.map(self.get_trained_agent, *arg_lists)
                # dilled_results = list(executor.map(self.get_trained_agent, *arg_lists))
            # for dilled_res in dilled_results:
            #     agent = dill.loads(dilled_res)
            #     agents.append(agent)
        else:
            for i in range(self.num_of_training_variants):
                agents.append(self.get_trained_agent(seed=seeds[i], h_dim=hdims[i]))

        # return agents

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
        if self.args.resume:
            last_ckpt = RLAgentTrainer.get_most_recent_checkpoint(args=self.args, name=name)
            if last_ckpt:
                agent_ckpt_info, env_info, training_info = RLAgentTrainer.load_agents(args=self.args, name=name, tag=last_ckpt)
                agent_ckpt = agent_ckpt_info[0]
                start_step = env_info["step_count"]
                start_timestep = env_info["timestep_count"]
                ck_list = training_info["ck_list"]
                print(f"Restarting training from step: {start_step} (timestep: {start_timestep})")

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
            n_envs=self.args.n_envs,
            start_step=start_step,
            start_timestep=start_timestep
        )

        rlat.train_agents(
            total_train_timesteps=total_train_timesteps,
            tag_for_returning_agent=self.tag_for_returning_agent,
            resume_ck_list=ck_list
        )

        agent = rlat.get_agents()[0]

        # if self.parallel:
        #     dill.dumps(agent)

        # return agent


class MultiSetupSPTrainer(MultiSetupTrainer):
    def get_trained_agent(self, seed, h_dim):
        name = generate_name(
            args=self.args,
            prefix=Prefix.SELF_PLAY,
            seed=seed,
            h_dim=h_dim,
            train_types=self.train_types,
            has_curriculum=not self.curriculum.is_random
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

def get_SP_agents(args, train_types, eval_types, curriculum, tag_for_returning_agent):
    sp_trainer = MultiSetupSPTrainer(
        args=args,
        train_types=train_types,
        eval_types=eval_types,
        curriculum=curriculum,
        tag_for_returning_agent=tag_for_returning_agent,
    )
    return sp_trainer.get_multiple_trained_agents()

# Example usage:
# sp_trainer = MultiSetupSPTrainer(args=args, num_of_training_variants=4, train_types=train_types, eval_types=eval_types, curriculum=curriculum, tag=tag)
# trained_agents = sp_trainer.get_multiple_trained_agents()
# Alternatively:
# trained_agents = get_SP_agents(args=args, num_of_training_variants=4, train_types=train_types, eval_types=eval_types, curriculum=curriculum, parallel=True, tag=tag)
