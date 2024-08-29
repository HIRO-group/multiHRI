class LearnerType:
    '''
    Different learners would receive differnt rewards from the base_overcooked_env.py.
    The primary difference between learners are on the weights of personal and group reward.
    Saboteur and Selfisher weight personal reward positively and group reward negatively.
    Soloworker, weight individual reward positively, and group reward with a zero.
    Collaborator, and Supporter weight both rewards positively.
    If we grade how much they care about group: 
    Supporter > Collaborator > Soloworker >> Selfisher > Saboteur.
    If more details are desired, learner.py define these learners.
    '''
    ORIGINALER = "originaler"
    SABOTEUR = "saboteur"
    SELFISHER = "selfisher"
    SOLOWORKER = "soloworker"
    COLLABORATOR = "collaborator"
    SUPPORTER = "supporter"

class Learner:
    def __new__(cls, learner_type: str, magnifier: float):
        learner_classes = {
            LearnerType.ORIGINALER: Originaler,
            LearnerType.SABOTEUR: Saboteur,
            LearnerType.SELFISHER: Selfisher,
            LearnerType.SOLOWORKER: SoloWorker,
            LearnerType.COLLABORATOR: Collaborator,
            LearnerType.SUPPORTER: Supporter,
        }

        if learner_type not in learner_classes:
            raise ValueError(f"Invalid learner type: {learner_type}")

        # Create an instance of the appropriate class
        instance = super().__new__(learner_classes[learner_type])
        return instance
    def __init__(self, learner_type: str, magnifier: float):
        '''
        magnifier is used to magnify the received reward. 
        This magnification would maginify the advantage.
        This further increase the gradient for the policy optimization.
        '''
        self.magnifier = magnifier 
        self.personal_reward = 0
        self.group_reward = 0
    
    def extract_reward(self, p_idx, env_info, ratio, num_players):
        group_sparse_r = sum(env_info['sparse_r_by_agent'])
        group_shaped_r = sum(env_info['shaped_r_by_agent'])               
        sparse_r = env_info['sparse_r_by_agent'][p_idx] if p_idx is not None else group_sparse_r
        shaped_r = env_info['shaped_r_by_agent'][p_idx] if p_idx is not None else group_shaped_r
        self.personal_reward = group_sparse_r * ratio + shaped_r * (1 - ratio)
        self.group_reward = (1/num_players) * (num_players * group_sparse_r * ratio + group_shaped_r * (1 - ratio))
    
    def calculate_reward(self, p_idx, env_info, ratio):
        raise NotImplementedError("This method should be overridden by subclasses")


class Originaler(Learner):
    def calculate_reward(self, p_idx, env_info, ratio, num_players):
        super().extract_reward(p_idx, env_info, ratio, num_players)
        return self.personal_reward


class Saboteur(Learner):
    def calculate_reward(self, p_idx, env_info, ratio, num_players):
        super().extract_reward(p_idx, env_info, ratio, num_players)
        return self.magnifier * (2/3 * self.personal_reward - 1/3 * self.group_reward)


class Selfisher(Learner):
    def calculate_reward(self, p_idx, env_info, ratio, num_players):
        super().extract_reward(p_idx, env_info, ratio, num_players)
        return self.magnifier * (1 * self.personal_reward - 1 * self.group_reward)


class SoloWorker(Learner):
    def calculate_reward(self, p_idx, env_info, ratio, num_players):
        super().extract_reward(p_idx, env_info, ratio, num_players)
        return self.magnifier * (1 * self.personal_reward + 0 * self.group_reward)


class Collaborator(Learner):
    def calculate_reward(self, p_idx, env_info, ratio, num_players):
        super().extract_reward(p_idx, env_info, ratio, num_players)
        return self.magnifier * (0.5 * self.personal_reward + 0.5 * self.group_reward)


class Supporter(Learner):
    def calculate_reward(self, p_idx, env_info, ratio, num_players):
        super().extract_reward(p_idx, env_info, ratio, num_players)
        return self.magnifier * (1/3 * self.personal_reward + 2/3 * self.group_reward)