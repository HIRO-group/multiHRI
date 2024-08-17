class Learner:
    magnifier = 3
    def calculate_reward(self, individual_reward, group_reward):
        raise NotImplementedError("This method should be overridden by subclasses")


class Saboteur(Learner):
    def calculate_reward(self, individual_reward, group_reward):
        return Learner.magnifier * (2 * individual_reward - 1 * group_reward)


class Selfisher(Learner):
    def calculate_reward(self, individual_reward, group_reward):
        return Learner.magnifier * (1 * individual_reward - 1 * group_reward)


class SoloWorker(Learner):
    def calculate_reward(self, individual_reward, group_reward):
        return Learner.magnifier * (1 * individual_reward + 0 * group_reward)


class Collaborator(Learner):
    def calculate_reward(self, individual_reward, group_reward):
        return Learner.magnifier * (0.5 * individual_reward + 0.5 * group_reward)


class Helper(Learner):
    def calculate_reward(self, individual_reward, group_reward):
        return Learner.magnifier * (1/3 * individual_reward + 2/3 * group_reward)