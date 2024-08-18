import random
import numpy as np
from oai_agents.common.tags import TeamType
import wandb

class Curriculum:
    '''
    Example:
        training_phases_durations_in_order = {
            TeamType.LOW_FIRST: 0.5,  first 50% of the training time
            TeamType.MEDIUM_FIRST: 0.125,  next 12.5% of the training time
            TeamType.HIGH_FIRST: 0.125,   next 12.5% of the training time
        },
    
    For the rest of the training time (12.55%)
    Choose training types with the following probabilities
        rest_of_the_training_probabilities={
            TeamType.LOW_FIRST: 0.4,
            TeamType.MEDIUM_FIRST: 0.3, 
            TeamType.HIGH_FIRST: 0.3,
        },

        probabilities_decay_over_time = 0.1,
    Everytime an update_happens, the probabilities will be updated
    probability_of_playing becomes:
        TeamType.LOW_FIRST: 0.4 - 0.1,
        TeamType.MEDIUM_FIRST: 0.3 + (0.1/2), 
        TeamType.HIGH_FIRST: 0.3 + (0.1/2),
    
    
    WHENEVER we don't care about the order of the training types, we can set is_random=True.
    and we can just call Curriculum(train_types=sp_train_types, is_random=True) and ignore
    the rest of teh parameters.
    '''
    def __init__(self, train_types, is_random, total_steps=None, training_phases_durations_in_order=None, 
                 rest_of_the_training_probabilities=None, probabilities_decay_over_time=None):
        self.train_types = train_types
        self.is_random = is_random
        self.current_step = 0
        self.total_steps = total_steps
        self.training_phases_durations_in_order = training_phases_durations_in_order
        self.rest_of_the_training_probabilities = rest_of_the_training_probabilities
        self.probabilities_decay_over_time = probabilities_decay_over_time
        self.is_valid()
    
    def is_valid(self):
        if self.is_random:
            assert self.total_steps is None, "total_steps should be None for random curriculums"
            assert self.training_phases_durations_in_order is None, "training_phases_durations_in_order should be None for random curriculums"
            assert self.rest_of_the_training_probabilities is None, "rest_of_the_training_probabilities should be None for random curriculums"
            assert self.probabilities_decay_over_time is None, "probabilities_decay_over_time should be None for random curriculums"
        else:
            assert set(self.train_types) == set(self.training_phases_durations_in_order.keys()), "Invalid training types"
            assert set(self.train_types) == set(self.rest_of_the_training_probabilities.keys()), "Invalid training types"
            assert sum(self.training_phases_durations_in_order.values()) <= 1, "Sum of training_phases_durations_in_order should be <= 1"
            assert 0 <= self.probabilities_decay_over_time <= 1, "probabilities_decay_over_time should be between 0 and 1"
            if sum(self.training_phases_durations_in_order.values()) < 1:
                assert sum(self.rest_of_the_training_probabilities.values()) == 1, "Sum of rest_of_the_training_probabilities should be 1"

    def update(self, current_step):
        self.current_step = current_step
    
    def select_teammates(self, population_teamtypes):
        '''
        Population_teamtypes = {
            TeamType.HIGH_FIRST: [[agent1, agent2], [agent3, agent4], ...],
            TeamType.MEDIUM_FIRST: [[agent1, agent2], [agent3, agent4], ...],
            ...
        }
        '''
        if self.is_random:
            population = [population_teamtypes[t] for t in population_teamtypes.keys()]
            teammates_per_type = population[np.random.randint(len(population))]
            teammates = teammates_per_type[np.random.randint(len(teammates_per_type))]
            return teammates
        return self.select_teammates_based_on_curriculum(population_teamtypes)


    def select_teammates_based_on_curriculum(self, population_teamtypes):
        # Calculate the current phase based on current_step and total_steps
        cumulative_duration = 0
        for team_type, duration in self.training_phases_durations_in_order.items():
            cumulative_duration += duration
            if self.current_step / self.total_steps <= cumulative_duration:
                teammates_per_type = population_teamtypes[team_type]
                wandb.log({"team_type_index": TeamType.map_to_index(team_type)})
                return random.choice(teammates_per_type)
        
        # If the current_step is in the remaining training time
        decay = self.probabilities_decay_over_time * (self.current_step / self.total_steps)
        adjusted_probabilities = {
            team_type: max(0, prob - decay) if team_type == TeamType.LOW_FIRST else prob + decay / 2
            for team_type, prob in self.rest_of_the_training_probabilities.items()
        }
        adjusted_probabilities = {k: v / sum(adjusted_probabilities.values()) for k, v in adjusted_probabilities.items()}
        team_type = np.random.choice(
            list(adjusted_probabilities.keys()), 
            p=list(adjusted_probabilities.values())
        )
        wandb.log({"team_type_index": TeamType.map_to_index(team_type)})
        teammates_per_type = population_teamtypes[team_type]
        return random.choice(teammates_per_type)
    
    def print_curriculum(self):
        print("----------------")
        print("Curriculum:")
        if self.is_random:
            print("Random curriculum")
        else:
            print("Total steps:", self.total_steps)
            print("Training phases durations in order:", self.training_phases_durations_in_order)
            print("Rest of the training probabilities:", self.rest_of_the_training_probabilities)
            print("Probabilities decay over time:", self.probabilities_decay_over_time)
        print("---------------")


    def validate_curriculum_types(self, expected_types:list, unallowed_types:list) -> None:
        # Ensure at least one expected type is present in train_types
        assert any(et in self.train_types for et in expected_types), \
            "Error: None of the expected types are present in train_types."

        # Ensure no unallowed types are present in train_types
        assert not any(ut in self.train_types for ut in unallowed_types), \
            "Error: One or more unallowed types are present in train_types."
    

    

