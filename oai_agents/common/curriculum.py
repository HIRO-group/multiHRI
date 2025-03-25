import random
import numpy as np
from oai_agents.common.tags import TeamType
import wandb

class Curriculum:
    '''
    Example:
        training_phases_durations_in_order = {
            (TeamType.LOW_FIRST, TeamType.SELF_PLAY_ADVERSARY): 0.5,  first 50% of the training time randomly select between low_first and self_play_adversary
            (TeamType.MEDIUM_FIRST): 0.125,                           next 12.5% of the training time
            (TeamType.HIGH_FIRST): 0.125,                             next 12.5% of the training time
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
    the rest of the parameters.

    Prioritized sampling can also be used to sample teammates such that the teammates with the worst historical
    performance are given the highest probability of selection. In this case, TeamTypes are ignored.
    '''
    def __init__(self, 
                 train_types, 
                 is_random,
                 eval_types=None,
                 prioritized_sampling=False,
                 priority_scaling = 1.0,
                 total_steps=None,
                 training_phases_durations_in_order=None,
                 rest_of_the_training_probabilities=None,
                 probabilities_decay_over_time=None):

        self.train_types = train_types
        self.eval_types = eval_types
        self.is_random = is_random
        self.prioritized_sampling = prioritized_sampling
        self.priority_scaling = priority_scaling
        self.teamtype_performances = {}
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
        elif self.prioritized_sampling:
            assert self.train_types == (self.eval_types['generate'] + self.eval_types['load']), "train_types and eval_types must be identical when using prioritized sampling"
        else:
            phase_team_types = {team for teams in self.training_phases_durations_in_order.keys() for team in (teams if isinstance(teams, tuple) else (teams,))}
            rest_team_types = set(self.rest_of_the_training_probabilities.keys())
            assert set(self.train_types) == phase_team_types.union(rest_team_types), "Invalid training types"
            assert sum(self.training_phases_durations_in_order.values()) <= 1, "Sum of training_phases_durations_in_order should be <= 1"
            assert 0 <= self.probabilities_decay_over_time <= 1, "probabilities_decay_over_time should be between 0 and 1"
            if sum(self.training_phases_durations_in_order.values()) < 1:
                assert sum(self.rest_of_the_training_probabilities.values()) == 1, "Sum of rest_of_the_training_probabilities should be 1"

    def update(self, current_step):
        self.current_step = current_step

    def update_teamtype_performances(self, teamtype_performances:dict)->None:
        '''
        Update the curriculum with the latest dictionary of performances for use in prioritized sampling
        NOTE: this must contain performance values for all possible combinations of teammates

        :param teamtype_performances: Dictionary mapping agent teamtypes to their latest rollout performance e.g. for a 3-player game with 3 possible teammates
        {<layout> : {TeamType.HIGH_FIRST: : <score>, TeamType.MEDIUM_FIRST: : <score>, ... }}
        '''
        assert self.prioritized_sampling is True, "Updating curriculum teamtype performances while curriculum is not set for prioritized sampling"
        self.teamtype_performances = teamtype_performances


    def select_teammates_for_layout(self, population_teamtypes, layout):
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

        elif self.prioritized_sampling:
            teammates = self.select_teammates_prioritized_sampling(population_teamtypes, layout)
        else:
            teammates = self.select_teammates_based_on_curriculum(population_teamtypes)
        return teammates


    def select_teammates_based_on_curriculum(self, population_teamtypes):
        # Calculate the current phase based on current_step and total_steps
        cumulative_duration = 0
        for team_type_tuple, duration in self.training_phases_durations_in_order.items():
            cumulative_duration += duration
            if self.current_step / self.total_steps <= cumulative_duration:
                
                if type(team_type_tuple) is tuple:
                    team_type = random.choice(team_type_tuple)
                else:
                    team_type = team_type_tuple

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

    def select_teammates_prioritized_sampling(self, population_teamtypes:dict, layout:str):
        '''
        Select teammates according to prioritized sampling (lower performing teammates are prioritized for selection)
        NOTE: This function just uses the latest performance values for selection, it does not track performance of TeamTypes over time

        :param population_teamtypes: Dictionary mapping TeamTypes to lists of agent teammates of each type
        :param layout: The name fo the layout that teammates are being selected for
        '''

        teamtype_options = list(population_teamtypes.keys())

        # Check if teamtype performances have been set (only happens after first training round)
        if self.teamtype_performances:

            assert layout in self.teamtype_performances.keys(), "Requesting prioritized sampling teammates for unrecognized layout"

            # Ignore the performances for all layouts except the requested one
            teamtype_performances_for_layout = self.teamtype_performances[layout]
            teamtype_options = list(teamtype_performances_for_layout.keys())

            # Convert scores to priorities (lower score = higher priority)
            scores = list(teamtype_performances_for_layout.values())
            max_score = np.max(scores)
            # Invert scores
            priorities = max_score + 1 - scores 

            # Apply power transformation to increase contrast between priorities
            priorities = np.power(priorities, self.priority_scaling)

            # Add small epsilon to ensure no zero probabilities
            epsilon = 1e-5
            priorities += epsilon

            # Convert to probabilities
            probabilities = priorities / np.sum(priorities)

            # Create dict for easier logging
            teamtype_proabilities = {tt : prob for tt, prob in zip(teamtype_options, probabilities)}

            # Sample the teamtypes using the calculated probabilities
            prioritized_teamtype = np.random.choice(teamtype_options, p=probabilities)
        else:
            # Randomly select a teamtype
            prioritized_teamtype = np.random.choice(teamtype_options)
            # No probabilities defined yet
            teamtype_proabilities = {}

        # Randomly sample a team of agents from this teamtype
        wandb.log({"team_type_index": TeamType.map_to_index(prioritized_teamtype)})

        for teamtype in teamtype_proabilities:
            wandb.log({f'teamtype_{teamtype}_prioritized_sampling_probability': teamtype_proabilities[teamtype], 'current_step': self.current_step})

        teammates_for_teamtype = population_teamtypes[prioritized_teamtype]
        teammates = random.choice(teammates_for_teamtype)

        return teammates

    def print_curriculum(self):
        print("Curriculum:")
        if self.is_random:
            print("Random curriculum: ", self.train_types)
        elif self.prioritized_sampling:
            print("Prioritized sampling curriculum current teamtype performances (lower score = higher priority):")
            if not self.teamtype_performances:
                print("  No teamtype performance recieved yet, teamtype will be randomly sampled")
            else:
                for layout in self.teamtype_performances.keys():
                    print("  Layout: ", layout)
                    teamtype_perforamnces_for_layout = self.teamtype_performances[layout]
                    for teamtype in teamtype_perforamnces_for_layout.keys():
                        score = teamtype_perforamnces_for_layout[teamtype]
                        print("    TeamType: ", teamtype, " Score: ", score)
        else:
            print("Total steps:", self.total_steps)
            print("Training phases durations in order:", self.training_phases_durations_in_order)
            print("Rest of the training probabilities:", self.rest_of_the_training_probabilities)
            print("Probabilities decay over time:", self.probabilities_decay_over_time)


    def validate_curriculum_types(self, expected_types:list, unallowed_types:list) -> None:
        # Ensure at least one expected type is present in train_types
        assert any(et in self.train_types for et in expected_types), \
            "Error: None of the expected types are present in train_types."

        # Ensure no unallowed types are present in train_types
        assert not any(ut in self.train_types for ut in unallowed_types), \
            "Error: One or more unallowed types are present in train_types."