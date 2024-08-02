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
    
    '''
    def __init__(self, train_types, training_phases_durations_in_order, 
                 rest_of_the_training_probabilities, probabilities_decay_over_time):
        self.train_types = train_types
        self.training_phases_durations_in_order = training_phases_durations_in_order
        self.rest_of_the_training_probabilities = rest_of_the_training_probabilities
        self.probabilities_decay_over_time = probabilities_decay_over_time
        self.is_valid()

    def is_valid(self):
        assert set(self.train_types) == set(self.training_phases_durations_in_order.keys()), "Invalid training types"
        assert set(self.train_types) == set(self.rest_of_the_training_probabilities.keys()), "Invalid training types"
        assert sum(self.training_phases_durations_in_order.values()) <= 1, "Sum of training_phases_durations_in_order should be <= 1"
        assert 0 <= self.probabilities_decay_over_time <= 1, "probabilities_decay_over_time should be between 0 and 1"
        if sum(self.training_phases_durations_in_order.values()) < 1:
            assert sum(self.rest_of_the_training_probabilities.values()) == 1, "Sum of rest_of_the_training_probabilities should be 1"


def get_curriculum(train_types, training_phases_durations_in_order, 
                   rest_of_the_training_probabilities, probabilities_decay_over_time):
    return Curriculum(train_types, training_phases_durations_in_order,
                        rest_of_the_training_probabilities, probabilities_decay_over_time)
    