# test_population.py

from oai_agents.common.population import generate_hdim_and_seed

def test_generate_hdim_and_seed():
    '''
    Test function for generate_hdim_and_seed to ensure:
    1. The number of (hidden_dim, seed) pairs matches the number of required agents.
    2. All generated seeds are unique.
    3. Hidden dimensions are as expected (either 64 or 256).
    '''

    # Test cases
    test_cases = [3, 5, 8, 10]  # Testing for fewer than, equal to, and more than predefined settings

    for num_agents in test_cases:
        print(f"\nTesting with {num_agents} agents:")

        # Generate (hidden_dim, seed) pairs
        selected_seeds, selected_hdims = generate_hdim_and_seed(num_agents)

        # Check that the correct number of agents is generated
        assert len(selected_seeds) == num_agents, f"Expected {num_agents} seeds, got {len(selected_seeds)}"
        assert len(selected_hdims) == num_agents, f"Expected {num_agents} hidden dims, got {len(selected_hdims)}"

        # Check that all seeds are unique
        assert len(set(selected_seeds)) == num_agents, "Duplicate seeds found in the generated seeds."

        # Check that hidden dims are from the valid set (64, 256)
        assert all(hdim in [256, 512] for hdim in selected_hdims), "Invalid hidden dimension found. Only 64 and 256 are allowed."

        print(f"Test passed for {num_agents} agents.")
        print("Selected seeds:", selected_seeds)
        print("Selected hidden dimensions:", selected_hdims)

# Ensure that this test script only runs when executed directly
if __name__ == "__main__":
    print("Running tests in population.py...")
    test_generate_hdim_and_seed()
