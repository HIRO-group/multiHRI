from oai_agents.agents.base_agent import OAITrainer
from pathlib import Path
from oai_agents.common.tags import KeyCheckpoints
import shutil

def test_list_agent_checked_tags():
    # Define base directory based on the current working directory
    base_dir = Path.cwd()

    # Set up the directory structure for testing
    # This will create the following structure within the current working directory:
    #
    # <current_working_directory>/
    # └── agent_models/
    #     └── test_agents_folder/
    #         └── test_agent/
    #             ├── ck_0/
    #             ├── ck_1_rew_59.5/
    #             ├── ck_2_rew_140.0/
    #             ├── ck_10_rew_336.8888888888889/
    #             ├── ck_3_invalid/                 # Should not match
    #             ├── ck_4_rew_invalid/             # Should not match
    #             ├── unrelated_tag/                # Should not match
    #             ├── best/                         # Should not match
    #             └── last/                         # Should not match
    #
    # Only `ck_0`, `ck_1_rew_59.5`, `ck_2_rew_140.0`, and `ck_10_rew_336.8888888888889`
    # should be returned by the function.

    test_dir = base_dir / "agent_models" / "test_agents_folder" / "test_agent"
    test_dir.mkdir(parents=True, exist_ok=True)  # Ensure all parent directories are created

    # Simulate directory structure with various tags
    tag_names = [
        "ck_0",
        "ck_1_rew_59.5",
        "ck_2_rew_140.0",
        "ck_10_rew_336.8888888888889",
        "ck_3_invalid",               # Invalid because it doesn't have a valid float after the integer
        "ck_4_rew_invalid",            # Invalid because reward value is not a float
        "unrelated_tag",                # Invalid because it doesn't start with `KeyCheckpoints.CHECKED_MODEL_PREFIX`
        "best",
        "last"
    ]

    # Create these tag directories within the test directory
    for tag_name in tag_names:
        (test_dir / tag_name).mkdir(parents=True, exist_ok=True)

    # Mock args object with base_dir and exp_dir pointing to the test directory
    class MockArgs:
        def __init__(self, base_dir, exp_dir, layout_names=[]):
            self.base_dir = base_dir
            self.exp_dir = "test_agents_folder"
            self.layout_names = layout_names

    args = MockArgs(base_dir=base_dir, exp_dir="test_agents_folder")

    # Call the function to test
    checked_tags = OAITrainer.list_agent_checked_tags(args, name="test_agent")

    # Expected tags should only include those that match the pattern
    expected_tags = [
        "ck_0",
        "ck_1_rew_59.5",
        "ck_2_rew_140.0",
        "ck_10_rew_336.8888888888889"
    ]

    # Print results for verification
    if sorted(checked_tags) == sorted(expected_tags):
        print("Test passed: Tags returned as expected.")
    else:
        print(f"Test failed: Expected {expected_tags}, but got {checked_tags}")

    # Clean up the test directories after the test
    # This will remove the entire "agent_models/test_agents_folder" structure created for testing
    shutil.rmtree(base_dir / "agent_models" / "test_agents_folder")

if __name__ == "__main__":
    # Run the test function
    test_list_agent_checked_tags()




