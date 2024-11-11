from oai_agents.common.tags import KeyCheckpoints
import re
from typing import Optional

from pathlib import Path
from typing import List, Union
from oai_agents.common.tags import KeyCheckpoints

class CheckedModelNameHandler:
    def __init__(self):
        """
        Initializes the CheckedModelNameHandler with optional custom prefix and reward substring.

        :param prefix: Custom prefix for model names.
        :param reward_substr: Custom reward substring for model names.
        """
        self.prefix = KeyCheckpoints.CHECKED_MODEL_PREFIX
        self.reward_substr = KeyCheckpoints.REWARD_SUBSTR
        self.pattern = re.compile(f"^{re.escape(self.prefix)}(\\d+)(?:{re.escape(self.reward_substr)}[\\d.]+)?$")

    def generate_checked_model_name(self, id: int, mean_reward: Optional[float] = None) -> str:
        """
        Generate a checked model name based on the given id and mean reward.

        :param id: The identifier for the model, used as a numeric suffix.
        :param mean_reward: The mean reward to include in the model name, if applicable.
        :return: A string representing the generated checked model name.
        :raises ValueError: If id is negative or if mean_reward is not provided for ids greater than 0.
        """
        # Validate id
        if id < 0:
            raise ValueError("id must be a non-negative integer.")

        # When id is 0, mean_reward can be None
        if id == 0:
            return f"{self.prefix}{id}"

        # For id > 0, mean_reward must be provided
        if mean_reward is None:
            raise ValueError("mean_reward must be provided for ids greater than 0.")

        # Return the model name including mean_reward
        return f"{self.prefix}{id}{self.reward_substr}{mean_reward}"



    def is_valid_checked_model_name(self, model_name: str) -> bool:
        """
        Check if a model name matches the required pattern for checked models.

        :param model_name: The model name to validate.
        :return: True if the model name matches the pattern; otherwise, False.
        """
        return bool(self.pattern.match(model_name))

    def get_checked_model_tags(self, path: Union[Path, None] = None) -> List[str]:
        """
        Retrieve all valid checked model tags (subdirectories) under the specified path that match the pattern.

        :param path: The directory path to search for valid checked model tags. Can be a Path object or None.
        :return: A list of valid checked model tag names.
        :raises ValueError: If the path is None.
        :raises FileNotFoundError: If the specified path does not exist.
        :raises NotADirectoryError: If the specified path is not a directory.
        """
        if path is None:
            raise ValueError("The path cannot be None.")

        # # Convert to Path if not already a Path object
        path = Path(path) if not isinstance(path, Path) else path

        if not path.exists():
            raise FileNotFoundError(f"The specified path '{path}' does not exist.")
        if not path.is_dir():
            raise NotADirectoryError(f"The specified path '{path}' is not a directory.")

        tags = []
        for tag in path.iterdir():
            if tag.is_dir() and self.pattern.match(tag.name):
                match = self.pattern.match(tag.name)
                integer_part = int(match.group(1))
                # Only add tags that either have no reward substring for integer 0, or have it when integer > 0
                if integer_part == 0 or (integer_part > 0 and self.reward_substr in tag.name):
                    tags.append(tag.name)
        return tags

