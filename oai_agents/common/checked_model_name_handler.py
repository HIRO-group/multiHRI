from oai_agents.common.tags import KeyCheckpoints
import re
from pathlib import Path
from typing import Optional, List, Union

class CheckedModelNameHandler:
    def __init__(self):
        """
        Initializes the CheckedModelNameHandler with default prefix and reward substring.
        """
        self.prefix = KeyCheckpoints.CHECKED_MODEL_PREFIX
        self.reward_substr = KeyCheckpoints.REWARD_SUBSTR
        self.pattern = re.compile(f"^{re.escape(self.prefix)}(\\d+)(?:{re.escape(self.reward_substr)}[\\d.]+)?$")

    def generate_tag(self, id: int, mean_reward: Optional[float] = None) -> str:
        """
        Generate a checked model name based on the given id and mean reward.

        :param id: The identifier for the model, used as a numeric suffix.
        :param mean_reward: Optional mean reward to include in the model name, required for ids greater than 0.
        :return: A string representing the generated checked model name.
        :raises ValueError: If id is negative or if mean_reward is not provided for ids greater than 0.
        """
        if id < 0:
            raise ValueError("ID must be a non-negative integer.")

        if id == 0:
            return f"{self.prefix}{id}"

        if mean_reward is None:
            raise ValueError("Mean reward must be provided for IDs greater than 0.")

        return f"{self.prefix}{id}{self.reward_substr}{mean_reward}"

    def is_valid_checked_tag(self, tag: str) -> bool:
        """
        Check if a tag name matches the required pattern for checked models.

        :param tag: The tag name to validate.
        :return: True if the tag name matches the pattern; otherwise, False.
        """
        return bool(self.pattern.match(tag))

    def get_all_checked_tags(self, path: Union[Path, None] = None) -> List[str]:
        """
        Retrieve all valid checked model tags (subdirectories) under the specified path that match the pattern.

        :param path: The directory path to search for valid checked model tags. Can be a Path object or None.
        :return: A list of valid checked model tag names.
        """
        if path is None:
            raise ValueError("The path cannot be None.")

        path = Path(path) if not isinstance(path, Path) else path

        if not path.exists():
            raise FileNotFoundError(f"The specified path '{path}' does not exist.")
        if not path.is_dir():
            raise NotADirectoryError(f"The specified path '{path}' is not a directory.")

        tags = []
        for tag_path in path.iterdir():
            if tag_path.is_dir() and self.pattern.match(tag_path.name):
                match = self.pattern.match(tag_path.name)
                integer_part = int(match.group(1))
                # Only add tags that either have no reward substring for integer 0, or have it when integer > 0
                if integer_part == 0 or (integer_part > 0 and self.reward_substr in tag_path.name):
                    tags.append(tag_path.name)
        return tags
