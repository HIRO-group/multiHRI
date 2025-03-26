from typing import Optional, Union
from pathlib import Path

def get_model_path(base_dir: Union[str, Path], exp_folder: Optional[str], model_name: str) -> Path:
    """
    Constructs a path for saving or loading an agent model.

    Parameters:
        base_dir (str or Path): The base directory where models are stored.
        exp_folder (str or None): The experiment folder name, or None if not applicable.
        model_name (str): The name of the model.

    Returns:
        Path: A Path object representing the constructed path.
    """
    # Ensure base_dir is a Path object
    base_dir = Path(base_dir) if isinstance(base_dir, str) else base_dir
    experiment_name = get_experiment_name(exp_folder=exp_folder, model_name=model_name)
    path = base_dir / 'agent_models' /experiment_name

    return path

def get_experiment_models_dir(base_dir: Union[str, Path], exp_folder: Optional[str]) -> Path:
    """
    Constructs a path for saving or loading agent models.

    Parameters:
        base_dir (str or Path): The base directory where models are stored.
        exp_folder (str or None): The experiment folder name, or None if not applicable.
        model_name (str): The name of the model.

    Returns:
        Path: A Path object representing the constructed path.
    """
    # Ensure base_dir is a Path object
    base_dir = Path(base_dir) if isinstance(base_dir, str) else base_dir
    path = base_dir / 'agent_models' /exp_folder

    return path

def get_experiment_name(exp_folder: Optional[str], model_name: str):
    return f"{exp_folder}/{model_name}" if exp_folder else model_name
