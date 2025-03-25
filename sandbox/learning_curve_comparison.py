import numpy as np
import hashlib
from pathlib import Path
import pickle as pkl
import matplotlib.pyplot as plt
from typing import Dict, Tuple

import wandb

def cache_run_data(run, key):
    run_hash = hashlib.md5(run.id.encode()).hexdigest()
    cache_path = Path(f"eval_cache/{run.name}_{run_hash}.pkl")

    if cache_path.is_file():
        with open(cache_path, "rb") as f:
            data = pkl.load(f)
        print(f"Loaded data for {run.name} from cache.")
    else:
        data = np.array([row[key] for row in run.scan_history(keys=[key])])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pkl.dump(data, f)
        print(f"Cached data for {run.name}.")

    return data

def get_runs_data(layout_name):
    api = wandb.Api()
    amh_cur = api.run("ava-abd/overcooked_ai/982g6cd9")
    amh_ran = api.run("ava-abd/overcooked_ai/4nqumyn1")
    almh_ran = api.run("ava-abd/overcooked_ai/gu07go47")
    almh_cur = api.run("ava-abd/overcooked_ai/fv8gp8i0")
    key = f"eval_mean_reward_{layout_name}_teamtype_SPL"

    runs = [amh_cur, amh_ran, almh_ran, almh_cur]
    runs_data = {
        run.name: cache_run_data(run, key) for run in runs
    }

    return runs_data

def analyze_convergence(
    runs_data: Dict[str, np.ndarray],
    window_size: int = 10,
    threshold: float = 0.01,
    min_epochs: int = 20,
) -> Tuple[Dict[str, int], Dict[str, bool], plt.Figure]:
    """
    Analyze convergence of learning curves from wandb runs.
    Args:
        runs_data: Dictionary mapping run names to numpy arrays of learning curves
        window_size: Size of rolling window for calculating stability
        threshold: Maximum allowed relative change to consider converged
        min_epochs: Minimum epochs before considering convergence
    Returns:
        convergence_points: Dict mapping run names to convergence epoch (-1 if not converged)
        converged_flags: Dict mapping run names to convergence status
        fig: Matplotlib figure showing curves and convergence points
    """
    convergence_points = {}
    converged_flags = {}

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (run_name, curve) in enumerate(runs_data.items()):
        rolling_mean = np.convolve(curve, np.ones(window_size)/window_size, mode='valid')
        relative_changes = np.abs(np.diff(rolling_mean) / rolling_mean[:-1])
        stable_points = relative_changes < threshold
        convergence_point = -1
        converged = False

        for i in range(min_epochs, len(stable_points)):
            if np.all(stable_points[i:i+window_size]):
                convergence_point = i + window_size
                converged = True
                break

        convergence_points[run_name] = convergence_point
        converged_flags[run_name] = converged
        epochs = np.arange(len(curve))
        ax.plot(epochs, curve, label=run_name, alpha=0.7)

        if converged:
            ax.axvline(x=convergence_point, color=f'C{idx}', linestyle='--', alpha=0.5)
            ax.plot(convergence_point, curve[convergence_point], 'o',
                   color=f'C{idx}', markersize=10)

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Learning Curves Convergence Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return convergence_points, converged_flags, fig


if __name__ == "__main__":
    layout_name = 'selected_2_chefs_cramped_room'
    runs_data = get_runs_data(layout_name=layout_name)

    convergence_points, converged_flags, fig = analyze_convergence(
        runs_data,
        window_size=5,
        threshold=0.01,
        min_epochs=10,
    )

    for run_name in runs_data.keys():
        epoch = convergence_points[run_name]
        status = "converged" if converged_flags[run_name] else "not converged"
        print(f"{run_name}: {status} at timestep {epoch if epoch != -1 else 'N/A'}")

    plt.show()
