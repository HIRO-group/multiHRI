from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

import copy
import itertools

def identify_resources(grid):
    """
    Identify all resource locations in the grid.
    
    Args:
        grid: 2D grid representation
    
    Returns:
        Dictionary with resource types as keys and lists of positions as values
    """
    resources = {
        "onion": [],  # O
        "dish": [],   # D
        "serving": [], # S
        "pot": []     # P
    }
    
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == "O":
                resources["onion"].append((i, j))
            elif grid[i][j] == "D":
                resources["dish"].append((i, j))
            elif grid[i][j] == "S":
                resources["serving"].append((i, j))
            elif grid[i][j] == "P":
                resources["pot"].append((i, j))
    
    return resources

def generate_masked_grids(grid):
    resources = identify_resources(grid)
    all_masked_grids = []
    
    # For each resource type, generate all possible combinations of keeping at least one
    for resource_type, positions in resources.items():
        if len(positions) <= 1:
            continue

        for keep_count in range(1, len(positions)):
            for kept_positions in itertools.combinations(positions, keep_count):
        
                masked_grid = copy.deepcopy(grid)        
                for pos in positions:
                    if pos not in kept_positions:
                        masked_grid[pos[0]][pos[1]] = "X"
                
                all_masked_grids.append({
                    "grid": masked_grid,
                    "masked_resource": resource_type,
                    "remaining_positions": kept_positions
                })
    
    # Generate combinations where multiple resource types are masked simultaneously
    # Get all unique resource types with more than one instance
    resource_types_to_mask = [r_type for r_type, positions in resources.items() if len(positions) > 1]
    
    # If we have multiple resource types that can be masked
    if len(resource_types_to_mask) > 1:
        for r_combo_size in range(2, len(resource_types_to_mask) + 1):
            for r_types_combo in itertools.combinations(resource_types_to_mask, r_combo_size):
                resource_position_combos = {}
                for r_type in r_types_combo:
                    positions = resources[r_type]
                    resource_position_combos[r_type] = []
                    for keep_count in range(1, len(positions)):
                        for kept_positions in itertools.combinations(positions, keep_count):
                            resource_position_combos[r_type].append(kept_positions)
                
                combo_keys = list(resource_position_combos.keys())
                for combo in itertools.product(*(resource_position_combos[key] for key in combo_keys)):
                    keep_map = {combo_keys[i]: combo[i] for i in range(len(combo_keys))}
                
                    masked_grid = copy.deepcopy(grid)
                    for r_type, kept_positions in keep_map.items():
                        for pos in resources[r_type]:
                            if pos not in kept_positions:
                                # Replace the resource with a space
                                masked_grid[pos[0]][pos[1]] = "X"
                    
                    all_masked_grids.append({
                        "grid": masked_grid,
                        "masked_resources": combo_keys,
                        "remaining_positions": keep_map
                    })
    
    return all_masked_grids

def grid_to_string(grid):
    """Convert a grid to a formatted string."""
    return "\n".join("".join(row) for row in grid)

def main(layout_name):
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    grid = mdp.terrain_mtx

    masked_grids = generate_masked_grids(grid)
    
    print(f"\nGenerated {len(masked_grids)} masked grid variations:")
    
    # for i, masked_info in enumerate(masked_grids):
    #     print(f"\nVariation {i+1}:")
    #     if "masked_resource" in masked_info:
    #         # Single resource type masked
    #         print(f"Masked resource type: {masked_info['masked_resource']}")
    #         print(f"Remaining positions: {masked_info['remaining_positions']}")
    #     else:
    #         # Multiple resource types masked
    #         print(f"Masked resource types: {masked_info['masked_resources']}")
    #         print(f"Remaining positions: {dict((r_type, list(pos)) for r_type, pos in masked_info['remaining_positions'].items())}")
        
    #     print("Grid:")
    #     print(grid_to_string(masked_info['grid']))
    
    # Count by resource type
    # count_by_type = {}
    # for masked_info in masked_grids:
    #     if "masked_resource" in masked_info:
    #         r_type = masked_info['masked_resource']
    #         count_by_type[r_type] = count_by_type.get(r_type, 0) + 1
    #     else:
    #         combo_key = "-".join(sorted(masked_info['masked_resources']))
    #         count_by_type[combo_key] = count_by_type.get(combo_key, 0) + 1
    # print("\nCount by resource type:")
    # for r_type, count in count_by_type.items():
    #     print(f"{r_type}: {count} variations")

if __name__ == "__main__":
    layout_name = 'c1'
    main(layout_name=layout_name)