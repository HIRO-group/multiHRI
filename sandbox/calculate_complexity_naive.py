from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


def find_all_paths(grid, start, end, path=None, all_paths=None):
    if path is None:
        path = []
    if all_paths is None:
        all_paths = []

    new_path = path.copy()
    new_path.append(start)

    if start == end:
        all_paths.append(new_path)
        return

    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    for dx, dy in directions:
        new_row, new_col = start[0] + dx, start[1] + dy

        if (0 <= new_row < len(grid) and
            0 <= new_col < len(grid[0]) and
            grid[new_row][new_col] != "X" and
            (new_row, new_col) not in new_path):

            find_all_paths(grid, (new_row, new_col), end, new_path, all_paths)
    return all_paths


def get_resource_locations(grid):
    resources = {
        "onion": [],
        "dish": [],
        "serving": [],
        'pot': []
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

def find_all_resource_paths(grid):
    resources = get_resource_locations(grid)
    all_resource_paths = {
        "onion_to_pot": [],
        "pot_to_dish": [],
        "pot_to_serving": [],
        "serving_to_onion": [],
        "serving_to_dish": [],
    }

    # Find all paths from onions to pot
    for onion_pos in resources["onion"]:
        for pot_pos in resources["pot"]:
            paths = find_all_paths(grid, onion_pos, pot_pos)
            if paths:
                all_resource_paths["onion_to_pot"].extend(paths)

    # Find all paths from pot to dish
    for pot_pos in resources["pot"]:
        for dish_pos in resources["dish"]:
            paths = find_all_paths(grid, pot_pos, dish_pos)
            if paths:
                all_resource_paths["pot_to_dish"].extend(paths)

    # Find all paths from pot to serving station
    for pot_pos in resources["pot"]:
        for serving_pos in resources["serving"]:
            paths = find_all_paths(grid, pot_pos, serving_pos)
            if paths:
                all_resource_paths["pot_to_serving"].extend(paths)


    # Find all paths from serving station to onions
    for serving_pos in resources["serving"]:
        for onion_pos in resources["onion"]:
            paths = find_all_paths(grid, serving_pos, onion_pos)
            if paths:
                all_resource_paths["serving_to_onion"].extend(paths)

    # Find all paths from serving station to dish
    for serving_pos in resources["serving"]:
        for dish_pos in resources["dish"]:
            paths = find_all_paths(grid, serving_pos, dish_pos)
            if paths:
                all_resource_paths["serving_to_dish"].extend(paths)
    return all_resource_paths


def format_path(path):
    """Format a path into a readable string."""
    return " -> ".join([f"({r}, {c})" for r, c in path])


def main(layout_name):
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    grid = mdp.terrain_mtx

    resource_paths = find_all_resource_paths(grid)

    print(f"{layout_name} resource paths:")

    # for key, value in resource_paths.items():
    #     print(f"{key} paths, len: ", len(value))
    #     for i, path in enumerate(value):
    #         print(f"  Path {i+1}: {format_path(path)}")

    total_paths = sum(len(paths) for paths in resource_paths.values())
    print(f"{layout_name}, Total paths: {total_paths} \n")


if __name__ == "__main__":
    layout_name = 'cramped_room'
    main(layout_name=layout_name)

    # layout_name = 'coordination_ring'
    # main(layout_name=layout_name)

    # layout_name = 'counter_circuit'
    # main(layout_name=layout_name)

    # layout_name = 'storage_room'
    # main(layout_name=layout_name)

    # layout_name = 'secret_heaven'
    # main(layout_name=layout_name)
