import os
import json
import numpy as np
import random
import multiprocessing

from utils import (
    load_config,
    taper_radius,
    compute_branch_direction,
    spline_branch_internode_ratios,
    rotate_points,
)


def generate_new_trunk(
    json_filepath,
    data_folder=".",
    json_filename="new_trunk.json",
    num_src_trunk=5,
    num_branch_range=[30, 50],
    max_tail_radius=0.006,
):
    """
    Generate a new trunk based on selected trunks from a JSON file, assign random weights,
    and compute a weighted average of trunk points and branch internode ratios.

    Args:
        json_filepath (str): Path to the JSON file containing trunk data.
        data_folder (str, optional): Path to the folder to save the new trunk JSON file. Defaults to ".".
        json_filename (str, optional): Filename of the new trunk JSON file. Defaults to "new_trunk.json".
        num_src_trunk (int, optional): The number of existing trunks used for interpolation.
        num_branch_range (list, optional): Range for branches attached to each trunk.
        max_tail_radius (float, optional): Maximum radius for tapering the trunk radius. Defaults to 0.006.
    Returns:
        str: Path to the saved new trunk JSON file.
    """
    # Load data from the JSON file
    with open(json_filepath, "r") as file:
        data = json.load(file)
        all_trunk_points = data["all_trunk_points"]
        all_trunk_radii = data["all_trunk_radii"]
        all_branch_internode_ratios = data["all_branch_internode_ratios"]

    # Randomly select 5 trunks
    selected_indices = random.sample(range(len(all_trunk_points)), num_src_trunk)

    # Assign random weights to the selected trunks
    weights = np.random.rand(num_src_trunk)
    weights /= weights.sum()  # Normalize the weights to sum up to 1

    # Initialize new trunk points
    num_points = len(
        all_trunk_points[0]
    )  # Assuming all trunks have the same number of points
    new_trunk_points = np.zeros((num_points, 3))

    # Process branch internode ratios
    new_branch_internode_ratios = []
    max_length = 0
    for idx in selected_indices:
        random_num_points = random.randint(*num_branch_range)
        splined_ratios = spline_branch_internode_ratios(
            all_branch_internode_ratios[idx], random_num_points
        )
        new_branch_internode_ratios.append(splined_ratios)
        max_length = max(max_length, len(splined_ratios))

    # Pad shorter lists to match max_length
    padded_branch_internode_ratios = []
    for ratios in new_branch_internode_ratios:
        padded_ratios = np.pad(ratios, (0, max_length - len(ratios)), "edge")
        padded_branch_internode_ratios.append(padded_ratios)

    # Compute weighted averages for points and internode ratios
    for i in range(num_points):
        for idx, weight in zip(selected_indices, weights):
            new_trunk_points[i] += weight * np.array(all_trunk_points[idx][i])
        new_trunk_points[i] /= weights.sum()

    averaged_branch_internode_ratios = np.average(
        padded_branch_internode_ratios, axis=0, weights=weights
    ).tolist()

    # Choose a radius randomly from one of the selected trunks
    new_trunk_radius = all_trunk_radii[random.choice(selected_indices)]

    # Taper branch radius to avoid short and thick branches
    new_trunk_radius = taper_radius(
        new_trunk_points, new_trunk_radius, max_tail_radius=max_tail_radius
    )

    # Save the new trunk as a JSON file
    new_trunk_data = {
        "trunk_points": new_trunk_points.tolist(),
        "trunk_radius": new_trunk_radius,
        "branch_internode_ratios": averaged_branch_internode_ratios,
    }
    output_filename = os.path.join(data_folder, json_filename)
    with open(output_filename, "w") as new_file:
        json.dump(new_trunk_data, new_file, indent=4)

    return output_filename


def add_branches_to_trunk(
        trunk_data, 
        new_branch_folder, 
        height_buffer=0.2, 
        topk=5,
        max_bottom_ratio=0.6, 
        min_top_ratio=0.05, 
        length_decay_factor=1.2):
    """
    Augments a trunk with additional branches loaded from JSON files.

    This function enhances a given trunk by appending additional branches from JSON files stored in a specified folder.
    It iterates through the branch internode ratios (i.e., branch height) of the trunk to determine the target height for each branch addition.
    For each target height, it searches for the closest branch file with a height less than or equal to the target height and loads the branch data.
    The function then finds the origin point on the trunk closest to or above the target height and rotates the branch points in the x-y plane by a random angle around this origin point.
    The adjusted and rotated branch points, along with their radii, are added to the trunk data.

    Args:
        trunk_data (dict): Dictionary containing information about the trunk, including trunk points and averaged branch internode ratios.
        new_branch_folder (str): Path to the folder containing JSON files storing data of additional branches.
        height_buffer (float): Buffer value for selecting branch files close to the target height (default is 0.2).
        topk (int): Number of top values of closest branches in terms of height.
        max_bottom_ratio (float): Max length for bottom branches (ratio of max length).
        min_top_ratio (float): Min length for top branches (ratio of max length).
        length_decay_factor (float): Controls how quickly length decreases with height.
            - 0-1 for slower decrease.
            - >1 for faster decrease.
    Returns:
        dict: Dictionary containing the updated trunk data with added branch points and radii.
    """
    trunk_points = trunk_data["trunk_points"]
    trunk_radius = trunk_data["trunk_radius"]
    averaged_branch_internode_ratios = trunk_data["branch_internode_ratios"]
    branch_points = []
    branch_points_radius = []

    zmin = trunk_points[0][2]  # First z-coordinate in trunk_points
    zmax = trunk_points[-1][2]  # Last z-coordinate in trunk_points
    trunk_height = zmax - zmin

    # Skip the bottom-most branch
    averaged_branch_internode_ratios = averaged_branch_internode_ratios[1:]
    for ci in averaged_branch_internode_ratios:
        # Find the closest branch files
        target_height = ci * zmax
        branch_files = []
        height_diffs = []
        for file_name in os.listdir(new_branch_folder):
            if file_name.endswith(".json"):
                branch_height = float(file_name.replace(".json", "").split("_")[-1])
                if (
                    target_height - height_buffer
                    <= branch_height
                    <= target_height + height_buffer
                ):
                    branch_files.append(os.path.join(new_branch_folder, file_name))
                    height_diffs.append(abs(branch_height - target_height))

        # Skip if no branches found
        if not branch_files:
            continue

        # Select top-k closest branch heights
        top_k_indices = np.argsort(height_diffs)[:topk]
        closest_branch_file = random.choice([branch_files[i] for i in top_k_indices])
        assert os.path.exists(closest_branch_file), f"{closest_branch_file} Not Found"

        with open(closest_branch_file, "r") as file:
            branch_data = json.load(file)
            all_centers = branch_data["all_centers"][0]
            center_radii = branch_data["center_radii"][0]

        all_centers = np.array(all_centers)
        center_radii = np.array(center_radii)

        # Find the origin point on the trunk closest to or above the target height
        trunk_z_values = np.array(trunk_points)[:, 2]
        possible_indices = np.where(trunk_z_values >= target_height)[0]
        closest_z_index = (
            possible_indices[0]
            if len(possible_indices) > 0
            else np.argmax(trunk_z_values)
        )
        origin_point = trunk_points[closest_z_index]
        origin_radius = trunk_radius[closest_z_index]

        # Make sure the branch radius is smaller than the trunk radius
        while center_radii[0] >= origin_radius:
            center_radii = center_radii * 0.9

        ### Add length constraint based on branch height to avoid super
        ### long branches on top
        # Compute branch length as accumulated distance between centers
        branch_length = np.sum(np.linalg.norm(all_centers[1:] - all_centers[:-1], axis=1))
        # Height-based length constraint
        height_ratio = (target_height - zmin) / trunk_height
        max_allowed_length = trunk_height * (
            min_top_ratio + 
            (max_bottom_ratio - min_top_ratio) * (1 - height_ratio)**length_decay_factor
        )

        # Scale branch if it exceeds max_allowed_length
        if branch_length > max_allowed_length:
            scale_factor = max_allowed_length / branch_length
            all_centers = all_centers[0] + (all_centers - all_centers[0]) * scale_factor
            # center_radii *= scale_factor ** 0.5  # Roughly preserve volume

        # Randomize the rotation angle for the branch in the x-y plane
        rotation_angle = random.uniform(0, 2 * np.pi)  # Random angle in radians
        # Adjust and rotate branch points by the origin point
        adjusted_branch_points = [
            [x + origin_point[0], y + origin_point[1], z + origin_point[2]]
            for [x, y, z] in all_centers
        ]
        rotated_branch_points = rotate_points(
            adjusted_branch_points, rotation_angle, origin=origin_point
        )

        # Remove branches that are lower than trunk root
        branch_z = np.asarray(rotated_branch_points)[:, -1]
        branch_zmin = branch_z.min()
        if branch_zmin >= zmin:
            branch_points.append(rotated_branch_points)
            branch_points_radius.append(center_radii.tolist())

    # Add branch data to the trunk data
    trunk_data["branch_points"] = branch_points
    trunk_data["branch_points_radius"] = branch_points_radius

    return trunk_data


def overlapping_detection(branch_point, branch_radii, other_branch, other_radii):
    """
    Detects overlap between two branches based on their points and radii.

    This function calculates the distances between each pair of points from the
    current branch and the other branch, and then computes the sum of radii for
    each pair of points. It checks if any overlap is found between the radii
    and distances, and returns True if overlap is detected, otherwise returns False.

    Args:
        branch_point (numpy.ndarray): Array of points representing the current branch.
        branch_radii (numpy.ndarray): Array of radii corresponding to the points of the current branch.
        other_branch (numpy.ndarray): Array of points representing the other branch.
        other_radii (numpy.ndarray): Array of radii corresponding to the points of the other branch.

    Returns:
        bool: True if overlap is detected, False otherwise.
    """

    branch_point = np.array(branch_point)
    branch_radii = np.array(branch_radii)
    other_branch = np.array(other_branch)
    other_radii = np.array(other_radii)

    # Calculate distances between points of current branch and other branch
    distances = np.sqrt(
        np.sum((branch_point[:, np.newaxis, :] - other_branch) ** 2, axis=2)
    )
    # Calculate sum of radii for each pair of points
    radii_sum = branch_radii[:, np.newaxis] + other_radii
    # Check for overlap between radii and distances
    overlap_mask = distances - radii_sum < 0
    # Check if any overlap is found
    overlap_found = np.any(overlap_mask)

    return overlap_found


def remove_overlapping_branches(
    branch_points,
    branch_points_radii,
    other_branch_points,
    other_branch_points_radii,
    same_branches=False,
    pc_branch=True,
    primary_branch=True,
    num_trial=2,
):
    """
    Remove overlapping branches from a list of branch points and their corresponding radii.

    This function iterates through each branch point and checks for overlap with other branches.
        1. branch_points == other_branch_points are primary branches, None value is not allowed in the input.
            and once an overlap is detected, the particular branch is removed from the branch_points.
        2. branch_points are children branches (== or != other_branch_points that could be primary branches),
            None value is allowed in the input (as it needs to align with the primary branch points).
            and once an overlap is detected, the particular branch is marked as None in the branch_points.

    Args:
        branch_points (list of list): List containing arrays of branch points.
        branch_points_radii (list of list): List containing arrays of branch radii.
        other_branch_points (list of list): Array of points representing the other branch.
        other_branch_points_radii (list of list): Array of radii corresponding to the points of the other branch.
        same_branches (bool): Flag to indicate if the branches are the same.
        pc_branch (bool): Flag to indicate if branch_points are direct children of other_branch_points.
        primary_branch (bool): Flag to indicate if branch_points are primary branches.
        num_trial (int): Number of trials the branch could be rotated to avoid collision.

    Returns:
        tuple: A tuple containing two lists:
               - A list of final non-overlapping branch points.
               - A list of final non-overlapping branch points' radii.
    """
    num_branches = len(branch_points)
    new_branch_points = []
    new_branch_points_radius = []
    for i, (branch_point, branch_radii) in enumerate(
        zip(branch_points, branch_points_radii)
    ):

        if branch_point is None:
            new_branch_points.append(None)
            new_branch_points_radius.append(None)
            continue

        start_idx = i + 1 if same_branches else 0
        overlap_found = False  # in case other_branch_points are None
        # Iterate over all other branches to check for overlap
        for j, (other_branch, other_radii) in enumerate(
            zip(other_branch_points[start_idx:], other_branch_points_radii[start_idx:]),
            start=start_idx,
        ):

            # Skip the check of the first 10 points (VERY likely to collide) with other_branch
            # when the current branch is the child of other_branch
            # Prevent the current branch go through other_branch
            skip_junction_idx = 10 if i == j and pc_branch else 0

            if other_branch is None:
                continue

            counter = 0
            overlap_found = overlapping_detection(
                branch_point[skip_junction_idx:],
                branch_radii[skip_junction_idx:],
                other_branch,
                other_radii,
            )
            # Rotate the branch to avoid collision if overlap is found
            while overlap_found and counter < num_trial:
                counter = counter + 1
                # Randomize the rotation angle for the branch in the x-y plane
                rotation_angle = random.uniform(0, 2 * np.pi)  # Random angle in radians
                branch_point = rotate_points(
                    branch_point, rotation_angle, origin=branch_point[0]
                )
                overlap_found = overlapping_detection(
                    branch_point[skip_junction_idx:],
                    branch_radii[skip_junction_idx:],
                    other_branch,
                    other_radii,
                )

            if overlap_found:
                break

        if overlap_found and not primary_branch:
            new_branch_points.append(None)
            new_branch_points_radius.append(None)

        if not overlap_found:
            new_branch_points.append(branch_point)
            new_branch_points_radius.append(branch_radii)

    if not primary_branch:
        assert (
            len(new_branch_points) == num_branches
        ), f"Inconsistent branch# after collision removal!"

    return new_branch_points, new_branch_points_radius


def add_branches(
    branch_points,
    branch_points_radius,
    trunk_points,
    trunk_radius,
    top_k=5,
    prob_add_branch=0.2,
    r_scale=0.5,
    s_scale=0.5,
    new_origin_index_range=[40, 60],
):
    """
    This function adds higher order branches (e.g., second-order, third-order branches) to a set of parent branches based on specified criteria.

    It enhances a given set of parent branches by generating additional higher-order branches.
    It selects parent branches based on a given probability and then identifies suitable child branches
    for growing. The orientation of the child branch was determined by randomly selecting from the top-k least-similar
    existing branches.

    Args:
        branch_points: List of lists representing the points of parent branches.
        branch_points_radius: List of lists representing the radii of parent branches.
        trunk_points: List of lists representing the points of the trunk.
        trunk_radius: List of lists representing the radii of the trunk.
        top_k: Number of top values of least similar branches in terms of orientation.
        prob_add_branch: Probability of adding a branch to each parent branch.
        r_scale: Scaling factor for adjusting the radii of branches.
        s_scale: Scaling factor for adjusting the size of branches.
        new_origin_index_range: Range for randomly selecting a point from the parent branch.
    Returns:
        add_branch_points: List of list representing the points of the generated branch.
        add_branch_points_radius: List of list representing the radii of the generated branch.
    """
    original_branch_count = len(branch_points)  # Store the original count of branches
    add_branch_points = list()  # Create a new list to store the modified branches
    add_branch_points_radius = list()  # Similarly, for radii

    # List to keep track of indices where secondary branches will be added
    indices_for_secondary_branches = []

    # TODO Optimization using ndarray
    for i in range(original_branch_count):
        # Check parent branch existence
        if branch_points[i] is None:
            continue
        if random.random() < prob_add_branch:
            indices_for_secondary_branches.append(i)
    for i in range(original_branch_count):

        if i in indices_for_secondary_branches:
            parent_branch = branch_points[i]
            parent_branch_radius = branch_points_radius[i][0]  # 1st skeleton's radius
            parent_branch_length = np.sum(
                np.linalg.norm(np.diff(parent_branch, axis=0), axis=1)
            )
            # Calculate source branch direction
            source_direction = compute_branch_direction(parent_branch)

            # Only consider original branches for finding the child branch
            similarities = np.ones(original_branch_count)
            for j in range(original_branch_count):
                if (i != j) and (branch_points[j] is not None):
                    # TODO Fix the corner case for circular branches
                    # Calculate target branch direction
                    target_direction = compute_branch_direction(branch_points[j])
                    dot_product = np.dot(source_direction, target_direction)
                    # Exclude branches growing towards the opposite direction
                    similarities[j] = (
                        abs(dot_product) if 0 < dot_product <= 0.8 else np.inf
                    )

            # Randomly choose the chlid branch from branches that are Top-K unlikely to intersect
            top_k_indices = np.argsort(similarities)[:top_k]
            random.shuffle(top_k_indices)

            for child_branch_index in top_k_indices:
                # child_branch_index = random.choice(top_k_indices)
                # assert child_branch_index is not None, "Child Branch Generation Failure"

                # Copy and modify the child branch
                child_branch = np.array(branch_points[child_branch_index])
                child_branch_radii = np.array(branch_points_radius[child_branch_index])

                # Normalize to origin and scale
                child_branch -= child_branch[0]
                child_branch *= s_scale
                child_branch_radii *= r_scale
                # Compute the length of the child branch
                child_length = np.sum(
                    np.linalg.norm(np.diff(child_branch, axis=0), axis=1)
                )
                # Make sure the child branch is shorter than the parent branch
                while child_length >= parent_branch_length:
                    child_branch *= s_scale
                    child_length = np.sum(
                        np.linalg.norm(np.diff(child_branch, axis=0), axis=1)
                    )
                # Make sure the child branch is smaller than the parent branch
                while child_branch_radii[0] >= parent_branch_radius:
                    child_branch_radii *= r_scale

                # Select a random point from the source branch as the new origin
                new_origin_index = random.randint(
                    new_origin_index_range[0],
                    min(new_origin_index_range[1], len(parent_branch) - 1),
                )
                new_origin = np.array(parent_branch[new_origin_index])
                # Translate and scale radii
                child_branch += new_origin

                # Check for overlap with the trunk
                overlap_found = overlapping_detection(
                    child_branch, child_branch_radii, trunk_points, trunk_radius
                )

                if not overlap_found:
                    break

            if overlap_found:
                add_branch_points.append(None)
                add_branch_points_radius.append(None)
            else:
                # Add the new branch and its radii
                add_branch_points.append(child_branch.tolist())
                add_branch_points_radius.append(child_branch_radii.tolist())
        else:
            add_branch_points.append(None)
            add_branch_points_radius.append(None)

    return add_branch_points, add_branch_points_radius


## TODO separte this function into remove_collision and grow_branches
def remove_collision(
    data_folder,
    save_folder,
    idx,
    num_trial=2,
    branch_level=1,
    top_k=5,
    prob_add_branch=[0.2],
    r_scales=[0.5],
    s_scales=[0.5],
    new_origin_index_range=[40, 60],
):
    """
    This function reads a JSON file containing information about a tree's branches, detects collisions among these branches,
    and removes overlapping branches to ensure a collision-free structure.
    It then adds higher-order branches based on specified criteria to enhance the complexity of the tree.
    The modified tree data is saved back to a JSON file.

    Args:
        json_filepath (str): Path to the JSON file.
        save_folder (str): Folder to save the updated JSON file.
        idx (int): Index of the tree.
        num_trial (int): Number of trials the branch could be rotated to avoid collision.
        branch_level (int): Level of branches that are added to the parent branch (default is 1, meaning second-order branch).
        top_k: Number of top values of least similar branches in terms of orientation.
        prob_add_branch: Probability of adding a branch to each parent branch.
        r_scale: Scaling factor for adjusting the radii of branches.
        s_scale: Scaling factor for adjusting the size of branches.
        new_origin_index_range: Range for randomly selecting a point from the parent branch.
    """
    print(
        "-------------------------------Branch Collision Detection-------------------------------"
    )
    json_filepath = os.path.join(data_folder, f"tree{idx}_tmp.json")
    assert os.path.exists(json_filepath), f"{json_filepath} Not Found"
    print(f"Processing {json_filepath}")
    with open(json_filepath, "r") as f:
        new_trunk_with_branches = json.load(f)
    branch_points = new_trunk_with_branches["branch_points"]
    branch_points_radius = new_trunk_with_branches["branch_points_radius"]

    trunk_points = new_trunk_with_branches["trunk_points"]
    trunk_radius = new_trunk_with_branches["trunk_radius"]

    # Collision check for primary branches
    new_branch_points, new_branch_points_radius = remove_overlapping_branches(
        branch_points,
        branch_points_radius,
        other_branch_points=branch_points,
        other_branch_points_radii=branch_points_radius,
        same_branches=True,
        pc_branch=False,
        primary_branch=True,
        num_trial=num_trial,
    )
    print(
        f"Removed {len(branch_points) - len(new_branch_points)} branches due to overlap."
    )

    # Updating the data dictionary with non-overlapping branches
    new_trunk_with_branches["branch_points"] = new_branch_points
    new_trunk_with_branches["branch_points_radius"] = new_branch_points_radius

    primary_branch_points = new_branch_points
    primary_branch_points_radius = new_branch_points_radius

    if branch_level > 0:
        assert (
            len(prob_add_branch) == len(r_scales) == len(s_scales) == branch_level
        ), "Unmatched parameters for higher order branches"

    for level in range(1, branch_level + 1):

        prob = prob_add_branch[level - 1]
        r_scale = random.uniform(*r_scales[level - 1])
        s_scale = random.uniform(*s_scales[level - 1])

        print(
            f"-------------------------------{level+1}-order Branch Generation-------------------------------"
        )
        # Add higher order branches
        # Notice: new_branch_points might contain None value indicating the corresponding parent branch
        #         do not have children
        branch_points, branch_points_radius = add_branches(
            new_branch_points,
            new_branch_points_radius,
            trunk_points,
            trunk_radius,
            top_k=top_k,
            prob_add_branch=prob,
            r_scale=r_scale,
            s_scale=s_scale,
            new_origin_index_range=new_origin_index_range,
        )

        # Collision check for higher order branches
        new_branch_points, new_branch_points_radius = remove_overlapping_branches(
            branch_points,
            branch_points_radius,
            other_branch_points=primary_branch_points,
            other_branch_points_radii=primary_branch_points_radius,
            same_branches=False,
            pc_branch=True,
            primary_branch=False,
            num_trial=num_trial,
        )

        print(
            f"Removed {len(branch_points) - len(new_branch_points)} {level+1}-order branches due to overlap."
        )

        # Updating the data dictionary
        new_trunk_with_branches[f"new{level+1}_branch_points"] = new_branch_points
        new_trunk_with_branches[f"new{level+1}_branch_points_radius"] = (
            new_branch_points_radius
        )

    # Optionally, save the updated data back to a JSON file
    updated_tree_json_filepath = os.path.join(save_folder, f"tree{idx}.json")
    with open(updated_tree_json_filepath, "w") as file:
        json.dump(new_trunk_with_branches, file, indent=4)
    print(f"Saved to {updated_tree_json_filepath}")


if __name__ == "__main__":  #

    config_filepath = os.path.join("Config", "config.yaml")
    config = load_config(config_filepath)

    # Main folder path
    data_folder = config["experiment"]["save_folder"]
    num_new_trees = config["parameters"]["num_new_trees"]
    num_new_branches = config["parameters"]["num_new_branches"]

    # New trunk parameters
    num_src_trunk = config["parameters"]["num_src_trunk"]
    num_branch_range = config["parameters"]["num_branch_range"]

    # Higher order of branches parameters
    num_trial = config["parameters"]["num_trial"]
    branch_level = config["parameters"]["branch_level"]
    top_k = config["parameters"]["top_k"]
    prob_add_branch = config["parameters"]["prob_add_branch"]
    r_scales = config["parameters"]["r_scales"]
    s_scales = config["parameters"]["s_scales"]
    max_trunk_tail_radius = config["parameters"]["max_trunk_tail_radius"]
    new_origin_index_range = config["parameters"]["new_origin_index_range"]

    assert (
        len(prob_add_branch) == len(r_scales) == len(s_scales)
    ), "Inconsistent config for children branches! Make sure len(prob_add_branch) == len(r_scales) == len(s_scales)."

    new_tree_folder = os.path.join(
        data_folder, f"interpolation_num-new-tree-{num_new_trees}"
    )
    process_mode = config["parameters"]["process_mode"]
    # Specify the output JSON file path (same directory as the script)
    json_filepath = os.path.join(
        data_folder, "meta", "json", "all_trunks_combined.json"
    )
    new_branch_folder = os.path.join(
        new_tree_folder, f"{num_new_branches}_new_branches"
    )
    assert os.path.exists(new_branch_folder), f"{new_branch_folder} Not Found"

    tmp_tree_folder = os.path.join(new_tree_folder, f"{num_new_trees}_tmp")
    os.makedirs(tmp_tree_folder, exist_ok=True)

    updated_tree_folder = os.path.join(new_tree_folder, "new_tree_json")
    os.makedirs(updated_tree_folder, exist_ok=True)

    # Generate new trunk and get the file name
    for i in range(1, num_new_trees + 1):
        max_tail_radius = random.uniform(*max_trunk_tail_radius)
        new_trunk_file_name = generate_new_trunk(
            json_filepath,
            data_folder=new_branch_folder,
            json_filename=f"trunk{i}.json",
            num_src_trunk=num_src_trunk,
            num_branch_range=num_branch_range,
            max_tail_radius=max_tail_radius,
        )

        # Load the trunk data from the file
        with open(new_trunk_file_name, "r") as file:
            new_trunk_data = json.load(file)

        new_tree_branch_folder = os.path.join(new_branch_folder, f"tree{i}")
        assert os.path.exists(
            new_tree_branch_folder
        ), f"{new_tree_branch_folder} Not Found"

        # Add branches to the new trunk
        new_trunk_with_branches = add_branches_to_trunk(
            new_trunk_data, new_tree_branch_folder
        )

        # Save the new trunk with branches as a JSON file
        output_file_with_branches = os.path.join(tmp_tree_folder, f"tree{i}_tmp.json")
        with open(output_file_with_branches, "w") as new_file:
            json.dump(new_trunk_with_branches, new_file, indent=4)

        print(f"New trunk with branches saved as {output_file_with_branches}")

    ### ======================== ###
    ### Remove crossing branches ###
    ### ======================== ###
    if process_mode == "parallel":
        num_processes = multiprocessing.cpu_count()  # Use all available CPU cores
        pool = multiprocessing.Pool(processes=num_processes)

        # Use pool.map to parallelize the processing of files
        results = pool.starmap(
            remove_collision,
            [
                (
                    tmp_tree_folder,
                    updated_tree_folder,
                    i,
                    num_trial,
                    branch_level,
                    top_k,
                    prob_add_branch,
                    r_scales,
                    s_scales,
                    new_origin_index_range,
                )
                for i in range(1, num_new_trees + 1)
            ],
        )

        # Close the pool to release resources
        pool.close()
        pool.join()
    elif process_mode == "single":
        for i in range(1, num_new_trees + 1):
            remove_collision(
                data_folder=tmp_tree_folder,
                save_folder=updated_tree_folder,
                idx=i,
                num_trial=num_trial,
                branch_level=branch_level,
                top_k=top_k,
                prob_add_branch=prob_add_branch,
                r_scales=r_scales,
                s_scales=s_scales,
                new_origin_index_range=new_origin_index_range,
            )
    else:
        print(f"Invalid process_mode {process_mode}")
