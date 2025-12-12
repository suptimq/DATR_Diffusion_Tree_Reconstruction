import os
import json
import shutil
import random
import numpy as np

from utils import load_config, normalize_path_group, taper_radius


def generate_new_branch(
    data,
    base_folder,
    num_closest_branch1=40,
    num_closest_branch2=10,
    radius_base_perturb=-0.001,
    length_base_scale=1.2,
    radius_decrease_factor=0.5,
    length_decrease_factor=0.5,
    max_tail_radius=0.003,
):
    """
    Generates a new branch based on the existing branches and saves it as a JSON file.

    This function takes a dictionary `data` containing information about existing branches, including their centers,
    radii, heights, and directions. It randomly selects a height for the new branch and finds the N1 closest branches
    based on height difference. From these, it randomly selects N2 branches for averaging and 1 branch for
    the direction of the new branch. Using weighted averaging, it computes the centers and radii of the new branch.

    Next, it determines the rotation needed to align the direction of the new branch with the vector [1, 1, 1]. Using
    Rodrigues' rotation formula, it applies this rotation to the new branch centers.

    Finally, it saves the new branch data as a JSON file in the specified `base_folder`, with a filename indicating
    the height of the branch.

    Args:
        data (dict): A dictionary containing information about existing branches.
                     Expected keys: 'all_centers', 'center_radii', 'height', 'directions'.
        base_folder (str): The base folder where the new branch JSON file will be saved.
        num_closest_branch1 (int, optional): The number of closest branches to consider based on height difference.
        num_closest_branch2 (int, optional): The number of branches to randomly select for averaging.
        radius_base_perturb (float, optional): Base value for radius perturbation.
        length_base_scale (float, optional): Base value for length scaling.
        radius_decrease_factor (float, optional): Decrease factor for radius scaling based on height.
        length_decrease_factor (float, optional): Decrease factor for length scaling based on height.
        max_tail_radius (float, optional): Maximum radius for tapering the branch radii.
    Returns:
        str: The path to the saved JSON file containing the new branch data.
    """
    # Generate a random height
    random_height = np.random.uniform(min(data["height"]), max(data["height"]))

    # Calculate height-based scaling factors
    normalized_height = (random_height - min(data["height"])) / (
        max(data["height"]) - min(data["height"])
    )
    # If base perturbation > 0 then top branches should perturb less
    if radius_base_perturb > 0:
        radius_scale_factor = 1.0 - (radius_decrease_factor * normalized_height)
    # If base perturbation <= 0 then top branches should perturb more
    else:
        radius_scale_factor = 1.0 + (radius_decrease_factor * normalized_height)

    length_scale_factor = 1.0 - (length_decrease_factor * normalized_height)

    # Find the closest branches
    height_diff = np.abs(np.array(data["height"]) - random_height)
    closest_indices = np.argsort(height_diff)[:num_closest_branch1]

    # Randomly select num_closest_branch2 out of these num_closest_branch1 branches for averaging
    selected_indices_for_averaging = random.sample(
        list(closest_indices), num_closest_branch2
    )

    # Randomly select 1 out of these 40 branches for direction
    selected_index_for_direction = random.choice(closest_indices)

    # Calculate weights (inverse of height difference)
    epsilon = 1e-6
    weights = 1 / (height_diff[selected_indices_for_averaging] + epsilon)
    weights /= weights.sum() / num_closest_branch2  # Normalize to sum up to 10

    # Initialize new branch centers and radii
    num_points = len(
        data["all_centers"][0]
    )  # Assuming all branches have the same number of points
    new_branch_centers = np.zeros((num_points, 3))
    new_branch_radii = np.zeros(num_points)
    # Compute weighted averages
    for i in range(num_points):
        for idx, weight in zip(selected_indices_for_averaging, weights):
            new_branch_centers[i] += weight * np.array(data["all_centers"][idx][i])
            new_branch_radii[i] += weight * data["center_radii"][idx][i]
        new_branch_centers[i] /= weights.sum()
        new_branch_radii[i] /= weights.sum()

    # Radius perturbation
    perturb_radius = radius_base_perturb * radius_scale_factor
    # Avoid negative branch radii
    if perturb_radius < 0 and abs(perturb_radius) > new_branch_radii.min():
        perturb_radius = -abs(new_branch_radii.min() - epsilon)
    new_branch_radii += perturb_radius
    new_branch_radii = np.maximum(new_branch_radii, epsilon)  # Avoid negative radii

    # Determine the rotation needed to align [1, 1, 1] to the new direction
    original_direction = np.array([1, 1, 1]) / np.sqrt(3)
    new_direction = np.array(data["directions"][selected_index_for_direction])
    new_direction /= np.linalg.norm(new_direction)

    # Calculate axis and angle for rotation
    axis = np.cross(original_direction, new_direction)
    axis_length = np.linalg.norm(axis)
    if axis_length != 0:
        axis /= axis_length
    angle = np.arccos(np.clip(np.dot(original_direction, new_direction), -1.0, 1.0))

    # Rodrigues' rotation formula
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    one_minus_cos = 1 - cos_angle
    axis_outer = np.outer(axis, axis)
    rotation_matrix = (
        cos_angle * np.eye(3)
        + sin_angle * np.cross(np.eye(3), axis)
        + one_minus_cos * axis_outer
    )

    # Apply rotation to each point in new branch centers
    rotated_branch_centers = np.dot(new_branch_centers, rotation_matrix.T)

    # Scale the branch centers to increase the branch length
    length_scale = length_base_scale * length_scale_factor
    branch_base = rotated_branch_centers[0]  # The base point of the branch
    scaled_branch_centers = (
        branch_base + (rotated_branch_centers - branch_base) * length_scale
    )

    # Taper branch radius to avoid short and thick branches
    new_branch_radii = taper_radius(
        scaled_branch_centers, new_branch_radii, max_tail_radius=max_tail_radius
    )

    # Save the new branch as a JSON file
    new_branch_data = {
        "all_centers": [scaled_branch_centers.tolist()],
        "center_radii": [new_branch_radii],
        "height": [random_height],
        "direction": new_direction.tolist(),
    }
    output_file_name = os.path.join(base_folder, f"Branch_at_{random_height:.2f}.json")
    with open(output_file_name, "w") as new_file:
        json.dump(new_branch_data, new_file, indent=4)

    return output_file_name


if __name__ == "__main__":

    config_filepath = os.path.join("Config", "config.yaml")
    config = load_config(config_filepath)

    # Main folder path
    data_folder = config["experiment"]["save_folder"]

    # Branch interpolation parameters
    num_closest_branch1 = config["parameters"]["num_closest_branch1"]
    num_closest_branch2 = config["parameters"]["num_closest_branch2"]
    radius_perturb_range = config["parameters"]["radius_perturb_range"]
    length_scale_range = config["parameters"]["length_scale_range"]
    num_new_branches = config["parameters"]["num_new_branches"]
    num_new_trees = config["parameters"]["num_new_trees"]
    max_branch_tail_radius = config["parameters"]["max_branch_tail_radius"]

    branch_json_folder = os.path.join(data_folder, "meta", "json", "branch")
    output_folder = os.path.join(
        data_folder, f"interpolation_num-new-tree-{num_new_trees}"
    )
    new_branch_folder = os.path.join(output_folder, f"{num_new_branches}_new_branches")
    output_json_file_name = os.path.join(
        output_folder, r"combined_normalized_branches.json"
    )

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(new_branch_folder, exist_ok=True)

    shutil.copyfile(config_filepath, os.path.join(output_folder, "config.yaml"))

    all_path_groups = []
    all_radii = []
    all_heights = []
    all_directions = []  # List to store direction vectors

    for root, dirs, files in os.walk(branch_json_folder):

        for file in files:
            if file.endswith(".json"):
                json_file_path = os.path.join(root, file)
                with open(json_file_path, "r") as f:
                    data = json.load(f)
                centers = data.get("centers", [])
                center_radii = data.get("center_radii", [])
                for path_group, radii in zip(centers, center_radii):
                    # Calculate and normalize the direction vector
                    start_point, end_point = np.array(path_group[0]), np.array(
                        path_group[-1]
                    )
                    direction_vector = end_point - start_point
                    direction_vector /= np.linalg.norm(direction_vector)
                    all_directions.append(direction_vector.tolist())

                    normalized_path_group = normalize_path_group(path_group)

                    x0, y0, z0 = path_group[0]
                    height = np.sqrt(x0**2 + y0**2 + z0**2)
                    all_heights.append(height)

                    all_path_groups.append(normalized_path_group)
                    all_radii.append(radii)

    print(f"Processed {len(all_path_groups)} path groups from JSON files.")

    data_dict = {
        "all_centers": all_path_groups,
        "center_radii": all_radii,
        "height": all_heights,
        "directions": all_directions,  # Include directions in the JSON
    }

    # save data to json
    json_content = json.dumps(data_dict, indent=4)

    # Split and reformat the JSON string to put each key on a new line
    lines = json_content.split("\n")
    formatted_lines = []
    for line in lines:
        if '"all_centers":' in line or '"center_radii":' in line or '"height":' in line:
            formatted_lines.append("\n" + line)
        else:
            formatted_lines.append(line)
    formatted_json_content = "\n".join(formatted_lines)

    with open(output_json_file_name, "w") as f:
        f.write(formatted_json_content)

    # Generate multiple branches for each new tree
    for i in range(1, num_new_trees + 1):
        new_tree_branch_folder = os.path.join(new_branch_folder, f"tree{i}")
        os.makedirs(new_tree_branch_folder, exist_ok=True)

        # Randomly generate base values and taper values for all branches in the same tree
        radius_base_perturb = random.uniform(*radius_perturb_range)
        length_base_scale = random.uniform(*length_scale_range)
        radius_decrease_factor = random.uniform(0, 1)
        length_decrease_factor = random.uniform(0, 1)

        # Generate multiple branches
        for _ in range(int(num_new_branches)):
            max_tail_radius = random.uniform(*max_branch_tail_radius)
            new_branch_file = generate_new_branch(
                data_dict,
                new_tree_branch_folder,
                num_closest_branch1=num_closest_branch1,
                num_closest_branch2=num_closest_branch2,
                radius_base_perturb=radius_base_perturb,
                length_base_scale=length_base_scale,
                radius_decrease_factor=radius_decrease_factor,
                length_decrease_factor=length_decrease_factor,
                max_tail_radius=max_tail_radius,
            )
            print(f"New branch saved as {new_branch_file}")
