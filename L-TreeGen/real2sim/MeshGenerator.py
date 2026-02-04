import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from natsort import natsorted

from utils import load_config, taper_radius
from utils import interpolate_points, interpolate_radii, calculate_direction_vectors
from lpy_utils import *
from leaf_generator import process_branch_leaves


# Load configuration from YAML file
config_filepath = os.path.join("Config", "config.yaml")
config = load_config(config_filepath)

# Determine if it's in new tree mode based on command-line argument
parser = argparse.ArgumentParser()
parser.add_argument(
    "-new_tree_mode", action="store_true", help="Generate new tree by interpolation"
)
parser.add_argument("-branch_obj", action="store_true", help="Save branch obj files.")
parser.add_argument("-trunk_obj", action="store_true", help="Save trunk obj files.")
parser.add_argument("-tree_obj", action="store_true", help="Save tree obj files.")
args = parser.parse_args()

new_tree_mode = args.new_tree_mode

# Directories setup
save_folder = config["experiment"]["save_folder"]
num_new_trees = config["parameters"]["num_new_trees"]
save_folder = (
    os.path.join(save_folder, f"interpolation_num-new-tree-{num_new_trees}")
    if new_tree_mode
    else save_folder
)
data_folder = (
    os.path.join(save_folder, "new_tree_json")
    if new_tree_mode
    else config["experiment"]["data_folder"]
)
assert os.path.exists(data_folder), f"{data_folder} Not Found"
if new_tree_mode:
    assert os.path.exists(save_folder), f"{save_folder} Not Found"

output_obj_folder = os.path.join(save_folder, "obj")
output_lpy_folder = os.path.join(save_folder, "lpy")
output_npy_folder = os.path.join(save_folder, "meta", "npy")
output_json_folder = os.path.join(save_folder, "meta", "json")
output_trunk_json_folder = os.path.join(output_json_folder, "trunk")
output_branch_json_folder = os.path.join(output_json_folder, "branch")

# Create necessary directories if they don't exist
os.makedirs(output_obj_folder, exist_ok=True)
os.makedirs(output_lpy_folder, exist_ok=True)
os.makedirs(output_npy_folder, exist_ok=True)
os.makedirs(output_json_folder, exist_ok=True)
os.makedirs(output_trunk_json_folder, exist_ok=True)
os.makedirs(output_branch_json_folder, exist_ok=True)

# Load parameters from configuration
orchard_setup = config["experiment"]["orchard_setup"]
convert_to_m = False if new_tree_mode else config["experiment"]["convert_to_m"]
taper = False if new_tree_mode else config["experiment"]["taper"]

trunk_taper = config["parameters"]["trunk_taper"]
branch_taper = config["parameters"]["branch_taper"]

branch_obj = args.branch_obj if args.branch_obj else config["parameters"]["branch_obj"]
trunk_obj = args.trunk_obj if args.trunk_obj else config["parameters"]["trunk_obj"]
tree_obj = args.tree_obj if args.tree_obj else config["parameters"]["tree_obj"]

branch_level = config["parameters"]["branch_level"]

trunk_counter = 0
branch_counter = 0
secondary_branch_counter = 0
row13_offset = 0
row15_offset = 0
row16_offset = 0
trunk_json_file_list = []
tree_skeleton_num_list = []
# {'level1': [], 'level2': [], ...}
branch_radius_stats = {}

# Recursively generate Lpy and .obj files
for root, dirs, files in os.walk(data_folder):
    files = natsorted(files)
    for file in files:
        # Each tree
        if file.endswith(".json"):
            print(f"Processing {file}")

            tree_lpy_modules = []
            tree_lpy_points = []
            tree_lpy_radii = []
            tree_lpy_leaf_data = []

            # Calculate paths
            row = os.path.basename(root)
            tree_id = os.path.splitext(file)[0]
            obj_row_dir = os.path.join(output_obj_folder, row)
            obj_tree_dir = os.path.join(obj_row_dir, tree_id)
            obj_trunk_dir = os.path.join(obj_tree_dir, "trunk")
            obj_branch_dir = os.path.join(obj_tree_dir, "branch")

            lpy_tree_dir = os.path.join(output_lpy_folder, row, tree_id)
            lpy_trunk_dir = os.path.join(lpy_tree_dir, "trunk")
            lpy_branch_dir = os.path.join(lpy_tree_dir, "branch")

            # Create directories if they don't exist
            for directory in [
                obj_trunk_dir,
                obj_branch_dir,
                lpy_trunk_dir,
                lpy_branch_dir,
            ]:
                os.makedirs(directory, exist_ok=True)

            # Add orchard setup
            x_offset = 0
            y_offset = 0
            if orchard_setup:
                if row == "new_tree_json":
                    y_offset = row13_offset
                    x_offset = 0
                    row13_offset = row13_offset + 1
                elif row == "row15":
                    x_offset = 3
                    y_offset = row15_offset
                    row15_offset = row15_offset + 1
                elif row == "row16":
                    x_offset = 6
                    y_offset = row16_offset
                    row16_offset = row16_offset + 1
            offset_vector = np.array([x_offset, y_offset, 0])

            # Load tree data from JSON file
            with open(os.path.join(root, file), "r") as json_file:
                data = json.load(json_file)

            # Convert unit from cm to m if specified
            if convert_to_m:
                for key in data.keys():
                    if key in [
                        "trunk_radius",
                        "trunk_points",
                        "branch_points",
                        "branch_points_radius",
                    ]:
                        if isinstance(data[key], list):
                            for i in range(len(data[key])):
                                data[key][i] = (np.asarray(data[key][i]) / 100).tolist()
                        else:
                            data[key] = data[key] / 100

            # Taper trunk radius if specified
            data["trunk_radius"] = (
                taper_radius(
                    data["trunk_points"],
                    [data["trunk_radius"]],
                    taper_angle=trunk_taper,
                )
                if taper
                else data["trunk_radius"]
            )
            offset_trunk_points = np.asarray(data["trunk_points"]) + offset_vector
            data["trunk_points"] = offset_trunk_points.tolist()

            # Trunk
            interpolated_trunk_points = interpolate_points(data["trunk_points"])
            interpolated_trunk_radii = interpolate_radii(data["trunk_radius"])
            trunk_point_directions = calculate_direction_vectors(
                interpolated_trunk_points
            )

            # Copy the branch_internode_ratio if it exists in the original data
            branch_internode_ratio = data.get("branch_internode_ratio", [])

            trunk_skeleton_dict = {
                "trunk_points": interpolated_trunk_points,
                "trunk_radius": interpolated_trunk_radii,
                "trunk_direction": trunk_point_directions,
                "branch_internode_ratio": branch_internode_ratio,  # Added line
            }

            # Save trunk JSON file
            trunk_json_filepath = os.path.join(
                output_trunk_json_folder, f"{tree_id}_trunk.json"
            )
            with open(trunk_json_filepath, "w") as f:
                json.dump(trunk_skeleton_dict, f, indent=4)
            trunk_json_file_list.append(trunk_json_filepath)

            # Produce Lpy files for trunk
            trunk_content = generate_lpy_content(
                ["T"], [data["trunk_points"]], [data["trunk_radius"]]
            )
            trunk_file_name = f"{tree_id}_trunk.lpy"
            with open(os.path.join(lpy_trunk_dir, trunk_file_name), "w") as file:
                file.write(trunk_content)

            # Append to tree-level Lpy content
            tree_lpy_modules.append("T")
            tree_lpy_points.append(data["trunk_points"])
            tree_lpy_radii.append(data["trunk_radius"])
            tree_lpy_leaf_data.append(None)  # Trunk has no leaves

            # Initialize empty lists for branch skeleton points, radii, and directions
            all_branch_skeleton_pts = []
            all_branch_skeleton_rd = []
            all_branch_skeleton_dirs = []

            # Branches
            branch_radius_stats["level1"] = branch_radius_stats.get("level1", [])
            for branch_idx, (branch, radii) in enumerate(
                zip(data["branch_points"], data["branch_points_radius"])
            ):
                taper_radii = (
                    taper_radius(branch, radii, taper_angle=branch_taper)
                    if taper
                    else radii
                )
                offset_branch = np.asarray(branch) + offset_vector
                branch = offset_branch.tolist()
                branch_content = generate_lpy_content(
                    [f"B{branch_idx+1}"], [branch], [taper_radii]
                )

                # Append to tree-level Lpy content
                tree_lpy_modules.append(f"B{branch_idx+1}")
                tree_lpy_points.append(branch)
                tree_lpy_radii.append(taper_radii)

                branch_file_name = f"{tree_id}_branch{branch_idx+1}.lpy"
                with open(os.path.join(lpy_branch_dir, branch_file_name), "w") as file:
                    file.write(branch_content)
                branch_counter += 1

                interpolated_branch_points = interpolate_points(branch)
                interpolated_branch_radii = interpolate_radii(taper_radii)
                branch_point_directions = calculate_direction_vectors(
                    interpolated_branch_points
                )

                branch_skeleton_dict = {
                    "centers": interpolated_branch_points,
                    "center_radii": interpolated_branch_radii,
                    "center_directions": branch_point_directions,
                }

                # Generate leaves for this branch
                if config["parameters"].get("leaf_generation", False):
                    leaf_data = process_branch_leaves(
                        interpolated_branch_points,
                        branch_point_directions,
                        branch_level=1,
                        config=config
                    )
                    branch_skeleton_dict["leaf_data"] = leaf_data
                    tree_lpy_leaf_data.append(leaf_data)
                else:
                    branch_skeleton_dict["leaf_data"] = None
                    tree_lpy_leaf_data.append(None)

                if branch_obj:
                    # Save branch data to .npz file for compatibility
                    branch_npz_filepath = os.path.join(
                        output_npy_folder, f"{tree_id}_branch{branch_idx+1}.npz"
                    )
                    np.savez(branch_npz_filepath, **branch_skeleton_dict)
                    print(f"Saved {branch_npz_filepath}")

                all_branch_skeleton_pts.append(interpolated_branch_points)
                all_branch_skeleton_rd.append(interpolated_branch_radii)
                all_branch_skeleton_dirs.append(branch_point_directions)

                branch_radius_stats["level1"].append(taper_radii[0])

            # Generate higher order branches
            for level in range(1, branch_level + 1):
                # Match the keys in NewTreeGenerator.py
                key1 = f"new{level+1}_branch_points"
                key2 = f"new{level+1}_branch_points_radius"

                # Secondary Branches
                new_branch_points_list = data.get(key1, [])
                new_branch_points_radius_list = data.get(key2, [])
                assert len(new_branch_points_list) == len(
                    new_branch_points_radius_list
                ), f"Not Matched"

                if len(new_branch_points_list) > 0:
                    print(f"Processing level-{level+1} branches!")
                    branch_radius_stats[f"level{level+1}"] = branch_radius_stats.get(
                        f"level{level+1}", []
                    )

                for branch_idx, (branch, radii) in enumerate(
                    zip(new_branch_points_list, new_branch_points_radius_list)
                ):
                    if branch is None:
                        continue
                    offset_branch = np.asarray(branch) + offset_vector
                    branch = offset_branch.tolist()
                    branch_content = generate_lpy_content(
                        [f"BL{level+1}{branch_idx+1}"], [branch], [radii]
                    )

                    # Append to tree-level Lpy content
                    tree_lpy_modules.append(f"BL{level+1}{branch_idx+1}")
                    tree_lpy_points.append(branch)
                    tree_lpy_radii.append(radii)

                    branch_file_name = (
                        f"{tree_id}_level{level+1}_branch{branch_idx+1}.lpy"
                    )
                    with open(
                        os.path.join(lpy_branch_dir, branch_file_name), "w"
                    ) as file:
                        file.write(branch_content)
                    secondary_branch_counter += 1

                    interpolated_branch_points = interpolate_points(branch)
                    interpolated_branch_radii = interpolate_radii(radii)
                    branch_point_directions = calculate_direction_vectors(
                        interpolated_branch_points
                    )

                    secondary_branch_skeleton_dict = {
                        "centers": interpolated_branch_points,
                        "center_radii": interpolated_branch_radii,
                        "center_directions": branch_point_directions,
                    }

                    # Generate leaves for this hierarchical branch
                    if config["parameters"].get("leaf_generation", False):
                        leaf_data = process_branch_leaves(
                            interpolated_branch_points,
                            branch_point_directions,
                            branch_level=level+1,
                            config=config
                        )
                        secondary_branch_skeleton_dict["leaf_data"] = leaf_data
                        tree_lpy_leaf_data.append(leaf_data)
                    else:
                        secondary_branch_skeleton_dict["leaf_data"] = None
                        tree_lpy_leaf_data.append(None)

                    if branch_obj:
                        # Save branch data to .npz file for compatibility
                        branch_npz_filepath = os.path.join(
                            output_npy_folder,
                            f"{tree_id}_secondary_branch{branch_idx+1}.npz",
                        )
                        np.savez(branch_npz_filepath, **secondary_branch_skeleton_dict)
                        print(f"Saved {branch_npz_filepath}")

                    all_branch_skeleton_pts.append(interpolated_branch_points)
                    all_branch_skeleton_rd.append(interpolated_branch_radii)
                    all_branch_skeleton_dirs.append(branch_point_directions)

                    branch_radius_stats[f"level{level+1}"].append(radii[0])

            # Produce Lpy files for branches
            if trunk_obj:
                process_lpy_files(lpy_trunk_dir, obj_trunk_dir)
            if branch_obj:
                process_lpy_files(lpy_branch_dir, obj_branch_dir)

            # Write tree Lpy file and generate tree-level .obj file if specified
            if config["parameters"].get("leaf_generation", False) and any(tree_lpy_leaf_data):
                tree_content = generate_lpy_content_with_leaves(
                    tree_lpy_modules, tree_lpy_points, tree_lpy_radii, tree_lpy_leaf_data
                )
            else:
                tree_content = generate_lpy_content(
                    tree_lpy_modules, tree_lpy_points, tree_lpy_radii
                )
            tree_file_name = f"{tree_id}.lpy"

            with open(os.path.join(lpy_tree_dir, tree_file_name), "w") as file:
                file.write(tree_content)
            if tree_obj:
                process_lpy_files(lpy_tree_dir, obj_row_dir)

            # Save branch skeleton data to JSON file
            all_branch_skeleton_dict = {
                "centers": all_branch_skeleton_pts,
                "center_radii": all_branch_skeleton_rd,
                "center_directions": all_branch_skeleton_dirs,
            }
            branch_json_filepath = os.path.join(
                output_branch_json_folder, f"{tree_id}_branch-{branch_idx+1}.json"
            )
            with open(branch_json_filepath, "w") as f:
                json.dump(all_branch_skeleton_dict, f, indent=4)

            # Generate tree skeleton data by combining trunk and branch skeleton data
            tree_skeleton_dict = {
                "centers": interpolated_trunk_points,
                "center_radii": interpolated_trunk_radii,
                "center_directions": trunk_point_directions,
            }
            for key, value in all_branch_skeleton_dict.items():
                for value_list in value:
                    for v in value_list:
                        tree_skeleton_dict[key].append(v)

            if tree_obj:
                # Save tree skeleton data to .npz file for compatibility
                tree_npz_filepath = os.path.join(output_npy_folder, f"{tree_id}.npz")
                np.savez(tree_npz_filepath, **tree_skeleton_dict)
                print(f"Saved {tree_npz_filepath}")

            # Tree skeleton statistics
            tree_skeleton_num_list.append(len(tree_skeleton_dict["centers"]))

            trunk_counter += 1

print(f"Processing {trunk_counter} trees and {branch_counter} branches completed!")

# Merge all trunk JSON files into one combined JSON file
all_trunk_json_filepath = os.path.join(output_json_folder, "all_trunks_combined.json")
all_trunk_points = []
all_trunk_radii = []
all_branch_internode_ratios = []

for trunk_json_filepath in trunk_json_file_list:
    with open(trunk_json_filepath, "r") as f:
        data = json.load(f)
    trunk_points = data.get("trunk_points", [])
    trunk_radius = data.get("trunk_radius", 0)
    branch_internode_ratio = data.get("branch_internode_ratio", [])

    all_trunk_points.append(trunk_points)
    all_trunk_radii.append(trunk_radius)
    all_branch_internode_ratios.append(branch_internode_ratio)

print(f"Processed {len(trunk_json_file_list)} trunk data sets from JSON files.")

json_content = json.dumps(
    {
        "all_trunk_points": all_trunk_points,
        "all_trunk_radii": all_trunk_radii,
        "all_branch_internode_ratios": all_branch_internode_ratios,
    },
    indent=4,
)

# Save the formatted JSON string to a file
with open(all_trunk_json_filepath, "w") as f:
    f.write(json_content)

### Generate statistical figures
print(f"Generating simulated tree stats figures!")
# Create the tree skeleton # histogram
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.hist(np.array(tree_skeleton_num_list))
ax.set_title("Histogram of Tree Skeleton Numbers")
ax.set_xlabel("Tree Skeleton Numbers")
fig_filepath = os.path.join(save_folder, "tree_skeleton_number.png")
fig.savefig(fig_filepath, dpi=300)

# Close the figure
plt.close(fig)
print(f"save {fig_filepath}")

# Create the branch radisu histogram
fig, axes = plt.subplots(nrows=1, ncols=len(branch_radius_stats), figsize=(15, 6))
for index, (key, value) in enumerate(branch_radius_stats.items()):
    n = len(value)
    if len(branch_radius_stats) == 1:
        axes.hist(np.array(value) * 1000)
        axes.set_title(f"Histogram of {n} {key.capitalize()}-Branch Distribution")
        axes.set_xlabel("Branch Radius (mm)")
    else:
        axes[index].hist(np.array(value) * 1000)
        axes[index].set_title(
            f"Histogram of {n} {key.capitalize()}-Branch Distribution"
        )
        axes[index].set_xlabel("Branch Radius (mm)")

fig_filepath = os.path.join(save_folder, f"tree_branch_radius_stats.png")
fig.savefig(fig_filepath, dpi=300)
plt.close(fig)
print(f"save {fig_filepath}")
