"""
This script aims to convert the AppleQSM characterization results (i.e., mat files) to json files for simulated tree modeling.
Usually the AppleQSM results are saved in {SOMEWHERE}\AppleQSM\Characterization\Fusion
Notice 
    1. You might have to manually organize the AppleQSM results by creating a parent folder starting with "row" 
        and move all mat files into it
    2. The unit is cm for all output variables!
"""

import os
import sys
import json

import scipy
from scipy.spatial.distance import cdist

import numpy as np

from natsort import natsorted
from pathlib import Path

from utils import load_config, lr_adjustment


# Load configuration from YAML file
config = load_config("Config\knx.yaml")

data_folder = Path(config["experiment"]["fusion_folder"])
save_folder = Path(config["experiment"]["data_folder"])
row_ids = [x for x in os.listdir(data_folder) if x.startswith("row")]

for row_id in row_ids:
    print(f"=================================")
    print(f"processing {row_id}")
    save_json_folder = save_folder / row_id
    os.makedirs(save_json_folder, exist_ok=True)

    row_folder = data_folder / row_id
    mat_files = [x for x in os.listdir(row_folder) if x.endswith(".mat")]
    mat_files = natsorted(mat_files)

    # linear adjustment (unit in mm)
    a = 0.88
    b = 7.87

    for mat_file in mat_files:
        print(f"processing {mat_file}")
        mat_filepath = row_folder / mat_file
        tree_id = mat_file.split("_")[0]
        save_json_filepath = save_json_folder / f"{tree_id}.json"

        mat_data = scipy.io.loadmat(mat_filepath)
        P = mat_data["P"]

        # trunk parameters
        main_trunk_points = P["trunk_cpc_optimized_center"][0][0] * 100
        trunk_radius = P["trunk_radius"][0][0] * 100

        # branch parameters
        branch_internode_ratio = P["branch_internode_ratio"][0][0]
        primary_branch_points = P["branch_pts_list"][0][0][0] * 100
        primary_branch_points_radius = P["branch_pts_radius_list"][0][0][0] * 1000
        branch_diameter_list = lr_adjustment(P["branch_diameter"][0][0], a=a, b=b) / 10
        branch_angle_list = P["branch_angle"][0][0]

        # reshape
        trunk_radius = trunk_radius.reshape(-1, 1).squeeze()
        branch_internode_ratio = branch_internode_ratio.reshape(-1, 1).squeeze()
        branch_diameter_list = branch_diameter_list.reshape(-1, 1).squeeze()
        branch_angle_list = branch_angle_list.reshape(-1, 1).squeeze()

        # make sure the root is at (0, 0, 0)
        offset_points = main_trunk_points.copy()[0]
        main_trunk_points -= offset_points

        branch_points = []  # cm
        mean_branch_point_distance = []  # cm
        branch_points_radius = []  # cm
        for i in range(primary_branch_points.shape[0]):
            offset_primary_branch_points = primary_branch_points[i] - offset_points
            pairwise_ed = cdist(
                offset_primary_branch_points, offset_primary_branch_points
            )
            row = np.arange(pairwise_ed.shape[0] - 1)  # exclude the last point
            col = row + 1
            neighbor_ped = pairwise_ed[row, col]
            mean_branch_point_distance.append(np.mean(neighbor_ped))
            branch_points.append(offset_primary_branch_points.tolist())

            adjusted_radius = (
                lr_adjustment(primary_branch_points_radius[i].squeeze(), a=a, b=b) / 10
            )
            assert (
                adjusted_radius.shape[0] == offset_primary_branch_points.shape[0]
            ), f"adjusted_radius shape {adjusted_radius.shape} ~= offset_primary_branch_points shape {offset_primary_branch_points.shape}"
            branch_points_radius.append(adjusted_radius.tolist())

        tree_json = {
            "trunk_radius": trunk_radius.tolist(),
            "trunk_points": main_trunk_points.tolist(),  # (N, 3)
            "branch_internode_ratio": branch_internode_ratio.tolist(),  # (M, )
            # (K, P, 3) - K branches, P branch points (not determined)
            "branch_points": branch_points,
            # (K, P)    - accordingly
            "branch_points_radius": branch_points_radius,
            # (K, )
            "mean_branch_point_distance": mean_branch_point_distance,
            # (K, )     - this is not necessary to be identical with the 1st branch radius
            "branch_radius": branch_diameter_list.tolist(),
            "branch_angle": branch_angle_list.tolist(),  # (K, )
        }

        with open(save_json_filepath, "w") as outfile:
            json.dump(tree_json, outfile)

        print(f"{tree_id} saved to {save_json_filepath}")
