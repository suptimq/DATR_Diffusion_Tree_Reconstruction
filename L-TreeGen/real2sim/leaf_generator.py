"""
Leaf Generation Module for L-TreeGen/real2sim

This module provides functions to procedurally generate leaves on tree branches
using L-Py's built-in leaf primitives and phyllotaxis patterns.
"""

import numpy as np
import random


def select_bud_points(branch_centers, branch_directions, leaf_density,
                      start_ratio=0.1, end_ratio=0.95, phyllotaxis_angle=137.5):
    """
    Select bud points along a branch using evenly-spaced strategy with phyllotaxis.

    Args:
        branch_centers (list): List of 3D points [x,y,z] representing the branch skeleton (typically 100 points)
        branch_directions (list): Direction vectors at each point
        leaf_density (float): Number of leaves per cm of branch length
        start_ratio (float): Skip first X% of branch (default 0.1 = 10%)
        end_ratio (float): Stop at X% of branch (default 0.95 = 95%)
        phyllotaxis_angle (float): Golden angle for spiral leaf arrangement (default 137.5 degrees)

    Returns:
        list: List of dicts containing:
            - 'point_idx': int (index in branch_centers)
            - 'position': [x,y,z] (3D position)
            - 'branch_direction': [dx,dy,dz] (local branch direction)
            - 'rotation_angle': float (cumulative phyllotaxis angle in degrees)
    """
    if len(branch_centers) < 2:
        return []

    # Calculate arc length along branch
    segment_lengths = []
    total_length = 0.0
    for i in range(len(branch_centers) - 1):
        p1 = np.array(branch_centers[i])
        p2 = np.array(branch_centers[i + 1])
        length = np.linalg.norm(p2 - p1)
        segment_lengths.append(length)
        total_length += length

    # Determine valid region for leaf attachment
    start_length = total_length * start_ratio
    end_length = total_length * end_ratio
    valid_length = end_length - start_length

    if valid_length <= 0:
        return []

    # Calculate number of leaves
    num_leaves = int(valid_length * leaf_density)
    if num_leaves == 0:
        return []

    # Calculate cumulative lengths to map leaves to skeleton points
    cumulative_lengths = [0.0]
    for length in segment_lengths:
        cumulative_lengths.append(cumulative_lengths[-1] + length)

    # Evenly space leaves within valid region
    bud_points = []
    for i in range(num_leaves):
        # Target length along branch for this leaf
        target_length = start_length + (i / max(num_leaves - 1, 1)) * valid_length

        # Find closest skeleton point
        point_idx = 0
        for j in range(len(cumulative_lengths) - 1):
            if cumulative_lengths[j] <= target_length < cumulative_lengths[j + 1]:
                point_idx = j
                break
        else:
            point_idx = len(branch_centers) - 1

        # Cumulative phyllotaxis rotation
        rotation_angle = (i * phyllotaxis_angle) % 360.0

        bud_points.append({
            'point_idx': point_idx,
            'position': branch_centers[point_idx],
            'branch_direction': branch_directions[point_idx],
            'rotation_angle': rotation_angle
        })

    return bud_points


def compute_leaf_angles(branch_direction, rotation_angle, orientation_angle_range):
    """
    Compute pitch and rotation angles for L-Py leaf orientation commands.

    Args:
        branch_direction (list): Branch direction vector [dx, dy, dz]
        rotation_angle (float): Phyllotaxis rotation angle (degrees)
        orientation_angle_range (tuple): (min, max) angle range from branch axis (degrees)

    Returns:
        tuple: (pitch_angle, rotation_angle) in degrees
            - pitch_angle: Tilt from branch axis (for ^ command)
            - rotation_angle: Rotation around branch (for + command)
    """
    # Random pitch angle within range
    pitch_angle = random.uniform(*orientation_angle_range)

    # rotation_angle comes from phyllotaxis (already computed)
    return pitch_angle, rotation_angle


def generate_leaf_geometry(bud_point, leaf_size, orientation_angle_range):
    """
    Generate leaf data for L-Py's built-in leaf primitive (~).

    Args:
        bud_point (dict): Bud point information from select_bud_points()
        leaf_size (float): Leaf length in cm
        orientation_angle_range (tuple): (min, max) angle range from branch axis

    Returns:
        dict: Leaf data for L-Py with keys:
            - 'position': [x,y,z] (3D position)
            - 'pitch_angle': float (tilt from branch, degrees)
            - 'rotation_angle': float (phyllotaxis rotation, degrees)
            - 'size': float (leaf length in cm)
    """
    pitch_angle, rotation_angle = compute_leaf_angles(
        bud_point['branch_direction'],
        bud_point['rotation_angle'],
        orientation_angle_range
    )

    return {
        'position': bud_point['position'],
        'pitch_angle': pitch_angle,
        'rotation_angle': rotation_angle,
        'size': leaf_size
    }


def process_branch_leaves(branch_centers, branch_directions, branch_level, config):
    """
    Main entry point: process one branch to generate all leaf data.

    Args:
        branch_centers (list): Interpolated branch skeleton points (100 points)
        branch_directions (list): Direction vectors at each point (100 vectors)
        branch_level (int): Branch hierarchy level (1, 2, 3, ...)
        config (dict): Full configuration dictionary

    Returns:
        list: List of leaf geometry dicts ready for L-Py generation
    """
    leaf_params = config["parameters"]

    # Check if leaf generation is enabled
    if not leaf_params.get("leaf_generation", False):
        return []

    # Get level-specific parameters (with fallback to last value)
    leaf_density_list = leaf_params["leaf_density"]
    leaf_size_list = leaf_params["leaf_size"]

    # Use level-1 as index (level 1 → index 0)
    level_idx = branch_level - 1
    leaf_density = leaf_density_list[min(level_idx, len(leaf_density_list) - 1)]
    leaf_size = leaf_size_list[min(level_idx, len(leaf_size_list) - 1)]

    # IMPORTANT: Handle unit conversion
    # If convert_to_m is True, branch_centers are in meters, but leaf_density and leaf_size are in cm units
    convert_to_m = config["experiment"].get("convert_to_m", False)
    if convert_to_m:
        # Branch in meters, density in leaves/cm → convert to leaves/m
        leaf_density = leaf_density * 100  # 0.15 leaves/cm = 15 leaves/m
        # Leaf size in cm → convert to m
        leaf_size = leaf_size / 100  # 10cm = 0.1m

    orientation_angle_range = tuple(leaf_params["leaf_orientation_angle"])
    phyllotaxis_angle = leaf_params["leaf_phyllotaxis_angle"]
    start_ratio = leaf_params["leaf_start_ratio"]
    end_ratio = leaf_params["leaf_end_ratio"]

    # Select bud points
    bud_points = select_bud_points(
        branch_centers,
        branch_directions,
        leaf_density,
        start_ratio,
        end_ratio,
        phyllotaxis_angle
    )

    # Generate leaf geometry for each bud point
    leaf_data_list = []
    for bud_point in bud_points:
        leaf_data = generate_leaf_geometry(
            bud_point,
            leaf_size,
            orientation_angle_range
        )
        leaf_data_list.append(leaf_data)

    return leaf_data_list
