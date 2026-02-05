"""
Fruit Generation Module for L-TreeGen/real2sim

This module provides functions to procedurally generate fruits (apples) on tree branches
using L-Py's built-in sphere primitive (@O).
"""

import numpy as np
import random


def select_fruit_points(branch_centers, branch_directions, fruit_density,
                        start_ratio=0.2, end_ratio=0.8, spacing_variation=0.3):
    """
    Select fruit attachment points along a branch using spaced strategy with variation.

    Args:
        branch_centers (list): List of 3D points [x,y,z] representing the branch skeleton (typically 100 points)
        branch_directions (list): Direction vectors at each point
        fruit_density (float): Number of fruits per cm of branch length
        start_ratio (float): Skip first X% of branch (default 0.2 = 20%)
        end_ratio (float): Stop at X% of branch (default 0.8 = 80%)
        spacing_variation (float): Randomness in fruit spacing (0=even, 1=random)

    Returns:
        list: List of dicts containing:
            - 'point_idx': int (index in branch_centers)
            - 'position': [x,y,z] (3D position)
            - 'branch_direction': [dx,dy,dz] (local branch direction)
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

    # Determine valid region for fruit attachment
    start_length = total_length * start_ratio
    end_length = total_length * end_ratio
    valid_length = end_length - start_length

    if valid_length <= 0:
        return []

    # Calculate number of fruits
    num_fruits = int(valid_length * fruit_density)
    if num_fruits == 0:
        return []

    # Calculate cumulative lengths to map fruits to skeleton points
    cumulative_lengths = [0.0]
    for length in segment_lengths:
        cumulative_lengths.append(cumulative_lengths[-1] + length)

    # Place fruits with spacing variation
    fruit_points = []
    for i in range(num_fruits):
        # Base spacing with variation
        base_position = i / max(num_fruits - 1, 1)
        variation = (random.random() - 0.5) * spacing_variation
        position_ratio = np.clip(base_position + variation, 0, 1)

        target_length = start_length + position_ratio * valid_length

        # Find closest skeleton point
        point_idx = 0
        for j in range(len(cumulative_lengths) - 1):
            if cumulative_lengths[j] <= target_length < cumulative_lengths[j + 1]:
                point_idx = j
                break
        else:
            point_idx = len(branch_centers) - 1

        fruit_points.append({
            'point_idx': point_idx,
            'position': branch_centers[point_idx],
            'branch_direction': branch_directions[point_idx]
        })

    return fruit_points


def generate_fruit_geometry(fruit_point, fruit_radius, fruit_color, hang_distance=0.0):
    """
    Generate fruit data for L-Py's sphere primitive (@O).

    Args:
        fruit_point (dict): Fruit point information from select_fruit_points()
        fruit_radius (float): Radius of the fruit (apple) in cm
        fruit_color (list): RGB color values [r, g, b] (0-1 range)
        hang_distance (float): Distance fruits hang below branch (in same units as position)

    Returns:
        dict: Fruit data for L-Py with keys:
            - 'position': [x,y,z] (3D position with gravity offset)
            - 'radius': float (fruit radius)
            - 'color': [r, g, b] (RGB color)
    """
    # Apply gravity offset (fruits hang below branch)
    base_pos = np.array(fruit_point['position'])
    hanging_pos = base_pos + np.array([0, 0, -hang_distance])  # Negative Z = downward

    return {
        'position': hanging_pos.tolist(),
        'radius': fruit_radius,
        'color': fruit_color
    }


def generate_fruit_cluster(fruit_point, fruit_radius, fruit_color, cluster_size, cluster_spread=0.02, hang_distance=0.0):
    """
    Generate a cluster of fruits at a single attachment point.

    Args:
        fruit_point (dict): Base fruit point information
        fruit_radius (float): Radius of each fruit
        fruit_color (list): RGB color values [r, g, b]
        cluster_size (int): Number of fruits in cluster (1-3 typical)
        cluster_spread (float): How far fruits spread from center (in meters)
        hang_distance (float): Distance fruits hang below branch (in same units as position)

    Returns:
        list: List of fruit data dicts
    """
    fruits = []
    base_pos = np.array(fruit_point['position'])

    # Apply gravity offset to base position
    hanging_base_pos = base_pos + np.array([0, 0, -hang_distance])

    for i in range(cluster_size):
        # Add small random offset for clustering effect
        if i > 0:  # First fruit at exact position
            offset = np.array([
                random.uniform(-cluster_spread, cluster_spread),
                random.uniform(-cluster_spread, cluster_spread),
                random.uniform(-cluster_spread, cluster_spread)
            ])
            position = (hanging_base_pos + offset).tolist()
        else:
            position = hanging_base_pos.tolist()

        # Slight radius variation within cluster
        radius_variation = random.uniform(0.95, 1.05)
        varied_radius = fruit_radius * radius_variation

        fruits.append({
            'position': position,
            'radius': varied_radius,
            'color': fruit_color
        })

    return fruits


def process_branch_fruits(branch_centers, branch_directions, branch_level, config):
    """
    Main entry point: process one branch to generate all fruit data.

    Args:
        branch_centers (list): Interpolated branch skeleton points (100 points)
        branch_directions (list): Direction vectors at each point (100 vectors)
        branch_level (int): Branch hierarchy level (1, 2, 3, ...)
        config (dict): Full configuration dictionary

    Returns:
        list: List of fruit geometry dicts ready for L-Py generation
    """
    fruit_params = config["parameters"]

    # Check if fruit generation is enabled
    if not fruit_params.get("fruit_generation", False):
        return []

    # Get level-specific parameters (with fallback to last value)
    fruit_density_list = fruit_params["fruit_density"]
    fruit_radius_list = fruit_params["fruit_radius"]

    # Use level-1 as index (level 1 → index 0)
    level_idx = branch_level - 1
    fruit_density = fruit_density_list[min(level_idx, len(fruit_density_list) - 1)]
    fruit_radius = fruit_radius_list[min(level_idx, len(fruit_radius_list) - 1)]

    # Skip if density is 0 for this level
    if fruit_density <= 0:
        return []

    # IMPORTANT: Handle unit conversion
    # If convert_to_m is True, branch_centers are in meters, but fruit_density and fruit_radius are in cm units
    convert_to_m = config["experiment"].get("convert_to_m", False)
    if convert_to_m:
        # Branch in meters, density in fruits/cm → convert to fruits/m
        fruit_density = fruit_density * 100  # 0.05 fruits/cm = 5 fruits/m
        # Fruit radius in cm → convert to m
        fruit_radius = fruit_radius / 100  # 6cm = 0.06m

    # Get other parameters
    fruit_color = fruit_params.get("fruit_color", [1.0, 0.0, 0.0])  # Default red
    start_ratio = fruit_params.get("fruit_start_ratio", 0.2)
    end_ratio = fruit_params.get("fruit_end_ratio", 0.8)
    spacing_variation = fruit_params.get("fruit_spacing_variation", 0.3)

    # Clustering parameters
    enable_clustering = fruit_params.get("fruit_clustering", True)
    cluster_size_range = fruit_params.get("fruit_cluster_size", [1, 2])
    cluster_spread = fruit_params.get("fruit_cluster_spread", 0.02)

    # Gravity parameters (fruits hang below branch)
    hang_distance = fruit_params.get("fruit_hang_distance", 3)  # Default 3 cm
    if convert_to_m:
        hang_distance = hang_distance / 100  # Convert cm to m

    # Select fruit points
    fruit_points = select_fruit_points(
        branch_centers,
        branch_directions,
        fruit_density,
        start_ratio,
        end_ratio,
        spacing_variation
    )

    # Generate fruit geometry for each point
    fruit_data_list = []
    for fruit_point in fruit_points:
        if enable_clustering:
            # Determine cluster size
            cluster_size = random.randint(cluster_size_range[0], cluster_size_range[1])
            # Generate fruit cluster
            fruits = generate_fruit_cluster(
                fruit_point,
                fruit_radius,
                fruit_color,
                cluster_size,
                cluster_spread if convert_to_m else cluster_spread * 100,  # Convert spread if needed
                hang_distance
            )
            fruit_data_list.extend(fruits)
        else:
            # Single fruit
            fruit_data = generate_fruit_geometry(
                fruit_point,
                fruit_radius,
                fruit_color,
                hang_distance
            )
            fruit_data_list.append(fruit_data)

    return fruit_data_list
