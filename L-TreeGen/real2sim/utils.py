import os
import yaml
import json
import string
import random
import trimesh

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from sklearn.decomposition import PCA


def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def lr_adjustment(x, a=1, b=0):
    """
    Adjust traits based on the estimated linear model
    """
    return a * x + b


def radius_check(radii, min_radius, taper):
    """
    Check and adjust the radii list to ensure that each radius is greater than or equal to a minimum value.

    Args:
        radii (list): List of radii to be checked and adjusted.
        min_radius (float): Minimum acceptable radius value.

    Returns:
        radii: The adjusted radii list where each radius is greater than or equal to min_radius.
    """
    first_min_radius = False
    for i in range(1, len(radii)):
        radii[i] = abs(radii[i])
        if radii[i] < min_radius:
            radii[i] = min_radius if not first_min_radius else min_radius * taper
            first_min_radius = True
        if radii[i] > radii[i - 1]:
            radii[i] = radii[i - 1]
    return radii


def taper_radius(points, radii, taper_angle=-0.3, max_tail_radius=None):
    """
    Calculate the tapered radii of a trunk/branch given its points, initial radii, taper angle, and maximum tail radius.
    Two modes:
        1. Given taper_angle.
        2. Given max_tail_radius.
        3. If both are not None, the 2nd mode will be forced.

    Args:
        points (list or np.ndarray): A list or array of points.
        radii (list or np.ndarray): Initial radii of points.
        taper_angle (optional): Taper angle in degrees. Default is -0.3.
        max_tail_radius (optional): Maximum tail radius. Default is 0.003 (m).

    Returns:
        list: A list of tapered trunk radii.
    """
    if isinstance(points, list):
        points = np.asarray(points)
    N = points.shape[0]
    diffs = points[1:] - points[:-1]
    distances = np.linalg.norm(diffs, axis=1)
    accumulated_lengths = np.cumsum(distances)
    radius, last_radius = radii[0], radii[-1]
    tapered_radius = np.full(N, radius)

    if taper_angle is not None:
        tapered_radius[1:] = (
            radius - np.tan(np.deg2rad(-taper_angle)) * accumulated_lengths
        )

    # Recalculate taper if the last radius is greater than or equal to max_tail_radius
    # import pdb; pdb.set_trace()
    if max_tail_radius is not None and last_radius > max_tail_radius:
        taper_angle = -np.rad2deg(
            np.arctan((radius - max_tail_radius) / accumulated_lengths[-1])
        )
        tapered_radius[1:] = (
            radius - np.tan(np.deg2rad(-taper_angle)) * accumulated_lengths
        )

    return tapered_radius.tolist()


def interpolate_points(branch, num_points=100):
    """
    Interpolate points along a trunk/branch to increase the point density.

    Args:
        branch (list): List of points representing the branch.
        num_points (int): Number of points to interpolate along the branch.

    Returns:
        interpolated_points: List of interpolated points along the branch.
    """
    n = len(branch)
    u = np.linspace(0, n - 1, n)
    u_new = np.linspace(0, n - 1, num_points)

    transposed_points = np.array(branch).T
    spline = [interp1d(u, transposed_points[i], kind="cubic") for i in range(3)]

    interpolated_points = np.array([s(u_new) for s in spline]).T.tolist()
    return interpolated_points


def interpolate_radii(radii, num_points=100):
    """
    Interpolate radii along a trunk/branch, ensuring non-increasing order.

    Args:
        radii (list): List of radii representing the branch radii.
        num_points (int): Number of points to interpolate along the branch.

    Returns:
        new_radii: List of interpolated radii along the branch.
    """

    n = len(radii)
    u = np.linspace(0, n - 1, n)
    u_new = np.linspace(0, n - 1, num_points)

    spline_r = interp1d(u, radii, kind="cubic")
    new_radii = spline_r(u_new)

    # Ensuring that each radius is equal or larger than the following one
    for i in range(1, len(new_radii)):
        new_radii[i] = abs(new_radii[i])
        if new_radii[i] > new_radii[i - 1]:
            new_radii[i] = new_radii[i - 1]

    return new_radii.tolist()


def calculate_direction_vectors(points):
    """
    Calculate direction vectors between consecutive points.

    Args:
        points (list): List of points.

    Returns:
        vectors: List of direction vectors.
    """
    vectors = [
        np.subtract(points[i], points[i - 1]).tolist() for i in range(1, len(points))
    ]
    vectors.append(vectors[-1])  # Repeat the last vector to match the length
    return vectors


def plot_path_groups(path_groups):
    """
    Plot a collection of path groups in 3D.

    Args:
        path_groups (list): List of path groups, where each path group is a list of points.

    Returns:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for path_group in path_groups:
        x_values = [point[0] for point in path_group]
        y_values = [point[1] for point in path_group]
        z_values = [point[2] for point in path_group]
        ax.plot(x_values, y_values, z_values)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("3D Plot of Tree Branches")

    plt.show()


def normalize_path_group(path_group):
    """
    Normalizes a path group by translating it so that the first point is at [0,0,0] and rotating it so that the last point
    aligns with [1,1,1].

    Args:
        path_group (list of lists): A list of 3D points representing the path group.

    Returns:
        rotated_path_group (list of lists): The normalized and rotated path group.
    """
    # Translate so that the first point is at [0,0,0]
    x0, y0, z0 = path_group[0]
    translated_path_group = [[x - x0, y - y0, z - z0] for x, y, z in path_group]

    # Rotate so that the last point aligns with [1,1,1]
    x_end, y_end, z_end = translated_path_group[-1]
    target_vector = np.array([1, 1, 1]) / np.sqrt(3)
    end_vector = np.array([x_end, y_end, z_end])
    end_vector_normalized = end_vector / np.linalg.norm(end_vector)

    # Axis of rotation (cross product)
    axis = np.cross(end_vector_normalized, target_vector)
    axis_length = np.linalg.norm(axis)
    if axis_length != 0:
        axis = axis / axis_length

    # Angle of rotation (dot product)
    angle = np.arccos(np.clip(np.dot(end_vector_normalized, target_vector), -1.0, 1.0))

    # Rodrigues' rotation formula components
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    one_minus_cos = 1 - cos_angle
    axis_outer = np.outer(axis, axis)

    # Rotation matrix
    rotation_matrix = (
        cos_angle * np.eye(3)
        + sin_angle * np.cross(np.eye(3), axis)
        + one_minus_cos * axis_outer
    )

    # Apply rotation to all points in the path group
    rotated_path_group = [
        rotation_matrix.dot(point).tolist() for point in translated_path_group
    ]

    return rotated_path_group


def compute_branch_direction(branch_points):
    """
    Computes the principal direction of a branch using PCA and aligns it with the vector pointing from branch_points[0] to branch_points[1].

    Args:
        branch_points (numpy.ndarray): An Nx3 array representing the coordinates of the branch points.

    Returns:
        principal_direction (numpy.ndarray): A unit vector representing the principal direction of the branch.
    """

    if isinstance(branch_points, list):
        branch_points = np.array(branch_points)

    # Perform PCA on the branch points
    pca = PCA(n_components=1)
    pca.fit(branch_points)

    # Extract the principal component (direction)
    principal_direction = pca.components_[0]

    # Normalize the direction to get a unit vector
    principal_direction /= np.linalg.norm(principal_direction)

    # Ensure the direction aligns with the vector from branch_points[0] to branch_points[1]
    reference_vector = branch_points[1] - branch_points[0]
    if np.dot(principal_direction, reference_vector) < 0:
        principal_direction = -principal_direction

    return principal_direction


def spline_branch_internode_ratios(ratios, num_points):
    n = len(ratios)
    if n == 0:
        return []

    u = np.linspace(0, n - 1, n)
    u_new = np.linspace(0, n - 1, num_points)

    # TODO use "quadratic" if it raises errors
    spline = interp1d(u, ratios, kind="cubic")
    new_ratios = spline(u_new).tolist()

    return new_ratios


def rotate_points(points, angle, origin=(0, 0, 0)):
    """
    Rotate a list of 3D points around the z-axis by a given angle using numpy arrays.

    Args:
        points: List of points [(x, y, z), ...].
        angle: Rotation angle in radians.
        origin: A point (x, y, z) representing the rotation origin.
    Returns:
        list : List of rotated points.
    """
    points = np.array(points)
    ox, oy, _ = origin

    # Translate points to the origin
    translated_points = points - np.array([ox, oy, 0])

    # Rotation matrix for rotation around the z-axis
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )

    # Rotate points
    rotated_points = np.dot(translated_points, rotation_matrix.T)

    # Translate points back
    rotated_points += np.array([ox, oy, 0])

    return rotated_points.tolist()


def model_id_generator(length=31):
    """
        Generate random strings as model ids
    """
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def rotate_mesh(file_path: str, output_path: str, mesh: trimesh.Trimesh = None, angle: float = -np.pi / 2, 
                 axis: np.ndarray = np.array([1, 0, 0]), 
                 rotate_point: np.ndarray = None) -> None:
    """
    Rotates a 3D model around a specified axis by a given angle and saves the rotated model.
    
    Parameters:
    - file_path (str): Path to the input 3D model file.
    - output_path (str): Path where the rotated 3D model will be saved.
    - angle (float): Rotation angle in radians. Default is -90 degrees (-np.pi/2).
    - axis (np.ndarray): A 3D vector specifying the rotation axis. Default is the x-axis ([1, 0, 0]).
    - rotate_point (np.ndarray or None): A 3D point around which to rotate the model. If None, rotates around the model's centroid.
    
    Returns:
    - None: The function modifies the mesh in place and saves the rotated model to `output_path`.
    """
    assert mesh is not None or os.path.isfile(file_path), f"Invalid mesh and {file_path}"
    # Load the model from the specified file path
    mesh = trimesh.load(file_path) if mesh is None else mesh
    
    # Default rotation point is the centroid of the mesh if not provided
    if rotate_point is None:
        rotate_point = mesh.centroid
    
    # Define the rotation matrix based on the given angle, axis, and rotation point
    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=angle,  # Rotation angle in radians
        direction=axis,  # Rotation axis (direction of the vector)
        point=rotate_point  # Point around which to rotate the model
    )
    
    # Apply the rotation to the mesh
    mesh.apply_transform(rotation_matrix)
    
    # Save the rotated model to the specified output path
    mesh.export(output_path)
    print(f"Model successfully saved to {output_path}")


def file_io_json(json_path, content=None, mode='r'):
    if mode == 'r':
        with open(json_path, 'r') as json_file:
            meta_dict = json.load(json_file)
        return meta_dict
    elif mode == 'w':
        assert content is not None, 'Content is None'
        with open(json_path, 'w') as meta_file:
            json.dump(content, meta_file)
        print(f'dump to {json_path}')
        return None
    else:
        raise NotImplementedError