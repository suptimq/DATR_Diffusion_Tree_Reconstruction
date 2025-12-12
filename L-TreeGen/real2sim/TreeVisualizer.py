import os
import trimesh
import fpsample

import numpy as np
import open3d as o3d


def o3d_pc_wrapper(pts, colors=None):
    """
    Convert ndarray to o3d.geometry.PointCloud
    """

    if isinstance(pts, o3d.cpu.pybind.geometry.PointCloud):
        return pts

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts)

    if colors is not None:
        assert colors.shape == pts.shape, "Shape Not Match for Color"
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud


def assemble_trees(stl_folder, skeleton_folder, num_vis=5):
    """
    Function to assemble trees from STL files starting with 'tree'.

    Args:
        stl_folder (str): Path to the data folder containing STL files.
        skeleton_folder (str): Path to the skeleton folder containing NPZ files.
    Returns:
        list: List of Open3D geometry objects, each representing a tree.
    """
    trees = []
    skeletons = []
    tree_filenames = [
        f for f in os.listdir(stl_folder) if f.startswith("tree") and f.endswith(".stl")
    ]
    skeleton_filenames = [f.replace(".stl", ".npz") for f in tree_filenames]

    for index, (tree_filename, skeleton_filename) in enumerate(
        zip(tree_filenames, skeleton_filenames)
    ):
        if index > num_vis:
            return trees, skeletons
        stl_path = os.path.join(stl_folder, tree_filename)
        skeleton_path = os.path.join(skeleton_folder, skeleton_filename)

        # Load mesh and skeleton info
        tree_mesh = o3d.io.read_triangle_mesh(stl_path)
        skeleton_dict = np.load(skeleton_path)

        # Add offset=index*1 for visualization
        offset = np.array([index, 0, 0])
        skeleton_coords = skeleton_dict["centers"] + offset
        skeletons.append(o3d_pc_wrapper(skeleton_coords))

        # Translate the tree along the x-axis
        tree_mesh.translate(offset)
        trees.append(tree_mesh)

    return trees, skeletons


def obj_to_stl(obj_folder, stl_folder):
    tree_filenames = [
        f for f in os.listdir(obj_folder) if f.startswith("tree") and f.endswith(".obj")
    ]

    for tree_filename in tree_filenames:
        obj_path = os.path.join(obj_folder, tree_filename)
        stl_path = os.path.join(stl_folder, tree_filename.replace(".obj", ".stl"))

        if os.path.exists(stl_path):
            print(f"skip existing {stl_path}")
            continue

        mesh1 = trimesh.load_mesh(obj_path)
        mesh1.export(stl_path, file_type="stl")
        print(f"save to {stl_path}")


if __name__ == "__main__":
    obj_folder = r"D:\Data\LPy\LT81_New_Tree\new_tree\obj\new_tree_json"
    stl_folder = r"D:\Data\LPy\LT81_New_Tree\new_tree\stl\new_tree_json"
    skeleton_folder = r"D:\Data\LPy\LT81_New_Tree\new_tree\meta\npy"

    os.makedirs(stl_folder, exist_ok=True)
    obj_to_stl(obj_folder, stl_folder)

    num_vis = 10
    trees, skeletons = assemble_trees(stl_folder, skeleton_folder, num_vis=num_vis)
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # Enable showing axis
    # vis.get_render_option().show_coordinate_frame = True

    fps_skeletons = [skeleton.farthest_point_down_sample(100) for skeleton in skeletons]

    for index, obj in enumerate(trees):
        if index < num_vis:
            vis.add_geometry(obj)

    # Set up view control
    view_control = vis.get_view_control()
    view_control.rotate(90.0, 0.0)  # Rotate around y-axis by 180 degrees

    # Run the visualization
    vis.run()
