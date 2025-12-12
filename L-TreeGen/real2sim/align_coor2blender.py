import os
import shutil
import trimesh
from multiprocessing import Pool, cpu_count
from utils import rotate_mesh, model_id_generator, file_io_json

OBJ_EXT = ".obj"

# Function to merge and process branch files
def process_branch_files(folder_name, branch_files):
    # Group files based on their branch identifier
    branch_meshes = {}
    for file_path in branch_files:
        file_name = os.path.basename(file_path)[:-4]
        branch_id = file_name.split("_")[-1]
        if branch_id not in branch_meshes:
            branch_meshes[branch_id] = []
        branch_meshes[branch_id].append(file_path)

    # Load, merge, and process each branch group
    results = []
    for branch_id, files in branch_meshes.items():
        meshes = [trimesh.load(file) for file in files]
        merged_mesh = trimesh.util.concatenate(meshes)

        # Generate a unique hash ID for the merged branch
        hash_id = model_id_generator(length=32)

        # Define the output path for the rotated model
        output_filename = f"{hash_id}{OBJ_EXT}"
        output_path = os.path.join(branch_output_dir, output_filename)

        # Rotate and save the merged mesh
        rotate_mesh(file_path=None, output_path=output_path, mesh=merged_mesh)

        results.append(
            {
                "output_path": output_path,
                "hash_id": hash_id,
                "mapping_key": f"{folder_name}/{branch_id}",
            }
        )
    return results


# Function to process a single OBJ file
def process_obj_file(args):
    _, input_path = args
    # input_path = os.path.join(root_dir, folder_name, file_name)

    # Generate a unique hash ID for the file
    hash_id = model_id_generator(length=32)

    # Define the output path for the rotated model
    output_filename = f"{hash_id}{OBJ_EXT}"
    output_path = os.path.join(tree_output_dir, output_filename)

    # Rotate and save the model
    rotate_mesh(input_path, output_path)
    return {"output_path": output_path, "hash_id": hash_id, "mapping_key": input_path}


if __name__ == "__main__":

    # Define directories and files
    data_folder = "demo_output/interpolation_num-new-tree-10"
    root_dir = os.path.join(data_folder, "obj/new_tree_json")
    output_dir = os.path.join(data_folder, "rotated_obj")
    tree_output_dir = os.path.join(output_dir, "tree")
    branch_output_dir = os.path.join(output_dir, "branch")

    # Delete the output directory if it already exists
    if os.path.exists(output_dir):
        print(f"Found {output_dir} and Removed!")
        shutil.rmtree(output_dir)

    # Ensure the output directory exists
    os.makedirs(tree_output_dir, exist_ok=True)
    os.makedirs(branch_output_dir, exist_ok=True)

    output_file_json = os.path.join(output_dir, "tree_obj_filepaths.json")
    output_file_hashid_json = os.path.join(output_dir, "tree_obj_hashid_filepaths.json")
    output_file_txt = os.path.join(output_dir, "tree_obj_to_hashid_mapping.txt")

    branch_output_file_json = os.path.join(output_dir, "branch_obj_filepaths.json")
    branch_output_file_hashid_json = os.path.join(
        output_dir, "branch_obj_hashid_filepaths.json"
    )
    branch_output_file_txt = os.path.join(
        output_dir, "branch_obj_to_hashid_mapping.txt"
    )

    # Gather all .obj files to process
    tree_files = []
    branch_folders = []

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        # Check if it's a tree folder
        if os.path.isdir(folder_path):
            branch_folder = os.path.join(folder_path, "branch")
            assert os.path.exists(branch_folder), f"{branch_folder} Not Found"
            branch_files = []
            for file_name in os.listdir(branch_folder):
                if file_name.endswith(OBJ_EXT):
                    branch_files.append(os.path.join(branch_folder, file_name))
            if branch_files:
                branch_folders.append((folder_name, branch_files))
        # Regular folder with .obj files
        else:
            if folder_path.endswith(OBJ_EXT):
                tree_files.append((folder_name, folder_path))

    # Unit test
    # process_obj_file(tree_files[0])
    # process_branch_files(*branch_folders[0])

    # Use multiprocessing to process files in parallel
    with Pool(processes=cpu_count()) as pool:
        tree_results = pool.map(process_obj_file, tree_files)
        branch_results = pool.starmap(process_branch_files, branch_folders)

    # Filter out None results (failed processes)
    tree_results = [result for result in tree_results if result is not None]
    branch_results = [result for result in branch_results if result is not None]

    # Save the tree and branch results separately
    tree_rotated_obj_files = [r["output_path"] for r in tree_results]
    tree_rotated_obj_hashids = [r["hash_id"] for r in tree_results]
    tree_obj_to_hashid = {r["mapping_key"]: r["hash_id"] for r in tree_results}

    from itertools import chain

    # After processing the branch files, flatten the results
    branch_results = list(chain.from_iterable(branch_results))

    branch_rotated_obj_files = [r["output_path"] for r in branch_results]
    branch_rotated_obj_hashids = [r["hash_id"] for r in branch_results]
    branch_obj_to_hashid = {r["mapping_key"]: r["hash_id"] for r in branch_results}

    # Save JSON files for trees and branches
    file_io_json(output_file_json, tree_rotated_obj_files, mode="w")
    file_io_json(output_file_hashid_json, tree_rotated_obj_hashids, mode="w")

    file_io_json(branch_output_file_json, branch_rotated_obj_files, mode="w")
    file_io_json(branch_output_file_hashid_json, branch_rotated_obj_hashids, mode="w")

    # Save the mapping files for trees and branches
    with open(output_file_txt, "w") as txt_file:
        for key, hash_id in tree_obj_to_hashid.items():
            txt_file.write(f"{key}\t{hash_id}\n")

    with open(branch_output_file_txt, "w") as txt_file:
        for key, hash_id in branch_obj_to_hashid.items():
            txt_file.write(f"{key}\t{hash_id}\n")

    print(
        f"Processed {len(tree_results)} tree files and {len(branch_results)} branch files. Outputs saved."
    )
