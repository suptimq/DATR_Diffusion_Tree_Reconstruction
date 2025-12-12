# Real2Sim Apple Tree Generator

##  Requirements

See requirements in `setup_env.sh`.

## Config File

All configuration parameters should be defined in a **yaml** file saved in the `config` folder.

## Data Description

A brief description for data in `json` files, for example, `demo/tree1.json`.

- trunk_radius: `float` = Tree trunk radius
- trunk_points: `List[List]` = Tree trunk key points
- branch_internode_ratio: `List` = Ratio of branch origin to root trunk
- branch_points: `List[List[List]]` = Branch key points
- branch_points_radius: `List[List]` = Radius of each branch key point

## Run

```bash
./run.sh
```

## MeshGenerator

This script is designed to process data related to tree structures, generating three directories (i.e., `meta`, `lpy`, and `obj`). The `lpy` folder saves the **.lpy** files as intermedia files to produce the tree meshes as **obj** files saved in `obj` folder. The `meta` folder contains a `json` and a `npy` folder. The `npy` folder has the individual branch meta information (e.g., ***branch_points***) for the training of completion models. The `json` folder has the combined trunk/branch meta information for the generation of new trees.

### Output Folder Structure

```
{save_folder}
├── lpy
├── meta
│   ├── json
│   └── npy
└── obj
```

**THIS IS IMPORTANT!!**

Run `MeshGenerator.py` to produce mesh objects of new trees while setting the `-new_tree_mode` in the command line.

```shell
python MeshGenerator.py -new_tree_mode
```

## NewBranchGenerator

This script generates new branches by interpolating from existing branches.

### BranchNormalizer

It processes JSON files containing path groups (i.e.,branch meta data). It normalizes each path group by translating it to have its first point at the origin `[0, 0, 0]` and rotating it so that the last point aligns with the target vector `[1, 1, 1]`. It produces a new `new_tree` folder containing the combined normalized branch meta infomration.

### BranchInterpolator

It generates new branches based on the existing branches by interpolating from randomly selected branches and saves the new branch data as JSON files.

### Output Folder Structure

```
{save_folder}
├── ...
├── interpolation_num-new-tree-X
│   ├── Y_new_branches
│   │   ├── tree1 (Containing Y branch candidates for tree1)
│   │   ├── tree2
│   │   ├── ...
│   │   └── treeX
│   └── combined_normalized_branches.json
```

## NewTreeGenerator

This script generates a new trunk based on a selection of existing trunks from a JSON file, interpolates branch internode ratios, and optionally adds branches to the new trunk. It combines data from multiple trunks, such as their points, radii, and branch internode ratios, to create a new trunk with a unique structure. The branches are added based on the internode ratios and the positions on the trunk.

### BranchCleaner 

It removes overlapping branches from a JSON file that contains branch points and their radii. It calculates the distance between points on different branches to determine overlap based on their radii. If an overlap is detected, the overlapping branch with higer position in the list is removed from the output.

### Output Folder Structure

```
{save_folder}
├── ...
├── interpolation_num-new-tree-X
│   ├── Y_new_branches
│   │   ├── new_tree_tmp.json
│   │   ├── trunk1.json
│   │   ├── trunk2.json
│   │   ├── ...
│   │   └── trunkX.json
│   └── new_tree_json
│       ├── tree1.json
│       ├── tree2.json
│       ├── ...
│       └── treeX.json
└── ...
```

### Output Folder Structure

```
{save_folder}
├── ...
├── interpolation_num-new-tree-X
│   ├── Y_new_branches
│   ├── combined_normalized_branches.json
│   ├── lpy
│   ├── meta
│   ├── new_tree_json
│   └── obj
└── ...
```

## Others 

### Tree Visualizataion

Run `TreeVisualizer.py` to plot out trees saved in **obj** format. Notice it is pretty slow when loading 10 trees.

### Coordinate System Transform

Run `align_coor2blender.py` to transform to Blender's coordinate system for rendering.

## Constraints

1. Generated branches should have two tapers: 1) taper along the its own growing direction and 2) taper from bottom to top (trunk growing direction). This is achieved by setting the **decrease_factor** and the **taper_radius** function in `generate_new_branch`.

2. Generated trunkes should have the second taper property as implemented in `generate_new_trunk`.

3. The branches added to the particular height of the trunk should have a radius that is smaller than the cross-section of the trunk. This prevents giant branches at the top of the tree in `add_branches_to_trunk`.

4. The children branches added to the parent branches should grow towards the same side of the tree and the length/radius of the children branches should be smaller than those of the parent branches in `add_branches`. (This is still buggy - could be `compute_branch_direction`)

5. The collision check is conducted between (each level branches, primary branches). A hacky solution is used to detect the collision between a child branch and its parent branch where the first 10 skeleton points were skipped in `remove_collision` and `remove_overlapping_branches`.