# Real2Sim Apple Tree Generator

Procedural generation system for synthetic apple tree models with realistic branch structures and foliage. Generates 3D meshes (OBJ format) using L-systems (L-Py) with support for hierarchical branching, tapering, and leaf generation.

![Synthetic Apple Tree](cover/image.png)

## Requirements

Install dependencies using the provided setup script:

```bash
bash setup_env.sh
```

**Key Dependencies:**
- Python 3.7+
- OpenAlea L-Py (L-system Python library)
- NumPy, Matplotlib
- PyYAML for configuration

See complete requirements in `setup_env.sh`.

## Configuration

All pipeline parameters are defined in `Config/config.yaml`. The config file controls:

- **Output settings**: OBJ file generation, unit conversion (cm ↔ m), tapering
- **Branch generation**: Number of trees, branch counts, interpolation settings
- **Leaf generation**: Density, size, orientation, phyllotaxis patterns
- **Collision detection**: Mode (single/parallel), branch overlap handling

Example configuration:

```yaml
experiment:
  convert_to_m: True          # Convert units from cm to meters
  taper: True                 # Apply tapering to branches and trunk

parameters:
  num_new_trees: 10           # Number of trees to generate
  branch_obj: True            # Save individual branch OBJ files
  leaf_generation: True       # Enable procedural leaf generation
  leaf_density: [0.3, 0.2, 0.1]  # Leaves per cm for levels 1,2,3+
```

## Input Data Format

Tree structure data is stored in JSON files (e.g., `demo/tree1.json`). Each file contains:

| Field | Type | Description |
|-------|------|-------------|
| `trunk_radius` | `float` | Base radius of tree trunk (cm) |
| `trunk_points` | `List[List[float]]` | 3D skeleton points defining trunk centerline `[[x,y,z], ...]` |
| `branch_internode_ratio` | `List[float]` | Position ratios (0-1) where branches attach to trunk |
| `branch_points` | `List[List[List[float]]]` | 3D skeleton points for each branch `[[[x,y,z], ...], ...]` |
| `branch_points_radius` | `List[List[float]]` | Radius at each point along branches `[[r1, r2, ...], ...]` |

**Example:**
```json
{
  "trunk_radius": 5.2,
  "trunk_points": [[0, 0, 0], [0, 0, 10], [0, 0, 20]],
  "branch_internode_ratio": [0.3, 0.5, 0.7],
  "branch_points": [
    [[0, 0, 6], [2, 0, 8], [4, 0, 10]],
    [[0, 0, 10], [0, 3, 12], [0, 5, 14]]
  ],
  "branch_points_radius": [
    [2.0, 1.5, 1.0],
    [1.8, 1.3, 0.8]
  ]
}
```

## Quick Start

```bash
# Run the basic pipeline
./run.sh

# OR generate new trees with interpolation
python MeshGenerator.py -new_tree_mode
```

## MeshGenerator

Converts tree structure data (JSON) into 3D mesh files (OBJ) using L-system generation. The pipeline creates three main output directories:

- **`lpy`**: L-system files (.lpy) - intermediate representation used to generate meshes
- **`obj`**: 3D mesh files (.obj) - final geometric output for rendering/simulation
- **`meta`**: Metadata for training and tree generation
  - `meta/json`: Combined trunk+branch data for new tree generation
  - `meta/npy`: Individual branch metadata for completion model training

### Basic Output Structure

```
{save_folder}/
├── lpy/           # L-system intermediate files
├── meta/
│   ├── json/      # Combined metadata
│   └── npy/       # Individual branch data
└── obj/           # 3D mesh outputs
```

**IMPORTANT**: Run with `-new_tree_mode` flag to generate interpolated trees:

```shell
python MeshGenerator.py -new_tree_mode
```

### Leaf Generation

The MeshGenerator supports procedural leaf generation using L-Py's built-in leaf primitives. Leaves are automatically generated on branches using phyllotaxis patterns (golden angle spiral).

#### Configuration Parameters

All leaf generation parameters are defined in `Config/config.yaml`:

```yaml
parameters:
  # Leaf Generation Parameters
  leaf_generation: True              # Master toggle for leaf generation
  leaf_density: [0.3, 0.2, 0.1]     # Leaves per cm for branch levels 1,2,3+
                                     # Auto-converts to leaves/m if convert_to_m: True
  leaf_size: [15, 8, 6]             # Leaf length in cm for branch levels 1,2,3+
                                     # Auto-converts to m if convert_to_m: True
  leaf_width_ratio: 0.6              # Width/length ratio (oval shape)
  leaf_orientation_angle: [45, 75]   # Angle range from branch axis (degrees)
  leaf_phyllotaxis_angle: 137.5      # Spiral angle between leaves (golden angle)
  leaf_start_ratio: 0.1              # Skip first 10% of branch (near trunk)
  leaf_end_ratio: 0.95               # Stop at 95% of branch (before tip)
  leaf_obj: True                     # Save individual leaf OBJ files
```


#### Technical Details

- **Module**: `leaf_generator.py` contains core leaf generation logic
- **L-Py Integration**: Uses L-Py's `~l()` leaf primitive with orientation commands (`^`, `+`)
- **Leaf Selection**: Evenly-spaced bud point selection along branch arc length

### Fruit Generation

The MeshGenerator supports procedural fruit (apple) generation using L-Py's built-in sphere primitive. Fruits are automatically generated on branches with realistic hanging physics and clustering behavior.

#### Configuration Parameters

All fruit generation parameters are defined in `Config/config.yaml`:

```yaml
parameters:
  # Fruit Generation Parameters
  fruit_generation: True             # Master toggle for fruit generation (apples)
  fruit_density: [0.05, 0.03, 0.0]  # Fruits per cm for branch levels 1,2,3+
                                     # Auto-converts to fruits/m if convert_to_m: True
  fruit_radius: [6, 5, 0]           # Apple radius in cm for branch levels 1,2,3+
                                     # Auto-converts to m if convert_to_m: True
  fruit_color: [1.0, 0.0, 0.0]      # RGB color (0-1 range): [1,0,0]=red, [0,1,0]=green
  fruit_start_ratio: 0.2             # Skip first 20% of branch (closer to trunk than leaves)
  fruit_end_ratio: 0.8               # Stop at 80% of branch (fruits avoid branch tips)
  fruit_spacing_variation: 0.3       # Randomness in fruit spacing (0=even, 1=fully random)
  fruit_clustering: True             # Enable fruit clustering (multiple fruits per attachment point)
  fruit_cluster_size: [1, 2]        # Number of fruits per cluster [min, max]
  fruit_cluster_spread: 0.02         # Spatial spread of fruit cluster (in meters or cm based on convert_to_m)
  fruit_hang_distance: 3             # Distance fruits hang below branch in cm
                                     # Auto-converts to m if convert_to_m: True
  fruit_obj: True                    # Save individual fruit OBJ files
```

#### Technical Details

- **Module**: `fruit_generator.py` contains core fruit generation logic
- **L-Py Integration**: Uses L-Py's `@O(radius)` sphere primitive with `SetColor(r,g,b)` color commands
- **Fruit Selection**: Spaced point selection along branch arc length with configurable spacing variation
- **Gravity Implementation**: Fruits offset downward in negative Z direction by `fruit_hang_distance` to simulate natural hanging
- **Clustering Algorithm**: Random radial offsets within `cluster_spread` distance, with slight radius variation (95%-105%) per fruit

## NewBranchGenerator

Generates novel branch structures by interpolating between existing branches from real tree data.

### BranchNormalizer

Normalizes branch geometry for interpolation:

1. **Translation**: Moves first point to origin `[0, 0, 0]`
2. **Rotation**: Aligns last point with target vector `[1, 1, 1]`
3. **Output**: Creates `new_tree/` folder with normalized branch metadata

This normalization ensures consistent orientation and scale for interpolation.

### BranchInterpolator

Creates synthetic branches by interpolating between randomly selected normalized branches:

- Samples K nearest branches based on geometric similarity
- Interpolates control points and radii
- Applies length/radius scaling variations
- Saves interpolated branches as JSON files

**Configuration**: Set `num_new_branches` and `num_closest_branch2` in `config.yaml`

### Output Folder Structure

```
{save_folder}/
└── interpolation_num-new-tree-X/
    ├── Y_new_branches/
    │   ├── tree1/              # Branch candidates for tree1
    │   ├── tree2/
    │   └── treeX/
    └── combined_normalized_branches.json
```

## NewTreeGenerator

Assembles complete tree models by combining interpolated trunks with generated branches.

**Pipeline:**
1. **Trunk Generation**: Interpolates new trunks from existing trunk data
2. **Branch Placement**: Attaches branches at interpolated internode positions
3. **Hierarchy Building**: Adds secondary/tertiary branches (configurable depth)
4. **Collision Removal**: Eliminates overlapping branches using spatial checks

### BranchCleaner

Removes branch collisions using geometric overlap detection:

- Calculates point-to-point distances between all branch pairs
- Detects overlap when distance < sum of branch radii at those points
- Removes higher-indexed branch when overlap is detected
- **Special handling**: Skips first 10 skeleton points to preserve trunk-branch connections

### Complete Pipeline Output Structure

After running the full generation pipeline (`python MeshGenerator.py -new_tree_mode`):

```
{save_folder}/
└── interpolation_num-new-tree-X/
    ├── Y_new_branches/
    │   ├── combined_normalized_branches.json    # All normalized branches
    │   ├── tree1/                               # Branch candidates per tree
    │   ├── tree2/
    │   └── treeX/
    ├── new_tree_json/                           # Complete tree definitions
    │   ├── tree1.json
    │   ├── tree2.json
    │   └── treeX.json
    ├── lpy/                                     # L-system intermediate files
    │   └── new_tree_json/
    │       └── tree1/
    │           ├── branch/                      # Branch .lpy files
    │           ├── leaf/                        # Leaf .lpy files
    │           ├── fruit/                       # Fruit .lpy files
    │           ├── trunk/                       # Trunk .lpy files
    │           └── tree1.lpy                    # Complete tree L-system
    ├── meta/                                    # Metadata
    │   ├── json/                                # Combined trunk+branch data
    │   └── npy/                                 # Individual branch data
    └── obj/                                     # 3D mesh outputs
        └── new_tree_json/
            └── tree1/
                ├── branch/                      # Individual branch meshes
                ├── leaf/                        # Individual leaf meshes
                ├── fruit/                       # Individual fruit meshes
                ├── trunk/                       # Trunk mesh
                └── tree1.obj                    # Complete tree mesh (branches + leaves + fruits)
```

#### Individual Tree Output Structure

```
demo_output/
└── obj/
    └── new_tree_json/
        └── tree1/
            ├── branch/
            │   ├── tree1_branch1.obj           # Only branch geometry
            │   ├── tree1_branch2.obj
            │   └── tree1_level2_branch1.obj    # Hierarchical branches
            ├── leaf/
            │   ├── tree1_leaf1.obj             # Only leaf geometry
            │   ├── tree1_leaf2.obj
            │   └── tree1_level2_leaf1.obj      # Hierarchical branch leaves
            ├── fruit/
            │   ├── tree1_fruit1.obj            # Only fruit geometry
            │   ├── tree1_fruit2.obj
            │   └── tree1_level2_fruit1.obj     # Hierarchical branch fruits
            └── tree1.obj                        # Complete tree (branches + leaves + fruits)
```

## Utilities

### Tree Visualization

Visualize generated tree meshes using the built-in visualizer:

```bash
python TreeVisualizer.py
```

**Note**: Loading 10+ trees can be slow due to mesh complexity.

### Coordinate System Transform

Convert to Blender's coordinate system for rendering:

```bash
python align_coor2blender.py
```

This transforms the Y-up coordinate system to Z-up for Blender compatibility.

## Implementation Constraints

The tree generator enforces several botanical and geometric constraints:

1. **Dual Tapering**: Branches taper in two dimensions:
   - Along their own growing direction (natural decrease)
   - From base to apex following trunk height
   - Implemented via `taper_radius()` function in `generate_new_branch()`

2. **Trunk Tapering**: Trunk radius decreases from base to apex
   - Implemented in `generate_new_trunk()`

3. **Radius Constraints**: Branch radius at attachment point must be smaller than trunk cross-section at that height
   - Prevents unrealistic thick branches at tree apex
   - Enforced in `add_branches_to_trunk()`

4. **Hierarchical Growth**: Child branches follow parent branch orientation
   - Children grow toward same side as parent
   - Child length/radius smaller than parent
   - Implemented in `add_branches()` (note: `compute_branch_direction()` has known issues)

5. **Collision Detection**: Multi-level collision checking
   - Checks between: (each-level branches, primary branches)
   - Skips first 10 skeleton points for parent-child connections
   - Implemented in `remove_collision()` and `remove_overlapping_branches()`

## Contributing

When modifying the generator, preserve these botanical constraints to maintain realistic tree structures.

## License

See LICENSE file for details.
