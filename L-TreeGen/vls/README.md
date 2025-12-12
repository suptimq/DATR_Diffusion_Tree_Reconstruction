# HELIOS++ Tree Segmentation Script

This script processes and segments `.obj` files using HELIOS++, generating LAS files. Ensure the processed and segmented directories do not exist in the same directory as the input folder before running the script. Expects already offset obj files.

## Requirements

1. **Conda**: Ensure you have Conda installed.
2. **HELIOS++**: Download source code from [HELIOS++](https://github.com/3dgeo-heidelberg/helios/releases/tag/v1.3.0) and extract it in this folder.
   > Notice the provided [conda environment yaml file](https://github.com/3dgeo-heidelberg/helios/blob/dev/conda-environment.yml) specified python=3.8, which does not work. We provided the updated yaml file for python environment. Additionally, follow [this](https://github.com/3dgeo-heidelberg/helios/issues/354#issuecomment-1513354483) to update the **ddl** accordingly. For Linux, follow [this](https://github.com/3dgeo-heidelberg/helios/issues/396#issuecomment-1816624088) to set the links.

## Setting Up

### Step 1: Activate the HELIOS++ Environment

Activate the HELIOS++ environment:
```sh
conda activate pyhelios_env
```

### Step 2: Modify the Script if Necessary

Modify the `data_folder` and `helios_folder` in the script matches your folders:

```python
data_folder = "YOUR_DATA_FOLDER"
helios_folder = "YOUR_HELIOS_FOLDER"
```

### Step 3: Run the Script

```sh
chmod +x heliosShell.sh
./heliosShell.sh
```

## Input and Output Structure

### Input Structure

Ensure your input directory follows a similar structure:
```
demo
    ├── panel1
        ├── tree1
        │   ├── tree1_branch1.obj
        │   ├── tree1_branch2.obj
        │   └── tree1_trunk.obj
        ├── tree2
        │   ├── tree2_branch1.obj
        │   ├── tree2_branch2.obj
        │   └── tree2_trunk.obj
        └── ...
    ├── panel2
    └── ...
```

### First Output Structure

```
demo_processed
├── panel1_processed
│   ├── panel1 2024-07-28_19-41-23
│   │   ├── leg000_points.las
│   └── Map.txt
├── panel2_processed
│   ├── panel2 2024-07-28_19-45-10
│   │   ├── leg000_points.las
│   └── Map.txt
└── ...
```

### Second Output Structure

```
demo_segmented
├── panel1
│   ├── tree1_branch1.las
│   ├── tree1_branch2.las
│   └── tree1_trunk.las
├── panel2
│   ├── tree2_branch1.las
│   ├── tree2_branch2.las
│   └── tree2_trunk.las
└── ...
```

## Parameters

There are two different scanner settings defined in the script where `field_setup` provides higher resolution than `demo`.