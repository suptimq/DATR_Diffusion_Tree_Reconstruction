import os
from openalea import lpy


def generate_lpy_content(modules, points_list, radii_list):
    """
    Generates L-system content for multiple modules with corresponding paths and radii.

    Args:
        modules (list): A list of L-system module names.
        points_list (list): A list of lists of 3D points (x, y, z) representing the path of each structure.
        radii_list (list): A list of lists of radii corresponding to each point in each path.

    Returns:
        str: Generated L-system content.
    """

    # Join module names with a space
    module_line = "module " + ", ".join(modules)

    # L-Py indent requirement
    content = f"{module_line}\n"

    # Join module names with their corresponding SectionResolution commands
    axiom_line = "Axiom: " + " ".join(
        [f"SectionResolution(30) {module}()" for module in modules]
    )

    # Append axiom line to content
    content += f"{axiom_line}\n"

    content += "derivation length: 1\n"
    content += "production:\n"

    # Iterate over modules, points, and radii
    for module, points, radii in zip(modules, points_list, radii_list):
        # Generate content for each module
        content += f"""
{module}():
    x, y, z = {points[0]}
    r = {radii[0]}
    nproduce !(r) MoveTo(x, y, z)
    nproduce @Gc
    for point, radius in zip({points[1:]}, {radii[1:]}):
        x, y, z = point
        nproduce !(radius) LineTo(x, y, z)
    nproduce @Ge

"""

    content += "interpretation:\n"
    content += "endlsystem"

    return content


def generate_lpy_content_with_leaves(modules, points_list, radii_list, leaf_data_list=None):
    """
    Generates L-system content with leaf support using L-Py's built-in leaf primitives.
    Reference: https://lpy.readthedocs.io/en/latest/index.html

    Args:
        modules (list): A list of L-system module names.
        points_list (list): A list of lists of 3D points representing each structure.
        radii_list (list): A list of lists of radii corresponding to each point.
        leaf_data_list (list, optional): List of leaf data lists for each module. Each leaf has:
            - 'position': [x,y,z]
            - 'pitch_angle': float (degrees, for ^ command)
            - 'rotation_angle': float (degrees, for + command)
            - 'size': float (leaf length)

    Returns:
        str: Generated L-system content with leaves.
    """
    # Join module names with a space
    module_line = "module " + ", ".join(modules)

    content = f"{module_line}\n"

    # Join module names with their corresponding SectionResolution commands
    axiom_line = "Axiom: " + " ".join(
        [f"SectionResolution(30) {module}()" for module in modules]
    )

    content += f"{axiom_line}\n"
    content += "derivation length: 1\n"
    content += "production:\n"

    # Iterate over modules, points, radii, and optional leaf data
    for idx, (module, points, radii) in enumerate(zip(modules, points_list, radii_list)):
        # Get leaf data for this module (if available)
        leaf_data = []
        if leaf_data_list and idx < len(leaf_data_list) and leaf_data_list[idx]:
            leaf_data = leaf_data_list[idx]

        # Generate branch content
        content += f"""
{module}():
    x, y, z = {points[0]}
    r = {radii[0]}
    nproduce !(r) MoveTo(x, y, z)
    nproduce @Gc
    for point, radius in zip({points[1:]}, {radii[1:]}):
        x, y, z = point
        nproduce !(radius) LineTo(x, y, z)
    nproduce @Ge
"""

        # Add leaves if available
        if leaf_data:
            # Generate individual nproduce statements for each leaf to avoid L-Py parser warnings
            for leaf in leaf_data:
                pos = leaf['position']
                pitch = leaf['pitch_angle']
                rotation = leaf['rotation_angle']
                size = leaf['size']
                content += f"""    nproduce [
    nproduce MoveTo({pos[0]}, {pos[1]}, {pos[2]})
    nproduce ;(2)
    nproduce ^({pitch})
    nproduce +({rotation})
    nproduce ~l({size})
    nproduce ]
"""

    content += "interpretation:\n"
    content += "endlsystem"

    return content


def generate_leaf_only_lpy_content(module_name, leaf_data_list):
    """
    Generates L-system content with ONLY leaves (no branch geometry).

    Args:
        module_name (str): Name for the leaf module (e.g., "Leaf1")
        leaf_data_list (list): List of leaf data dicts with:
            - 'position': [x,y,z]
            - 'pitch_angle': float (degrees)
            - 'rotation_angle': float (degrees)
            - 'size': float (leaf length)

    Returns:
        str: L-system content with only leaf geometry
    """
    content = f"module {module_name}\n"
    content += f"Axiom: {module_name}()\n"
    content += "derivation length: 1\n"
    content += "production:\n"
    content += f"\n{module_name}():\n"

    # Generate leaves without any branch geometry
    for leaf in leaf_data_list:
        pos = leaf['position']
        pitch = leaf['pitch_angle']
        rotation = leaf['rotation_angle']
        size = leaf['size']
        content += f"""    nproduce [
    nproduce MoveTo({pos[0]}, {pos[1]}, {pos[2]})
    nproduce ;(2)
    nproduce ^({pitch})
    nproduce +({rotation})
    nproduce ~l({size})
    nproduce ]
"""

    content += "interpretation:\n"
    content += "endlsystem"

    return content


# Function to process Lpy files
def process_lpy_files(input_directory, output_directory):
    lpy_files = [f for f in os.listdir(input_directory) if f.endswith(".lpy")]
    for lpy_file in lpy_files:
        full_input_path = os.path.join(input_directory, lpy_file)
        lsystem = lpy.Lsystem(full_input_path)
        scene = lsystem.sceneInterpretation(lsystem.iterate())
        output_file_name = os.path.splitext(lpy_file)[0] + ".obj"
        full_output_path = os.path.join(output_directory, output_file_name)
        scene.save(full_output_path)
