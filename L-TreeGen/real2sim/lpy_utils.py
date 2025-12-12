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
