import os
import laspy
import shutil
import subprocess

import numpy as np
import open3d as o3d


SCALE_FACTOR = 3 # Scale up tiny branches to avoid no-hit (Fixed in newer version)
# TODO set to False leads to non-intersection beams
INCLUDE_GROUND = True
GROUND_SCALE_FACTOR = 20
SCANNER_Y_OFFSET = 6 # Position of the scanner
SCANNER_LIST = ["demo", "field_setup"]

data_folder = os.path.join(os.getcwd(), "demo")
data_folder_base = os.path.basename(data_folder)

helios_folder = os.path.join(os.getcwd(), "helios-plusplus-lin")
survey_xml_folder = os.path.join(helios_folder, f"data/surveys/{data_folder_base}")
scene_xml_folder = os.path.join(helios_folder, f"data/scenes/{data_folder_base}")

os.makedirs(survey_xml_folder, exist_ok=True)
os.makedirs(scene_xml_folder, exist_ok=True)


def create_survey_xml(survey_xml_filepath, scene_xml_filepath, tree_num, scanner_id):
    """
    Configure scanner settings and generate an XML configuration file for the scanner:
        1. The `pulseFreq_hz` is a sensor property, which can be found in the datasheet
        2. The `verticalResolution_deg` and `horizontalResolution_deg` could be calculated based on user setting
            - For 6.1mm @ 10m distance, the angular resolution = arctan(0.00061) = 0.03495
        3. The `headRotateStart/Stop_deg` needs to be determined manually

    Args:
        survey_xml_filepath (str): Path to the survey XML file.
        scene_xml_filepath (str): Path to the scene XML file.
        tree_num (int): Number of trees to be scanned.
        scanner_id (int): Index of the scanner settings to use from SCANNER_LIST.
    """
    xml_output = f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
    <!-- Default scanner settings: -->
     <scannerSettings id="demo" active="true" pulseFreq_hz="600000" verticalResolution_deg="0.3" horizontalResolution_deg="0.3" verticalAngleMin_deg="-60" verticalAngleMax_deg="240"/>
     <scannerSettings id="field_setup" active="true" pulseFreq_hz="600000" verticalResolution_deg="0.03" horizontalResolution_deg="0.03" verticalAngleMin_deg="-60" verticalAngleMax_deg="240"/>
    <survey name="orchard_demo_tls" scene="{scene_xml_filepath}#orchard_demo" platform="data/platforms.xml#tripod" scanner="data/scanners_tls.xml#panoscanner">
        <FWFSettings binSize_ns="0.2" beamSampleQuality="3" />
        <leg>
            <platformSettings x="{tree_num}" y="{SCANNER_Y_OFFSET}" onGround="true" />
            <scannerSettings template="{SCANNER_LIST[scanner_id]}" headRotateStart_deg="-90.0" headRotateStop_deg="90.0"/>
        </leg>
        <leg>
            <platformSettings x="{tree_num}" y="-{SCANNER_Y_OFFSET}" onGround="true" />
            <scannerSettings template="{SCANNER_LIST[scanner_id]}" headRotateStart_deg="90.0" headRotateStop_deg="270.0"/>
        </leg>
    </survey>
</document>"""

    with open(survey_xml_filepath, "w") as file:
        file.write(xml_output)


def create_scene_xml(panel_folder, scene_xml_filepath):
    """
    Generate an XML configuration file for loading and scaling OBJ files.

    Args:
        panel_folder (str): Path to the folder containing the panel OBJ files.
        scene_xml_filepath (str): Path to the output XML file.

    Returns:
        int: Number of unique tree IDs found in the panel folder.
    """
    destination_folder = os.path.join(helios_folder, r"data/sceneparts/orchard")

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    xml_output = f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
    <scene id="orchard_demo" name="TLS Panel Simulation">"""
    
    if INCLUDE_GROUND:
        xml_output += f"""
        <part id="0">
            <filter type="objloader">
                <param type="string" key="filepath" value="data/sceneparts/basic/groundplane/groundplane.obj"/>
            </filter>
            <filter type="scale">
                <param type="double" key="scale" value="{GROUND_SCALE_FACTOR}" />
            </filter>
        </part>"""

    obj_counter = 1
    Map = {}
    tree_ids = []
    for root, _, files in os.walk(panel_folder):
        for obj_file in files:
            if obj_file.endswith(".obj"):
                parts = obj_file.split("_")
                tree_id = int(parts[0].replace("tree", ""))
                if not tree_id in tree_ids:
                    tree_ids.append(tree_id)

                obj_part_id = obj_file.split(".")[0]

                Map[obj_counter] = obj_part_id

                obj_filepath = os.path.join(root, obj_file)

                xml_output += f"""
        <part id="{obj_counter}">
            <filter type="objloader">
                <param type="string" key="filepath" value="{obj_filepath}"/>
                <param type="string" key="up" value="z" />
            </filter>
            <filter type="scale">
                <param type="double" key="scale" value="{SCALE_FACTOR}" />
            </filter>    
        </part>"""
                obj_counter += 1

    xml_output += """
    </scene>
</document>"""

    with open(scene_xml_filepath, "w") as file:
        file.write(xml_output)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    map_filepath = os.path.join(script_dir, "Map.txt")

    with open(map_filepath, "w") as map_file:
        for key, value in Map.items():
            map_file.write(f"{key}: {value}\n")

    return len(tree_ids)


def get_most_recent_folder(path):
    """
    Get the most recently modified folder in the specified path.

    Args:
        path (str): The directory path to search.

    Returns:
        str: The path of the most recently modified folder.
    """
    folders = [
        os.path.join(path, folder)
        for folder in os.listdir(path)
        if os.path.isdir(os.path.join(path, folder))
    ]
    most_recent_folder = max(folders, key=os.path.getmtime)
    return most_recent_folder


def read_map_file(map_filepath):
    """
    Read a mapping file that associates IDs with names.

    Args:
        map_filepath (str): Path to the mapping file.

    Returns:
        dict: A dictionary mapping IDs to names.
    """
    id_to_name = {}
    with open(map_filepath, "r") as map_file:
        for line in map_file:
            id, name = line.strip().split(": ")
            id_to_name[int(id)] = name
    return id_to_name


def segment_las_by_hit_object_id(input_path, output_dir, id_to_name):
    """
    Segment a LAS file by `hitObjectId` and save the segments as PCD files.

    Args:
        input_path (str): Path to the input LAS file.
        output_dir (str): Directory to save the segmented PCD files.
        id_to_name (dict): Dictionary mapping object IDs to names.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the input LAS file
    las = laspy.read(input_path)
    # Access scaled coordinates
    coords = np.vstack((las.x, las.y, las.z)).transpose()
    scaled_coords = coords / SCALE_FACTOR

    # Access the hitObjectId attribute
    hit_object_ids = las["hitObjectId"]
    # Get unique hitObjectIds excluding 0 if it's used for unclassified points
    unique_ids = np.unique(hit_object_ids)

    # Segment and write each object to a separate LAS file
    for object_id in unique_ids:
        if object_id != 0:
            mask = hit_object_ids == object_id

            output_filename = f'{id_to_name.get(object_id, f"hitObjectId_{object_id}")}'

            # # Save scaled points as las format
            # segmented_points = las.points[mask].copy()
            # output_file = laspy.LasData(las.header)
            # output_file.points = segmented_points

            # output_filepath = os.path.join(output_dir, f'{output_filename}.las')
            # output_file.write(output_filepath)

            # Scale back and save as pcd format
            segmented_points = scaled_coords[mask]
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(segmented_points)

            output_filepath = os.path.join(output_dir, f"{output_filename}.pcd")
            o3d.io.write_point_cloud(output_filepath, point_cloud)


def segment_raw_helios_las(raw_helios_output_folder):
    """
    Process raw las files generated by Helios++ to segment individual obj files by `hitObjectId`.

    Args:
        raw_helios_output_folder (str): Path to the folder containing raw Helios output.
    """
    segmented_helios_output_folder = data_folder + "_segmented"

    os.makedirs(segmented_helios_output_folder, exist_ok=True)

    # [panel1_processed, panel2_processed, ..., panelX_processed]
    processed_panel_folders = os.listdir(raw_helios_output_folder)
    
    for processed_panel_folder in processed_panel_folders:
        
        # Create folder to save segmented las/pcd files
        segmented_folder_name = processed_panel_folder.replace("_processed", "")
        segmented_folder_path = os.path.join(
            segmented_helios_output_folder, segmented_folder_name
        )
        os.makedirs(segmented_folder_path, exist_ok=True)

        processed_panel_folder_path = os.path.join(raw_helios_output_folder, processed_panel_folder)

        # Get the most recent output folder path
        most_recent_output = get_most_recent_folder(processed_panel_folder_path)

        map_file_path = os.path.join(processed_panel_folder_path, "Map.txt")
        assert os.path.exists(map_file_path), f"{map_file_path} Not Found"
        id_to_name = read_map_file(map_file_path)

        # Segment the las files generated by Helios++ into organ obj files
        las_files = [
            f for f in os.listdir(most_recent_output) if f.endswith(".las")
        ]
        leg_folders = []
        for las_file in las_files:
            leg_id = las_file.split("_")[0]
            leg_output_folder = os.path.join(segmented_folder_path, leg_id)
            os.makedirs(leg_output_folder, exist_ok=True)

            leg_folders.append(leg_output_folder)

            input_las_file = os.path.join(most_recent_output, las_file)
            segment_las_by_hit_object_id(
                input_las_file, leg_output_folder, id_to_name
            )

        # Merge partial pcd files from different legs
        # {'tree1_branch1': ['leg000/tree1_branch1.pcd', 'leg001/tree1_branch1.pcd], }
        pcd_data = {}
        for leg_folder in leg_folders:
            pcd_files = [x for x in os.listdir(leg_folder) if x.endswith(".pcd")]
            for pcd_file in pcd_files:
                pcd_filepath = os.path.join(leg_folder, pcd_file)
                pcd = o3d.io.read_point_cloud(pcd_filepath)
                obj_id = pcd_file.split(".")[0]
                if obj_id not in pcd_data:
                    pcd_data[obj_id] = []
                pcd_data[obj_id].append(pcd)

        merged_folder = os.path.join(segmented_folder_path, "merged_legs")
        os.makedirs(merged_folder, exist_ok=True)

        # Save merged PCD files
        for object_id, pcd_list in pcd_data.items():
            merged_pcd = o3d.geometry.PointCloud()
            for pcd in pcd_list:
                merged_pcd += pcd
            merged_pcd_file_path = os.path.join(merged_folder, f"{object_id}.pcd")
            o3d.io.write_point_cloud(merged_pcd_file_path, merged_pcd)


def run_simulation(panel_folder, panel_folder_base, raw_helios_output_folder, scanner_id=0):
    """
    Run the Helios++ simulation for each panel in the base folder path.

    Args:
        panel_folder (str): Path to the panel folder containing obj files.
        panel_folder_base (str): panel folder name.
        raw_helios_output_folder (str): Path to the folder to save raw Helios output.
        scanner_id (int): Index of the scanner settings to use from SCANNER_LIST.
    """

    raw_helios_panel_folder = os.path.join(
        raw_helios_output_folder, panel_folder_base + "_processed"
    )

    if not os.path.exists(raw_helios_panel_folder):
        os.makedirs(raw_helios_panel_folder)

    # Relative path to the Helios++ folder
    survey_xml_filepath = os.path.join(survey_xml_folder, f"{panel_folder_base}.xml")
    scene_xml_filepath = os.path.join(scene_xml_folder, f"{panel_folder_base}.xml")

    # Override object scene XML file with the current panel's configuration
    tree_num = create_scene_xml(panel_folder, scene_xml_filepath)
    # Override survey XML file with current scanner settings
    create_survey_xml(survey_xml_filepath, scene_xml_filepath, tree_num=tree_num, scanner_id=scanner_id)

    # Command to run Helios++ simulation
    cmd = (
        rf"run/helios {survey_xml_filepath} "
        "--writeWaveform --lasOutput --parallelization 0 --nthreads 0 --chunkSize 32 --warehouseFactor 4 -vv"
    )
    # Run the command
    subprocess.run(cmd, cwd=helios_folder, shell=True)

    # Get the most recent output folder path
    base_output_folder = os.path.join(helios_folder, r"output/orchard_demo_tls")
    os.makedirs(base_output_folder, exist_ok=True)
    most_recent_output = get_most_recent_folder(base_output_folder)

    # Copy the most recent output folder to the raw Helios panel folder
    shutil.copytree(
        most_recent_output,
        os.path.join(
            raw_helios_panel_folder, os.path.basename(most_recent_output)
        ),
    )

    # Copy the Map file to the raw Helios panel folder
    map_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "Map.txt"
    )
    shutil.copy(map_file_path, raw_helios_panel_folder)


if __name__ == "__main__":

    raw_helios_output_folder = data_folder + "_processed"

    if not os.path.exists(raw_helios_output_folder):
        os.makedirs(raw_helios_output_folder)

    scanner_id = 0  # 0 - demo (quick), 1 - real field config (slow)
    assert scanner_id < len(
        SCANNER_LIST
    ), f"{scanner_id} is out of length of {SCANNER_LIST}"

    panel_folders = [
        os.path.join(data_folder, x)
        for x in os.listdir(data_folder)
        if x.startswith("panel") and 
        os.path.isdir(os.path.join(data_folder, x)) 
    ]

    for panel_folder in panel_folders:
        
        panel_folder_base = os.path.basename(panel_folder)

        run_simulation(panel_folder, panel_folder_base, raw_helios_output_folder, scanner_id)

    segment_raw_helios_las(raw_helios_output_folder)
