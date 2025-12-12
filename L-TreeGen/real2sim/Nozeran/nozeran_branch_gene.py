"""
This scripts aims to generate a simulation dataset that puts minimum constrains to branches.
Pls refer to https://lpy.readthedocs.io/en/latest/user/tutorial.html#nozeran for original lpy file.
The reason why this script is based on Nozeran is they share similar branch patterns to some extent.
"""
import os
import random

from openalea import lpy


# Function to generate Lpy content
def generate_lpy_content(stagelength=4):
    # Randomize nbcycle
    index = random.randint(0, stagelength)
    nbcycle = random.randint(1, 8)  # Random integer between 1 and 8
    nbaxe = random.randint(3, 7)    # Random integer between 1 and 8
    radius = random.uniform(0.005, 0.05)  # Random float between 0.005 to 0.05 (m)
    
    # L-Py indent requirement
    content = f"""
stagelength = {stagelength}     # number of step between two verticille production
nbcycle = {nbcycle}         # total number of verticille branches wanted 
radinc = 0.0005      # increment of radius through time

def branch_angle(nc):         # branching angle according to position on trunk
    return 30 + 60 * ((nbcycle - nc) / float(nbcycle))

module A # represent trunk apical meristem
module B # represent apical meristem of lateral branches
module I # Internode

Axiom: SectionResolution(60) A({index})

derivation length: nbcycle * stagelength
production:

A(t) :
    cyclenb = t // stagelength
    nproduce @Gc
    nproduce [/(360 * t / {nbaxe}) & (branch_angle(cyclenb)) B]
    nproduce @Ge

B --> I(0.05, {radius}) B

I(s, r) --> I(s, r+radinc)

homomorphism:

I(a, r) --> SetWidth(r) F(a, r - radinc)

endlsystem
"""
    return content


# Function to process Lpy files
def process_lpy_files(input_directory, output_directory):
    lpy_files = [f for f in os.listdir(input_directory) if f.endswith('.lpy')]
    for lpy_file in lpy_files:
        full_input_path = os.path.join(input_directory, lpy_file)
        lsystem = lpy.Lsystem(full_input_path)
        scene = lsystem.sceneInterpretation(lsystem.iterate())
        output_file_name = os.path.splitext(lpy_file)[0] + ".obj"
        full_output_path = os.path.join(output_directory, output_file_name)
        scene.save(full_output_path)


if __name__ == '__main__':

    num_branch = 1432
    save_folder = r"E:\Data\LPy\Ablation_Apple\lpy"
    output_folder = r"E:\Data\LPy\Ablation_Apple\obj\row\tree\Branch"
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    for branch_idx in range(num_branch):
        lpy_content = generate_lpy_content()
        lpy_filename = f"branch{branch_idx+1}.lpy"
        lpy_filepath = os.path.join(save_folder, lpy_filename)
        with open(lpy_filepath, 'w') as file:
            file.write(lpy_content)

    process_lpy_files(save_folder, output_folder)