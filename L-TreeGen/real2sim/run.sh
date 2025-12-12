python MeshGenerator.py
python NewBranchGenerator.py
python NewTreeGenerator.py

# Metadata cannot be saved properly using MP
python MeshGenerator_MultiProcessing.py -new_tree_mode
# Do not set the obj saving flag so it will load from the config yaml file
python MeshGenerator.py -new_tree_mode