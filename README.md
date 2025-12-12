# Real2Sim Closed Loop Framework for 3D Tree Reconstruction

This workspace contains two related projects used for 3D apple tree reconstruction and synthetic data generation.

## DATR

- Location: `DATR`
- Purpose: The main reconstruction framework (DATR) that converts single or sparse-view images into 3D tree meshes using a diffusion-based multi-view generator (Zero123++) and a Large Reconstruction Model (LRM).
- Key subfolders:
  - `configs` — YAML configs used by `run.py` and `train.py`.
  - `data` — Example datasets.

## L-TreeGen

- Location: `L-TreeGen`
- Purpose: Synthetic apple-tree generator and renderer. Uses L-Py (L-systems), Blender, Helios++ (VLS), and custom Blender scripts to produce realistic RGB-D and point-cloud data for training DATR.
- Key subfolders:
  - `blender` — Virtual RGB-D cameras for rendering scenes in Blender.
  - `real2sim` — Generation scripts and utilities for converting reconstructed base trees into new synthetic variants.
  - `vls` — Virtual laser scanner integration and configs.

## Where to read more
- DATR implementation details and commands: [DATR/README.md](DATR/README.md).
- Real2Sim data generation details: [L-TreeGen/README.md](L-TreeGen/README.md).
