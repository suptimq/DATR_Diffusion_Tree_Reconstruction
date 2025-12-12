# Blender Rendering

`blender_script_datr.py` supports five modes that generate images and camera poses for 3D models. **zero123** produces uniformly sampled views around the object with RGB images and camera matrices. **plus** adds multi-view depth and EXR outputs from a fixed set of azimuth–elevation viewpoints. **amiga** renders images along a simulated robot trajectory with depth maps and repeated offsets. **lrm** follows the Large Reconstruction Model protocol, producing RGB, depth, normals, and full camera intrinsics and poses from random lighting and sampled camera locations. **single** renders dense azimuth sweeps for a single-view sequence, useful for tree reconstruction tasks.

### System requirements (GUI Required)

Follow the instructions in [Objaverse](https://github.com/allenai/objaverse-rendering?tab=readme-ov-file#installation) to setup the environments and resolve issues.


### Rendering

```bash
run.sh
```

This will then render the images into the `output_dir` directory.
