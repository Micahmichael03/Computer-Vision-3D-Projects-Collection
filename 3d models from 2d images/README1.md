# 🧠 2D-to-3D: Monocular Depth Estimation & Point Cloud Generation

![Figure 1](Figure%202026-03-29%20191737.png)
![Figure 2](ScreenCapture_2026-03-29-22-56-53.png)

> **Transform any single 2D image into a fully navigable 3D point cloud** — using a state-of-the-art transformer model and Open3D, with zero stereo cameras or LiDAR required.

---

## What This Project Does (And Why It Matters)

Most 3D reconstruction pipelines require expensive hardware — stereo camera rigs, depth sensors, or LiDAR arrays. This project breaks that assumption.

Using **GLPN (Global-Local Path Networks)** — a transformer-based monocular depth estimation model pre-trained on the NYU Depth V2 dataset — this pipeline extracts per-pixel depth information from a **single RGB image**, then projects it into a **colored 3D point cloud** that you can explore, export, and integrate into downstream applications.

**Real-world applications include:**

- 🏥 **Healthcare**: Surgical planning mockups, patient anatomy previsualization, and assistive tech for visually impaired navigation systems
- 🚗 **Autonomous Vehicles**: Cheap depth perception fallback for sensor-fusion pipelines
- 🏗️ **Architecture & Heritage Preservation**: Digitize physical spaces from photographs alone
- 🎮 **AR/VR & Gaming**: Generate 3D assets from flat concept art or real-world photos
- 🤖 **Robotics**: Scene understanding from monocular cameras in cost-constrained deployments

---

## Pipeline Overview

```
[Input Image]
     ↓
[Smart Resize] → Ensures dimensions are multiples of 32 (GLPN model requirement)
     ↓
[GLPN Feature Extractor] → Tokenizes the image into model-ready tensors
     ↓
[GLPN Depth Model] → Transformer inference (no gradient computation)
     ↓
[Depth Map] → Per-pixel depth in millimeters, squeezed to 2D NumPy array
     ↓
[Padding Crop] → Removes 16px border artifacts from model edges
     ↓
[RGBD Image] → Color + Depth fused into Open3D's RGBD structure
     ↓
[Pinhole Camera Model] → Projects RGBD pixels into 3D space
     ↓
[Point Cloud] → 3D colored point cloud (flipped upright)
     ↓
[Statistical Outlier Removal] → Cleans noise using neighbor-based filtering
     ↓
[Export → .ply] → Standard 3D format compatible with Blender, MeshLab, CloudCompare
```

---

## Requirements

### Python Version
Python 3.8 or higher is recommended.

### Hardware
- A CPU is sufficient for inference on single images
- A CUDA-compatible GPU will significantly accelerate processing for batch workloads

### Install Dependencies

```bash
pip install torch torchvision
pip install transformers
pip install Pillow
pip install matplotlib
pip install numpy
pip install open3d
```

> **Note on Open3D**: Open3D may not install cleanly on all platforms via pip alone. If you encounter issues, refer to the [official Open3D installation guide](http://www.open3d.org/docs/release/getting_started.html).

### HuggingFace Model (Auto-Downloaded)

The pipeline uses `vinvino02/glpn-nyu` from HuggingFace. It will be downloaded automatically on first run (~80MB). No manual model download required.

---

## Usage

### Step 1 — Set Your Input Image

In the script, update this line to point to your image:

```python
image = Image.open(r"path/to/your/image.jpeg")
```

Any standard image format (JPEG, PNG, BMP) is supported. The script handles resizing automatically.

### Step 2 — Run the Script

```bash
python 3d_modes-2d-images.py
```

The script will:
1. Display a side-by-side matplotlib figure showing the original image and its predicted depth map (rendered with `plasma` colormap)
2. Open an interactive Open3D viewer showing the 3D point cloud
3. Export a `.ply` file to your working directory

### Step 3 — Customize the Export Filename

```python
file_name = "Your_Output_Name.ply"
```

---

## Configuration & Key Parameters

| Parameter | Location | Default | Purpose |
|---|---|---|---|
| `depth_trunc` | `RGBDImage.create_from_color_and_depth()` | `5.0` | Truncates depth at 5 meters — filters out background noise |
| `depth_scale` | Same as above | `1.0` | Scale factor for depth values (already in meters after `/1000`) |
| `nb_neighbors` | `remove_statistical_outlier()` | `20` | Number of neighboring points examined per point for noise analysis |
| `std_ratio` | Same as above | `2.0` | Points beyond 2 standard deviations from mean distance are removed |
| Focal length | `set_intrinsics()` | `500.0` | Approximated pinhole focal length — adjust for your camera if known |
| Padding | `pad = 16` | `16` | Border pixels removed post-inference to eliminate model edge artifacts |

---

## Output

The script produces two outputs:

**1. Matplotlib Visualization** — Side-by-side comparison of the original image and its depth map:

```
| Original Image  |  Predicted Depth (plasma colormap) |
```

Bright/yellow = closer to the camera. Dark/purple = farther away.

**2. `.ply` Point Cloud File** — Importable into:
- Blender (File → Import → Stanford PLY)
- MeshLab
- CloudCompare
- ParaView

---

## Project Structure

```
📁 project-root/
├── 3d_modes-2d-images.py     # Main pipeline script
├── README.md                 # You are here
└── *.ply                     # Generated point cloud outputs
```

---

## How the Smart Resize Works

The GLPN model requires input dimensions to be **multiples of 32**. Raw images rarely satisfy this. The resize logic:

1. Caps height at 480px (or keeps original if smaller)
2. Snaps height down to nearest multiple of 32
3. Calculates width proportionally from the original aspect ratio
4. Snaps width to nearest multiple of 32, rounding to whichever multiple is closer

This preserves aspect ratio while guaranteeing valid model input — a subtle but important detail that prevents silent shape mismatches during inference.

---

## Known Limitations

- **Monocular depth estimation is inherently approximate.** The model infers depth from visual cues (perspective, shading, texture gradients), not physical measurement.
- **Textureless regions** (blank walls, clear skies) produce less reliable depth estimates.
- **The pinhole camera intrinsics are approximated** (focal length = 500, principal point = image center). For metrology-grade accuracy, calibrate your actual camera and substitute real intrinsic values.
- **Outdoor scenes with depth > 5m** will be truncated. Increase `depth_trunc` for wide outdoor shots.

---

## Author

**Michael Chukwuemeka Micah**
📧 makoflash05@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/michael-micah003)
✍️ [Substack](https://substack.com/@michaelchukwuemekamicah)
💻 [GitHub](https://github.com/Micahmichael03)

---

## License

MIT License — feel free to build on this, cite it, or adapt it for your domain.