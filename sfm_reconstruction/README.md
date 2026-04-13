# 🧊 Multi-Image SfM 3D Reconstruction Pipeline

> **COLMAP → MiDaS Depth Estimation → ICP Registration → Tesla-Style Coloring → GLB Export**

A production-grade, end-to-end Structure-from-Motion (SfM) pipeline that turns a folder of ordinary photos into a fully colored, watertight 3D mesh — exported in `.GLB` format for immediate use in browsers, game engines, and AR/VR platforms.

This pipeline bridges two worlds that rarely talk cleanly to each other: **geometric reconstruction** (COLMAP's sparse SfM) and **learned monocular depth estimation** (Intel MiDaS), unified under a single, configurable Python script that handles everything from raw image ingestion to final web-ready export.

---

## 📐 Pipeline Architecture

```
Images (folder)
    │
    ▼
[Stage 1] COLMAP Feature Extraction (SIFT keypoints)
    │
    ▼
[Stage 2] COLMAP Feature Matching (exhaustive / sequential / vocab_tree)
    │
    ▼
[Stage 3] COLMAP SfM Mapper → Camera poses (R, t) + Sparse 3D point cloud
    │
    ▼
[Stage 4] MiDaS Depth Estimation per image (CPU-compatible)
    │       ↕ Scale anchored via COLMAP sparse points
    ▼
[Stage 5] Per-frame Dense Point Clouds (pinhole back-projection)
    │
    ▼
[Stage 6] ICP Registration (Point-to-Plane, sequential frame alignment)
    │
    ▼
[Stage 7] Merge + Statistical Outlier Removal (SOR cleaning)
    │
    ▼
[Stage 8] Tesla-Style Height / Normal / Distance Coloring (plasma colormap)
    │
    ▼
[Stage 9]  Poisson Surface Mesh Reconstruction
[Stage 9b] Tesla BEV Voxel Cube Mesh (occupancy grid style)
    │
    ▼
[Stage 10] Export → .GLB / .GLTF / .PLY / .OBJ
```

---

## 🧠 How the Pipeline Actually Works

### Stage 1–3: COLMAP Structure-from-Motion

COLMAP runs in three sub-stages:

1. **Feature Extraction**: SIFT keypoints are detected in every image. Resolution is controlled by `COLMAP_IMAGE_QUALITY` (`low` = 1000px, `medium` = 2000px, `high` = 3200px). GPU acceleration is optional (`COLMAP_USE_GPU`).

2. **Feature Matching**: Keypoints across images are matched.
   - `exhaustive` — every image pair compared. Best for < 100 images (objects, rooms).
   - `sequential` — consecutive pairs only. Best for video frames or ordered walkthroughs.
   - `vocab_tree` — approximate matching. Best for > 200 images.

3. **SfM Mapper**: COLMAP triangulates matched keypoints into 3D points and jointly refines camera intrinsics (focal length, principal point, distortion) and extrinsics (rotation `R`, translation `t`) via bundle adjustment. Output is a sparse point cloud + camera pose for each image.

The binary output is then converted to text format (cameras.txt / images.txt / points3D.txt) for parsing.

### Stage 4: MiDaS Depth Estimation

Each registered image is fed through Intel MiDaS to produce a per-pixel depth map. Three model options are available:

| Model | Speed (CPU) | Quality |
|---|---|---|
| `MiDaS_small` | ~5 s/image | Fast, good for tests |
| `DPT_Hybrid` | ~30 s/image | Better |
| `DPT_Large` | ~60 s/image | Best |

**Important**: MiDaS produces *relative* (inverse) depth — values are not in metric units. The pipeline corrects this via a **depth scale anchor**: COLMAP's sparse 3D points are projected into each camera's frame to compute their metric depths. The median metric depth is then used to rescale MiDaS's normalized output, giving an approximate metric-scale depth map per frame.

### Stage 5: Back-Projection to 3D

Each pixel in the scaled depth map is unprojected from camera space to world space using the pinhole camera model:

```
x_cam = (u - cx) × depth / fx
y_cam = (v - cy) × depth / fy
z_cam = depth

p_world = R^T × (p_cam − t)
```

Intrinsics are scaled to match the MiDaS output resolution. Depth values outside `[0.01, 50.0]` meters are discarded to eliminate background noise.

### Stage 6: ICP Registration

COLMAP camera poses are geometrically accurate, but MiDaS depth has small per-frame scale inconsistencies. Sequential **Point-to-Plane ICP** corrects residual alignment errors between adjacent frame clouds.

- Frame 0 is the fixed reference.
- Each frame is registered to the growing accumulated cloud.
- Point-to-Plane ICP (vs. Point-to-Point) uses surface normals for more accurate alignment.
- If a frame's ICP fitness score falls below 0.05, the original COLMAP pose is used (no ICP correction), preventing bad corrections from poisoning the reconstruction.

### Stage 7: Merge + Statistical Outlier Removal

All frames are merged and cleaned in two passes:

1. **Voxel downsampling**: Reduces redundant overlapping points (controlled by `VOXEL_SIZE`).
2. **Statistical Outlier Removal (SOR)**: For each point, the mean distance to its `SOR_NB_NEIGHBORS` nearest neighbors is computed. Points whose mean distance exceeds `global_mean + SOR_STD_RATIO × global_std` are removed as noise.

### Stage 8: Tesla-Style Coloring

Four coloring modes are available:

| Mode | Description | Best For |
|---|---|---|
| `height` | Z-axis height → plasma colormap | Autonomous vehicles, construction |
| `rgb` | Original photo colors | E-commerce, photorealistic output |
| `distance` | Distance from scene centroid | Radar/sonar visualization |
| `normals` | Surface normal directions (RGB encoding) | Architecture, wall/floor detection |

### Stage 9: Mesh Reconstruction

**Poisson Surface Reconstruction** fits a smooth implicit surface through the oriented point cloud. It produces a watertight (closed) mesh — essential for 3D printing, real-time physics, and CAD workflows. `POISSON_DEPTH` controls detail level (8 = fast/rough, 9 = balanced, 10–11 = high detail).

An optional **Tesla BEV Voxel Mesh** generates a grid of colored cubes over the scene — the same occupancy grid representation used in Tesla's FSD visualization.

---

## 🛠️ Installation

### Prerequisites

**1. COLMAP**

```bash
# Ubuntu/Debian
sudo apt install colmap

# macOS
brew install colmap

# Windows
# Download installer: https://colmap.github.io/install.html
```

**2. Python dependencies**

```bash
pip install open3d numpy matplotlib opencv-python torch torchvision trimesh pillow scipy
```

> **Note**: PyTorch is required only for MiDaS depth estimation. If not installed, the pipeline falls back to the COLMAP sparse cloud only.

---

## 🚀 Quickstart

### Local (Python Script)

```bash
# 1. Clone the repo
git clone https://github.com/Micahmichael03/sfm-3d-reconstruction.git
cd sfm-3d-reconstruction

# 2. Place your images in the images/ folder
# Minimum: 10 images | Recommended: 30–100 images
# Image overlap: 60–70% between adjacent shots

# 3. Edit Section 0 of the script (APPLICATION_MODE, paths)
#    nano script1_sfm_reconstruction.py

# 4. Run
python script1_sfm_reconstruction.py

# 5. View output
# Drag output/reconstruction_surface.glb to https://gltf-viewer.donmccurdy.com
```

### Google Colab (Notebook)

Open `notebook1_sfm_reconstruction.ipynb` in Google Colab:

1. Run **Cell 1** — Mount Google Drive
2. Run **Cell 2** — Install COLMAP + Python packages (~3 minutes first time)
3. **Edit Cell 3** — Set your image folder path and application mode
4. Run all remaining cells in order
5. Your `.glb` file downloads automatically at the end

---

## ⚙️ Configuration Reference

All configuration lives in **Section 0** of `script1_sfm_reconstruction.py`. You should not need to edit anything else.

### Application Modes

Set `APPLICATION_MODE` to auto-tune all pipeline parameters for your use case:

| Mode | Matcher | Voxel Size | Coloring | Use Case |
|---|---|---|---|---|
| `general` | exhaustive | 0.02 m | height/plasma | Any scene (default) |
| `architectural` | sequential | 0.03 m | normals/viridis | Building/room interior |
| `medical` | exhaustive | 0.001 m | height/inferno | CT/MRI stacks |
| `ecommerce` | exhaustive | 0.005 m | rgb | Product 360 photography |
| `autonomous` | sequential | 0.10 m | height/plasma | AV dashcam, LiDAR-style |
| `construction` | sequential | 0.08 m | height/turbo | Drone survey |

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `COLMAP_MATCHER` | `exhaustive` | Feature matching strategy |
| `COLMAP_USE_GPU` | `False` | Enable NVIDIA GPU for COLMAP |
| `DEPTH_MODEL` | `MiDaS_small` | MiDaS model size |
| `DEPTH_DEVICE` | `cpu` | `cpu` / `cuda` / `auto` |
| `MAX_IMAGES_FOR_DENSE` | `10` | Max frames for dense reconstruction |
| `VOXEL_SIZE` | `0.02` | Downsampling resolution (meters) |
| `SOR_NB_NEIGHBORS` | `20` | SOR neighborhood size |
| `SOR_STD_RATIO` | `2.0` | SOR outlier threshold |
| `ICP_MAX_CORRESPONDENCE_DISTANCE` | `0.05` | ICP max point matching distance |
| `COLOR_MODE` | `height` | Point cloud coloring mode |
| `COLOR_MAP` | `plasma` | Matplotlib colormap |
| `MESH_METHOD` | `poisson` | Surface reconstruction method |
| `POISSON_DEPTH` | `8` | Poisson octree depth |
| `GENERATE_VOXEL_MESH` | `True` | Also generate BEV voxel mesh |
| `EXPORT_GLB` | `True` | Export as .GLB (primary format) |

---

## 📁 Output Files

```
output/
├── reconstruction_pointcloud.ply        # Clean merged point cloud
├── reconstruction_surface.glb           # Poisson surface mesh (web-ready)
├── reconstruction_voxel_bev.glb         # Tesla BEV voxel mesh
├── colmap_sparse/                       # COLMAP reconstruction files
│   └── 0_txt/
│       ├── cameras.txt
│       ├── images.txt
│       └── points3D.txt
└── colmap_database.db                   # COLMAP feature database
```

### Viewing .GLB Files

| Platform | Method |
|---|---|
| Browser | [gltf-viewer.donmccurdy.com](https://gltf-viewer.donmccurdy.com) — drag & drop |
| Browser | [sandbox.babylonjs.com](https://sandbox.babylonjs.com) |
| Desktop | Blender, Windows 3D Viewer |
| Game Engine | Unity, Unreal Engine, Godot |
| AR/VR | Apple AR Quick Look, Google Scene Viewer |

---

## 📸 Image Capture Guidelines

Good reconstruction depends on good image capture. Follow these guidelines:

- **Overlap**: Each image should share 60–70% of its view with adjacent images.
- **Count**: Minimum 10 images; 30–100 recommended for quality.
- **Lighting**: Even, consistent lighting. Avoid harsh shadows or direct glare.
- **Motion blur**: Keep camera still at the moment of capture.
- **For rooms/architecture**: Walk in a smooth path, shoot every ~30°.
- **For objects**: Shoot in a circle at multiple elevations.
- **For video extraction**: Extract 1 frame per second from a smooth walkthrough video.

---

## 🔧 Troubleshooting

**COLMAP produced no reconstruction**
- Not enough image overlap (aim for 60–70%).
- Too few images (minimum ~10, recommended 30+).
- Images are blurry or over/under-exposed.
- Try `COLMAP_IMAGE_QUALITY = "high"` for more keypoints.

**ICP showing low fitness scores**
- Increase `ICP_MAX_CORRESPONDENCE_DISTANCE` (try 0.1 or 0.2).
- Ensure images are captured with sufficient overlap so consecutive frames share enough structure.

**Out of memory during depth estimation**
- Lower `DEPTH_MAX_OUTPUT_RESOLUTION` (e.g., 800).
- Reduce `MAX_IMAGES_FOR_DENSE` (start with 5–10).
- Switch to `DEPTH_MODEL = "MiDaS_small"`.

**GLB export fails**
- Install trimesh: `pip install trimesh`.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `colmap` | Structure-from-Motion — camera pose estimation |
| `open3d` | Point cloud processing, ICP, Poisson mesh |
| `torch` + `torchvision` | MiDaS depth estimation |
| `opencv-python` | Image loading and color conversion |
| `trimesh` | GLB / GLTF export |
| `scipy` | KD-tree for color transfer |
| `numpy` | Numerical operations |
| `matplotlib` | Colormaps |
| `pillow` | Image I/O |

---

## 👤 Author

**Michael Chukwuemeka Micah** — Computer Vision Engineer

- 🐙 GitHub: [Micahmichael03](https://github.com/Micahmichael03)
- 💼 LinkedIn: [michael-micah003](https://linkedin.com/in/michael-micah003)
- ✍️ Substack: [@michaelchukwuemekamicah](https://substack.com/@michaelchukwuemekamicah)
- 📧 Email: makoflash05@gmail.com

---

## 📄 License

MIT License — see `LICENSE` for details.