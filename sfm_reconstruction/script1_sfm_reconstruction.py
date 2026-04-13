"""
================================================================================
 SCRIPT 1: Multi-Image SfM 3D Reconstruction Pipeline
 Written by: Michael Micah — Professional Computer Vision Pipeline
================================================================================

 PIPELINE OVERVIEW
 -----------------
  Stage 1  →  Load images from folder
  Stage 2  →  COLMAP Structure-from-Motion (SfM) → camera poses + sparse cloud
  Stage 3  →  MiDaS depth estimation per image (CPU-compatible)
  Stage 4  →  Generate per-frame dense point clouds using poses + depth 
  Stage 5  →  ICP Registration to refine frame-to-frame alignment
  Stage 6  →  Merge + Statistical Outlier Removal (SOR) cleaning
  Stage 7  →  Tesla-style thermal/height coloring (plasma colormap)
  Stage 8  →  Poisson surface mesh reconstruction
  Stage 9  →  Tesla BEV-style voxel cube mesh
  Stage 10 →  Export as .GLB / .GLTF for web & real-time rendering

 COMPATIBLE APPLICATIONS
 ------------------------
   "architectural"  — Building/room interior scan
   "medical"        — CT/MRI scan data (already-prepared image stacks)
   "ecommerce"      — Product photography 360 capture
   "autonomous"     — Camera-based AV perception (LiDAR-style output)
   "construction"   — Site survey drone / walk-through capture
   "general"        — Any scene (default balanced settings)

 HOW TO RUN
 ----------
  1. Install COLMAP:
       Windows → https://colmap.github.io/install.html  (download installer)
       Linux   → sudo apt install colmap
       macOS   → brew install colmap

  2. Install Python dependencies:
       pip install open3d numpy matplotlib opencv-python torch torchvision
       pip install trimesh pillow scipy

  3. Put your photos in the IMAGE_FOLDER (set below in Section 0)
     - Minimum: 10 images, recommended: 30–100 images
     - Each image should overlap 60–70% with adjacent images
     - Good lighting, avoid motion blur

  4. Run:
       python script1_sfm_reconstruction.py

  5. View the .GLB output:
       Drag & drop to: https://gltf-viewer.donmccurdy.com/
       Or open in: Blender, Windows 3D Viewer, Unity, Unreal Engine

================================================================================
"""

# ============================================================================ #
# SECTION 0 — CONFIGURATION (EDIT ONLY THIS SECTION)                         #
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# 0.1  Application Mode                                                        #
# ---------------------------------------------------------------------------- #
# Choose the application context. Automatically tunes processing parameters.
# Options: "general" | "architectural" | "medical" | "ecommerce" |
#           "autonomous" | "construction"
APPLICATION_MODE = "general"

# ---------------------------------------------------------------------------- #
# 0.2  Paths                                                                   #
# ---------------------------------------------------------------------------- #
# Folder containing your input images (.jpg / .jpeg / .png)
IMAGE_FOLDER = "images/"

# All output files (point clouds, meshes, GLB) will be saved here
OUTPUT_FOLDER = "output/"

# ---------------------------------------------------------------------------- #
# 0.3  COLMAP Settings                                                         #
# ---------------------------------------------------------------------------- #
# Full path to COLMAP executable.
# Windows example: "C:/COLMAP/COLMAP.bat"
# Linux / macOS: "colmap"  (if installed via apt/brew and in PATH)
COLMAP_EXECUTABLE = "colmap"

# Feature matcher type:
#   "exhaustive" → compares EVERY image pair. Best for <100 images (objects, rooms)
#   "sequential" → compares consecutive images. Best for video frames / ordered walkthroughs
#   "vocab_tree" → fast approximate matching. Best for >200 images
COLMAP_MATCHER = "exhaustive"

# Set True ONLY if you have an NVIDIA GPU. False = CPU mode (slower but always works)
COLMAP_USE_GPU = False

# Feature extraction quality. Higher = better matches but slower.
# "low" | "medium" | "high"
COLMAP_IMAGE_QUALITY = "high"

# ---------------------------------------------------------------------------- #
# 0.4  Depth Estimation Model (MiDaS — all run on CPU)                        #
# ---------------------------------------------------------------------------- #
# "MiDaS_small"  → Fastest. ~5s/image on CPU. Good for quick tests.  ← START HERE
# "DPT_Hybrid"   → Better quality. ~30s/image on CPU.
# "DPT_Large"    → Best quality. ~60s/image on CPU.
# Note: First run downloads the model weights from the internet (~200–400 MB)
DEPTH_MODEL = "MiDaS_small"

# Device for depth estimation:
#   "cpu"  → force CPU mode (recommended for Colab stability)
#   "cuda" → force GPU mode (requires NVIDIA GPU)
#   "auto" → pick CUDA if available, else CPU
DEPTH_DEVICE = "cpu"

# Cap MiDaS depth map output resolution to reduce memory spikes.
# If your source images are huge, depth is computed at a safely downscaled size.
# Set to None to keep full image resolution.
DEPTH_MAX_OUTPUT_RESOLUTION = 1600

# Maximum number of images to use for dense reconstruction.
# Set to None to process ALL images (slow on CPU for large sets).
# Recommended: 10–20 for first run. Increase when satisfied with results.
MAX_IMAGES_FOR_DENSE = 10

# ---------------------------------------------------------------------------- #
# 0.5  Point Cloud Settings                                                    #
# ---------------------------------------------------------------------------- #
# Voxel size for downsampling (in scene units, roughly meters for real scenes).
# Smaller = more detail but slower. Larger = faster but coarser.
# Architectural: 0.02–0.05 | E-commerce product: 0.005 | Construction: 0.1
VOXEL_SIZE = 0.02

# Statistical Outlier Removal (SOR) — removes noise points
SOR_NB_NEIGHBORS = 20    # Analyze each point against this many neighbors
SOR_STD_RATIO    = 2.0   # Remove points > N standard deviations from mean distance

# ---------------------------------------------------------------------------- #
# 0.6  ICP Registration Settings                                               #
# ---------------------------------------------------------------------------- #
# ICP aligns per-frame point clouds to correct small depth-scale mismatches.
# ICP_MAX_CORRESPONDENCE_DISTANCE: max allowed gap between matched points
# (in scene units). Increase if frames are badly misaligned.
ICP_MAX_CORRESPONDENCE_DISTANCE = 0.05
ICP_MAX_ITERATIONS = 50

# ---------------------------------------------------------------------------- #
# 0.7  Tesla-Style Coloring                                                    #
# ---------------------------------------------------------------------------- #
# "height"    → Color by Z-axis height using plasma colormap (Tesla BEV style)
# "rgb"       → Use original photo colors
# "distance"  → Color by distance from scene center (radar / sonar style)
# "normals"   → Color by surface normal direction (reveals planes/surfaces)
COLOR_MODE = "height"

# Matplotlib colormap. "plasma" = Tesla/thermal. Others: "turbo", "viridis", "jet"
COLOR_MAP = "plasma"

# Clip extreme height values to improve color spread (percentile 0–100)
HEIGHT_CLIP_LOW  = 5   # Ignore bottom 5% heights (floor noise)
HEIGHT_CLIP_HIGH = 95  # Ignore top 5% heights (ceiling/sky noise)

# ---------------------------------------------------------------------------- #
# 0.8  Mesh Reconstruction                                                     #
# ---------------------------------------------------------------------------- #
# Mesh method: "poisson" (smooth, watertight) | "ball_pivot" (keeps detail)
MESH_METHOD = "poisson"

# Poisson octree depth. Higher = more detail but slower and more memory.
# 8 = fast/rough, 9 = balanced (recommended), 10–11 = fine (slow)
POISSON_DEPTH = 8

# Remove low-density mesh regions (artifacts in empty space).
# 0.0 = keep all. 0.01 = remove bottom 1% density (recommended). 0.05 = aggressive.
MESH_DENSITY_QUANTILE = 0.01

# ---------------------------------------------------------------------------- #
# 0.9  Voxel Mesh (Tesla BEV Occupancy Style)                                 #
# ---------------------------------------------------------------------------- #
# Whether to generate a voxel-cube mesh in addition to the Poisson mesh
GENERATE_VOXEL_MESH = True

# Size of each cube in the voxel mesh (scene units)
VOXEL_MESH_CUBE_SIZE = 0.04

# ---------------------------------------------------------------------------- #
# 0.10  Export Settings                                                        #
# ---------------------------------------------------------------------------- #
EXPORT_PLY  = True    # Point cloud as .PLY
EXPORT_GLB  = True    # Mesh as .GLB  (web, real-time) ← PRIMARY FORMAT
EXPORT_OBJ  = False   # Mesh as .OBJ  (Blender, Maya)
EXPORT_GLTF = False   # Mesh as .GLTF (JSON version of GLB)

# Base name used for all output files
OUTPUT_NAME = "reconstruction"

# ---------------------------------------------------------------------------- #
# 0.11  Visualization                                                          #
# ---------------------------------------------------------------------------- #
# Set False on headless servers (no display) or when running in batch mode
VISUALIZE_INTERMEDIATE = True   # Show point cloud at intermediate steps
VISUALIZE_FINAL        = True   # Show final mesh result

# ============================================================================ #
# END OF CONFIGURATION — No need to edit below this line unless you are       #
# customizing the pipeline.                                                    #
# ============================================================================ #


# ============================================================================ #
# SECTION 1 — IMPORTS                                                         #
# ============================================================================ #

import os
import sys
import glob
import copy
import time
import subprocess
import warnings

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from PIL import Image

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not found. Depth estimation disabled. Install with: pip install torch")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("[WARN] trimesh not found. GLB export disabled. Install with: pip install trimesh")

warnings.filterwarnings("ignore")


# ============================================================================ #
# SECTION 2 — APPLICATION PRESETS                                             #
# ============================================================================ #

# Each preset overrides the defaults above for the specific domain.
# These are research-informed settings for each application context.
APPLICATION_PRESETS = {
    "general": {
        "description"     : "Balanced settings for any scene",
        # No overrides — use user config as-is
    },
    "architectural": {
        "description"     : "Building/room interior scanning",
        "COLMAP_MATCHER"  : "sequential",   # Assume video walkthrough
        "VOXEL_SIZE"      : 0.03,
        "POISSON_DEPTH"   : 10,             # High detail for blueprints
        "COLOR_MODE"      : "normals",      # Reveals wall/floor/ceiling planes
        "COLOR_MAP"       : "viridis",
        "SOR_STD_RATIO"   : 2.0,
    },
    "medical": {
        "description"     : "Medical imaging (CT/MRI stacks)",
        "COLMAP_MATCHER"  : "exhaustive",
        "VOXEL_SIZE"      : 0.001,          # Very fine — medical data is high-res
        "POISSON_DEPTH"   : 10,
        "COLOR_MODE"      : "height",
        "COLOR_MAP"       : "inferno",      # Traditional medical colormap
        "SOR_STD_RATIO"   : 1.5,            # Tighter — less noise tolerance
        "MESH_DENSITY_QUANTILE": 0.001,
    },
    "ecommerce": {
        "description"     : "Product photography 360",
        "COLMAP_MATCHER"  : "exhaustive",
        "VOXEL_SIZE"      : 0.005,          # Fine detail
        "POISSON_DEPTH"   : 10,
        "COLOR_MODE"      : "rgb",          # Product colors matter
        "SOR_STD_RATIO"   : 2.0,
    },
    "autonomous": {
        "description"     : "Autonomous vehicle camera pipeline",
        "COLMAP_MATCHER"  : "sequential",   # Dashcam / forward-facing
        "VOXEL_SIZE"      : 0.1,            # Coarser — AV needs speed
        "POISSON_DEPTH"   : 8,
        "COLOR_MODE"      : "height",       # BEV height coloring
        "COLOR_MAP"       : "plasma",       # Tesla style
        "GENERATE_VOXEL_MESH": True,
        "VOXEL_MESH_CUBE_SIZE": 0.15,
        "SOR_STD_RATIO"   : 3.0,            # More lenient — fast scenes are noisier
    },
    "construction": {
        "description"     : "Construction site survey",
        "COLMAP_MATCHER"  : "sequential",   # Drone/walk survey = ordered
        "VOXEL_SIZE"      : 0.08,
        "POISSON_DEPTH"   : 8,
        "COLOR_MODE"      : "height",       # Topographic height map
        "COLOR_MAP"       : "turbo",
        "SOR_STD_RATIO"   : 2.5,
        "VOXEL_MESH_CUBE_SIZE": 0.1,
    },
}


def apply_preset(mode):
    """
    Apply the selected application preset by overriding the global config.
    This is called at pipeline startup.
    """
    global COLMAP_MATCHER, VOXEL_SIZE, POISSON_DEPTH, COLOR_MODE, COLOR_MAP
    global SOR_STD_RATIO, MESH_DENSITY_QUANTILE, GENERATE_VOXEL_MESH, VOXEL_MESH_CUBE_SIZE

    preset = APPLICATION_PRESETS.get(mode, APPLICATION_PRESETS["general"])

    print(f"\n[PRESET] Application mode: '{mode}' — {preset['description']}")

    # Map preset keys to globals
    overrides = {
        "COLMAP_MATCHER"      : "COLMAP_MATCHER",
        "VOXEL_SIZE"          : "VOXEL_SIZE",
        "POISSON_DEPTH"       : "POISSON_DEPTH",
        "COLOR_MODE"          : "COLOR_MODE",
        "COLOR_MAP"           : "COLOR_MAP",
        "SOR_STD_RATIO"       : "SOR_STD_RATIO",
        "MESH_DENSITY_QUANTILE": "MESH_DENSITY_QUANTILE",
        "GENERATE_VOXEL_MESH" : "GENERATE_VOXEL_MESH",
        "VOXEL_MESH_CUBE_SIZE": "VOXEL_MESH_CUBE_SIZE",
    }

    for key, global_name in overrides.items():
        if key in preset:
            globals()[global_name] = preset[key]
            print(f"           {key:30s} = {preset[key]}")


# ============================================================================ #
# SECTION 3 — UTILITY / HELPER FUNCTIONS                                      #
# ============================================================================ #

def setup_directories():
    """Create output directories."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "colmap_sparse"), exist_ok=True)
    print(f"[SETUP] Output folder: {os.path.abspath(OUTPUT_FOLDER)}")


def load_image_paths(folder, extensions=("*.jpg", "*.jpeg", "*.png", "*.PNG")):
    """
    Scan IMAGE_FOLDER and return sorted list of image file paths.
    """
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    paths = sorted(paths)

    if not paths:
        raise FileNotFoundError(
            f"No images found in '{folder}'. "
            f"Please add .jpg/.jpeg/.png images to that folder."
        )

    print(f"[LOAD]  Found {len(paths)} images in '{folder}'")
    return paths


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """
    Convert a unit quaternion (COLMAP format: qw, qx, qy, qz) to a 3x3 rotation matrix.

    COLMAP stores camera-to-world rotation as a quaternion.
    The resulting R satisfies: p_cam = R @ p_world + t
    """
    # Normalize to unit quaternion
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)],
    ])
    return R


def safe_visualize(geometries, window_name="Open3D", width=1280, height=720):
    """
    Open3D viewer with graceful fallback for headless environments.
    Controlled by VISUALIZE_INTERMEDIATE / VISUALIZE_FINAL flags.
    """
    if not (VISUALIZE_INTERMEDIATE or VISUALIZE_FINAL):
        return
    try:
        o3d.visualization.draw_geometries(
            geometries, window_name=window_name,
            width=width, height=height
        )
    except Exception as e:
        print(f"[VIZ]  Could not open viewer ({e}).")
        print("[VIZ]  Set VISUALIZE_INTERMEDIATE = False for headless environments.")


def print_section(title):
    """Print a clearly visible section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ============================================================================ #
# SECTION 4 — COLMAP STRUCTURE-FROM-MOTION                                   #
# ============================================================================ #

def check_colmap(executable):
    """Check COLMAP is installed and return True/False."""
    try:
        r = subprocess.run(
            [executable, "--help"],
            capture_output=True, text=True, timeout=15
        )
        # COLMAP returns non-zero for --help but prints its name
        return "COLMAP" in (r.stdout + r.stderr)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_colmap_sfm(image_folder, output_folder, colmap_exe,
                   matcher, use_gpu, image_quality):
    """
    Run full COLMAP SfM pipeline:
      1. Feature extraction (SIFT keypoints)
      2. Feature matching (exhaustive / sequential)
      3. Sparse reconstruction (Structure-from-Motion)
      4. Convert output to text format for parsing

    Returns: path to the text-format reconstruction folder
    """
    db_path     = os.path.join(output_folder, "colmap_database.db")
    sparse_path = os.path.join(output_folder, "colmap_sparse")
    os.makedirs(sparse_path, exist_ok=True)

    quality_to_size = {"low": "1000", "medium": "2000", "high": "3200"}
    max_image_size = quality_to_size.get(image_quality, "2000")

    gpu_flag = "1" if use_gpu else "0"

    # ── Step 1: Feature extraction ────────────────────────────────────────────
    print_section("COLMAP  Stage 1/4 — Feature Extraction (SIFT)")
    print(f"  Image folder : {image_folder}")
    print(f"  Quality      : {image_quality} (max size {max_image_size}px)")
    print(f"  GPU          : {'yes' if use_gpu else 'no — CPU mode'}")

    cmd = [
        colmap_exe, "feature_extractor",
        "--database_path",              db_path,
        "--image_path",                 image_folder,
        "--ImageReader.single_camera",  "0",          # Each image may have its own camera
        "--SiftExtraction.use_gpu",     gpu_flag,
        "--SiftExtraction.max_image_size", max_image_size,
    ]
    _run_colmap_cmd(cmd, "feature_extractor")

    # ── Step 2: Feature matching ──────────────────────────────────────────────
    print_section(f"COLMAP  Stage 2/4 — Feature Matching ({matcher})")
    cmd = [
        colmap_exe, f"{matcher}_matcher",
        "--database_path",          db_path,
        "--SiftMatching.use_gpu",   gpu_flag,
    ]
    _run_colmap_cmd(cmd, f"{matcher}_matcher")

    # ── Step 3: Sparse reconstruction (SfM mapper) ───────────────────────────
    print_section("COLMAP  Stage 3/4 — SfM Mapper (this is the slowest step)")
    print("  Building 3D point cloud from matched features...")
    cmd = [
        colmap_exe, "mapper",
        "--database_path",              db_path,
        "--image_path",                 image_folder,
        "--output_path",                sparse_path,
        "--Mapper.min_num_matches",     "15",
        "--Mapper.init_min_num_inliers","30",
    ]
    _run_colmap_cmd(cmd, "mapper")

    # ── Step 4: Convert binary → text ─────────────────────────────────────────
    # COLMAP creates numbered subfolders: sparse/0, sparse/1, ...
    # We pick the largest reconstruction (usually folder 0).
    recon_dirs = sorted([
        d for d in os.listdir(sparse_path)
        if os.path.isdir(os.path.join(sparse_path, d))
    ])

    if not recon_dirs:
        raise RuntimeError(
            "COLMAP produced no reconstruction.\n"
            "  Common causes:\n"
            "  - Not enough image overlap (aim for 60-70%)\n"
            "  - Too few images (minimum ~10, recommended 30+)\n"
            "  - Images are blurry or under/over-exposed\n"
        )

    best_recon_bin = os.path.join(sparse_path, recon_dirs[0])
    best_recon_txt = best_recon_bin + "_txt"
    os.makedirs(best_recon_txt, exist_ok=True)

    print_section("COLMAP  Stage 4/4 — Converting Output to Text Format")
    cmd = [
        colmap_exe, "model_converter",
        "--input_path",  best_recon_bin,
        "--output_path", best_recon_txt,
        "--output_type", "TXT",
    ]
    _run_colmap_cmd(cmd, "model_converter")

    print(f"\n[COLMAP] SfM complete!")
    print(f"  Found {len(recon_dirs)} reconstruction(s). Using: {best_recon_txt}")
    return best_recon_txt


def _run_colmap_cmd(cmd, step_name):
    """Execute a COLMAP subprocess command with error checking."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and "error" in result.stderr.lower():
        raise RuntimeError(
            f"COLMAP '{step_name}' failed.\n"
            f"  STDERR:\n{result.stderr[-2000:]}\n"
            f"  Make sure COLMAP is installed and COLMAP_EXECUTABLE is correct."
        )
    # Print COLMAP's own progress messages
    for line in result.stdout.splitlines():
        if any(kw in line for kw in ["images", "point", "registered", "ERROR"]):
            print(f"  [colmap] {line}")


# ============================================================================ #
# SECTION 5 — COLMAP OUTPUT PARSERS                                          #
# ============================================================================ #

def parse_colmap_reconstruction(txt_folder):
    """
    Parse all three COLMAP text files and return a unified data structure.

    Returns dict with keys:
      'cameras'  → {camera_id: {fx, fy, cx, cy, width, height, dist_coeffs}}
      'images'   → {image_name: {camera_id, R, t, points2d}}
      'points3d' → np.array (N, 3) + colors np.array (N, 3) float [0,1]
    """
    cameras_file  = os.path.join(txt_folder, "cameras.txt")
    images_file   = os.path.join(txt_folder, "images.txt")
    points3d_file = os.path.join(txt_folder, "points3D.txt")

    cameras  = _parse_cameras(cameras_file)
    images   = _parse_images(images_file)
    pts, col = _parse_points3d(points3d_file)

    print(f"[PARSE] Cameras  : {len(cameras)}")
    print(f"[PARSE] Images   : {len(images)}")
    print(f"[PARSE] 3D points: {len(pts)}")

    return {"cameras": cameras, "images": images,
            "points3d_xyz": pts, "points3d_rgb": col}


def _parse_cameras(filepath):
    """
    Parse cameras.txt.
    Supports SIMPLE_PINHOLE, PINHOLE, RADIAL, SIMPLE_RADIAL, OPENCV models.

    Camera intrinsics format (COLMAP):
      PINHOLE        : fx  fy  cx  cy
      SIMPLE_PINHOLE : f   cx  cy
      RADIAL         : f   cx  cy  k1  k2
    """
    cameras = {}
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p     = line.strip().split()
            cam_id = int(p[0])
            model  = p[1]
            w, h   = int(p[2]), int(p[3])
            params = [float(x) for x in p[4:]]

            cam = {"model": model, "width": w, "height": h,
                   "dist_coeffs": np.zeros(5)}

            if model == "PINHOLE":
                cam["fx"], cam["fy"] = params[0], params[1]
                cam["cx"], cam["cy"] = params[2], params[3]
            elif model == "SIMPLE_PINHOLE":
                cam["fx"] = cam["fy"] = params[0]
                cam["cx"], cam["cy"] = params[1], params[2]
            elif model in ("RADIAL", "SIMPLE_RADIAL"):
                cam["fx"] = cam["fy"] = params[0]
                cam["cx"], cam["cy"] = params[1], params[2]
                k1 = params[3] if len(params) > 3 else 0.0
                k2 = params[4] if len(params) > 4 else 0.0
                cam["dist_coeffs"] = np.array([k1, k2, 0, 0, 0])
            elif model == "OPENCV":
                cam["fx"], cam["fy"] = params[0], params[1]
                cam["cx"], cam["cy"] = params[2], params[3]
                cam["dist_coeffs"]   = np.array([params[4], params[5],
                                                  params[6], params[7], 0])
            else:
                # Generic fallback: assume fx=fy=f, principal point at center
                cam["fx"] = cam["fy"] = params[0] if params else float(w)
                cam["cx"] = params[2] if len(params) > 2 else w / 2.0
                cam["cy"] = params[3] if len(params) > 3 else h / 2.0

            cameras[cam_id] = cam

    return cameras


def _parse_images(filepath):
    """
    Parse images.txt.

    Each image uses TWO lines:
      Line 1: IMAGE_ID  QW QX QY QZ  TX TY TZ  CAMERA_ID  NAME
      Line 2: POINTS2D as (X Y POINT3D_ID) triplets

    The quaternion (QW, QX, QY, QZ) and translation (TX, TY, TZ) define the
    world-to-camera transform: p_cam = R @ p_world + t
    """
    images = {}
    with open(filepath, "r") as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]

    i = 0
    while i < len(lines) - 1:
        p = lines[i].strip().split()
        if len(p) < 10:
            i += 1
            continue

        qw, qx, qy, qz = float(p[1]), float(p[2]), float(p[3]), float(p[4])
        tx, ty, tz      = float(p[5]), float(p[6]), float(p[7])
        cam_id          = int(p[8])
        img_name        = p[9]

        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])

        # Parse 2D keypoints line
        points2d = []
        kp_parts = lines[i + 1].strip().split()
        for j in range(0, len(kp_parts) - 2, 3):
            px, py  = float(kp_parts[j]), float(kp_parts[j + 1])
            pt3d_id = int(kp_parts[j + 2])
            points2d.append({"xy": np.array([px, py]), "point3d_id": pt3d_id})

        images[img_name] = {
            "camera_id" : cam_id,
            "R"         : R,         # 3x3  world-to-camera rotation
            "t"         : t,         # (3,) world-to-camera translation
            "points2d"  : points2d,
        }
        i += 2

    return images


def _parse_points3d(filepath):
    """
    Parse points3D.txt.
    Returns np.arrays: points (N,3) float64, colors (N,3) float32 [0,1]
    """
    pts, cols = [], []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.strip().split()
            if len(p) < 8:
                continue
            # Format: POINT3D_ID  X Y Z  R G B  ERROR  TRACK[]
            pts.append([float(p[1]), float(p[2]), float(p[3])])
            cols.append([int(p[4]) / 255.0, int(p[5]) / 255.0, int(p[6]) / 255.0])

    return np.array(pts, dtype=np.float64), np.array(cols, dtype=np.float32)


# ============================================================================ #
# SECTION 6 — DEPTH ESTIMATION (MiDaS — CPU Compatible)                     #
# ============================================================================ #

def load_midas(model_type, preferred_device="cpu"):
    """
    Load MiDaS depth estimation model via PyTorch Hub.

    On first call this downloads model weights (~100–400 MB) from the internet.
    Subsequent calls use the local cache (~/.cache/torch/hub).

    Returns: (model, transform_fn, device)
    """
    if not TORCH_AVAILABLE:
        return None, None, None

    print(f"\n[DEPTH] Loading MiDaS model: {model_type}")
    preferred = str(preferred_device).strip().lower()
    if preferred not in ("cpu", "cuda", "auto"):
        print(f"[DEPTH] Unknown DEPTH_DEVICE='{preferred_device}'. Falling back to 'cpu'.")
        preferred = "cpu"

    if preferred == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    elif preferred == "cuda":
        if torch.cuda.is_available():
            resolved = "cuda"
        else:
            print("[DEPTH] DEPTH_DEVICE='cuda' requested but CUDA is unavailable. Using CPU.")
            resolved = "cpu"
    else:
        resolved = "cpu"

    device = torch.device(resolved)
    print(f"[DEPTH] Device: {device}")

    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    midas.to(device).eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    if model_type in ("DPT_Large", "DPT_Hybrid"):
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform   # MiDaS_small, MiDaS

    print(f"[DEPTH] Model ready.")
    return midas, transform, device


def estimate_depth(image_bgr, midas_model, midas_transform, device, max_output_resolution=None):
    """
    Run MiDaS on a single OpenCV BGR image.

    MiDaS outputs inverse depth (disparity):
      - High value → close to camera
      - Low  value → far from camera

    We invert to get depth (high value = far) and normalize to [0, 1].

    Returns: depth_map (H, W) float32 normalized [0, 1]
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image_rgb.shape[:2]
    out_h, out_w = h_orig, w_orig
    if isinstance(max_output_resolution, int) and max_output_resolution > 0:
        max_side = max(h_orig, w_orig)
        if max_side > max_output_resolution:
            scale = float(max_output_resolution) / float(max_side)
            out_w = max(1, int(round(w_orig * scale)))
            out_h = max(1, int(round(h_orig * scale)))

    # MiDaS pre-processing
    input_batch = midas_transform(image_rgb).to(device)

    with torch.no_grad():
        pred = midas_model(input_batch)

    # Resize output back to target resolution (optionally downscaled for memory safety)
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=(out_h, out_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth = pred.cpu().numpy().astype(np.float32)

    # MiDaS gives disparity (closer = higher number).
    # Invert to get depth (farther = higher number).
    # Add small epsilon to avoid division by zero.
    depth = 1.0 / (depth + 1e-8)

    # Normalize to [0, 1]
    d_min, d_max = depth.min(), depth.max()
    depth = (depth - d_min) / (d_max - d_min + 1e-8)

    return depth


def compute_depth_scale(camera_R, camera_t, sparse_pts_world):
    """
    Estimate the scale factor to convert relative MiDaS depth to approximate
    metric depth.

    Strategy:
      - COLMAP's sparse point cloud is already in metric scale (relative scale
        consistent across the scene).
      - We project the sparse points into camera space to get their metric depths.
      - MiDaS depth is normalized [0, 1], with median ≈ 0.5.
      - Scale = (median metric depth) / 0.5

    This is an approximation. For production use (autonomous vehicles, surveying)
    use a proper scale calibration with known distance references.

    Args:
        camera_R        : (3,3) world-to-camera rotation from COLMAP
        camera_t        : (3,)  world-to-camera translation from COLMAP
        sparse_pts_world: (N,3) COLMAP sparse 3D points in world coordinates

    Returns: float scale factor
    """
    if len(sparse_pts_world) == 0:
        return 1.0

    # Project sparse points into camera frame
    # p_cam = R @ p_world + t   (COLMAP world-to-camera convention)
    pts_cam = sparse_pts_world @ camera_R.T + camera_t  # shape (N, 3)

    # Keep only points in front of the camera (positive Z)
    in_front = pts_cam[:, 2] > 0.01
    if in_front.sum() < 3:
        return 1.0

    # Median of visible metric depths (robust to outliers)
    median_metric_depth = np.median(pts_cam[in_front, 2])

    # MiDaS normalized depth has median ≈ 0.5 (by construction)
    scale = median_metric_depth / 0.5

    return float(np.clip(scale, 0.01, 1000.0))  # Safety clamp


# ============================================================================ #
# SECTION 7 — PER-FRAME POINT CLOUD GENERATION                               #
# ============================================================================ #

def depth_to_pointcloud(depth_map, image_bgr, camera_params,
                         extrinsics, depth_scale=1.0, max_depth_metric=10.0):
    """
    Back-project a depth map into 3D world-space point cloud.

    The pinhole camera model is:
      x_cam = (u - cx) * depth / fx
      y_cam = (v - cy) * depth / fy
      z_cam = depth

    Then transform from camera to world coordinates:
      p_world = R^T @ (p_cam - t)
    (Inverse of COLMAP's world-to-camera transform: p_cam = R @ p_world + t)

    Args:
        depth_map    : (H, W) float32 normalized depth [0, 1]
        image_bgr    : (H, W, 3) uint8 OpenCV BGR image
        camera_params: dict {fx, fy, cx, cy, width, height}
        extrinsics   : dict {R: (3,3), t: (3,)}  world-to-camera
        depth_scale  : float, converts normalized depth to metric units
        max_depth_metric: discard points farther than this (meters)

    Returns:
        points (N, 3) float32 in world coordinates
        colors (N, 3) float32 RGB [0, 1]
    """
    h_depth, w_depth = depth_map.shape
    h_cam,   w_cam   = camera_params["height"], camera_params["width"]

    # Scale intrinsics to match depth map resolution
    # (MiDaS may resize the image internally)
    sx = w_depth / w_cam
    sy = h_depth / h_cam
    fx = camera_params["fx"] * sx
    fy = camera_params["fy"] * sy
    cx = camera_params["cx"] * sx
    cy = camera_params["cy"] * sy

    # Create pixel coordinate grids (H, W)
    u, v = np.meshgrid(np.arange(w_depth, dtype=np.float32),
                       np.arange(h_depth, dtype=np.float32))

    # Scale depth to metric
    depth_metric = depth_map * depth_scale   # (H, W)

    # Mask out invalid depth (too close, too far, or zero)
    valid = (depth_metric > 0.01) & (depth_metric < max_depth_metric)

    # Back-project valid pixels to camera 3D coordinates
    x_cam = (u[valid] - cx) * depth_metric[valid] / fx
    y_cam = (v[valid] - cy) * depth_metric[valid] / fy
    z_cam = depth_metric[valid]
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (N, 3)

    # Extract colors for valid pixels (convert BGR → RGB and normalize to [0,1])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Resize if needed to match depth map resolution
    if (h_depth, w_depth) != (h_cam, w_cam):
        image_rgb = cv2.resize(image_rgb, (w_depth, h_depth))
    colors = image_rgb[valid].astype(np.float32) / 255.0  # (N, 3)

    # Transform camera → world coordinates
    R = extrinsics["R"]   # (3, 3) world-to-camera
    t = extrinsics["t"]   # (3,)   world-to-camera translation
    # Inverse: p_world = R^T @ (p_cam - t)
    pts_world = (pts_cam - t) @ R   # equivalent to R^T @ (pts_cam - t)^T

    return pts_world.astype(np.float32), colors


def generate_frame_pointclouds(image_paths, colmap_data, midas_model,
                                midas_transform, midas_device,
                                max_frames=None,
                                depth_max_output_resolution=None):
    """
    Generate one Open3D point cloud per image frame using:
      - COLMAP camera pose (extrinsics + intrinsics)
      - MiDaS depth estimate scaled to approximate metric units

    This is the bridge between the 2D image domain and 3D reconstruction.

    Args:
        image_paths   : list of image file paths
        colmap_data   : dict from parse_colmap_reconstruction()
        midas_model   : loaded MiDaS model
        midas_transform: MiDaS preprocessing transform
        midas_device  : torch.device
        max_frames    : cap number of frames to process
        depth_max_output_resolution: optional max depth-map side length

    Returns:
        list of o3d.geometry.PointCloud (one per successfully processed frame)
    """
    print_section("Generating Dense Frame Point Clouds")

    cameras     = colmap_data["cameras"]
    images_meta = colmap_data["images"]
    sparse_pts  = colmap_data["points3d_xyz"]

    # Map image filenames to their COLMAP metadata
    img_name_map = {os.path.basename(p): p for p in image_paths}

    # Intersect available images with those COLMAP registered
    registered_names = [
        name for name in img_name_map
        if name in images_meta
    ]

    if not registered_names:
        raise RuntimeError(
            "None of the images were registered by COLMAP. "
            "Check that IMAGE_FOLDER contains the same images COLMAP processed."
        )

    print(f"  COLMAP registered {len(registered_names)} / {len(image_paths)} images")

    if max_frames and len(registered_names) > max_frames:
        # Sample evenly across the sequence
        indices = np.linspace(0, len(registered_names) - 1, max_frames, dtype=int)
        registered_names = [registered_names[i] for i in indices]
        print(f"  Using {len(registered_names)} frames (MAX_IMAGES_FOR_DENSE = {max_frames})")

    frame_pcds = []

    for idx, img_name in enumerate(registered_names):
        img_path = img_name_map[img_name]
        meta     = images_meta[img_name]
        cam      = cameras[meta["camera_id"]]

        print(f"  Frame {idx+1:3d}/{len(registered_names)}  {img_name}", end="  ")

        # Load image
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print("→ [SKIP] Could not load image")
            continue

        # Estimate depth with MiDaS
        t0 = time.time()
        depth_norm = estimate_depth(
            image_bgr,
            midas_model,
            midas_transform,
            midas_device,
            max_output_resolution=depth_max_output_resolution,
        )
        dt_depth   = time.time() - t0

        # Compute metric scale using visible COLMAP sparse points
        scale = compute_depth_scale(meta["R"], meta["t"], sparse_pts)

        # Back-project to 3D
        extrinsics = {"R": meta["R"], "t": meta["t"]}
        pts, cols  = depth_to_pointcloud(
            depth_norm, image_bgr, cam, extrinsics,
            depth_scale=scale,
            max_depth_metric=50.0,  # Adjust for your scene scale
        )

        if len(pts) < 100:
            print(f"→ [SKIP] Too few valid points ({len(pts)})")
            continue

        # Build Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)

        # Light downsample per frame to keep RAM manageable
        pcd = pcd.voxel_down_sample(VOXEL_SIZE * 0.5)

        frame_pcds.append(pcd)
        print(f"→ {len(pcd.points):,} pts  (depth in {dt_depth:.1f}s, scale={scale:.3f})")

    print(f"\n  Generated {len(frame_pcds)} frame point clouds.")
    return frame_pcds


# ============================================================================ #
# SECTION 8 — ICP REGISTRATION                                               #
# ============================================================================ #

def icp_register_frames(frame_pcds):
    """
    Align multiple per-frame point clouds using sequential Point-to-Plane ICP.

    WHY ICP IS NEEDED HERE:
      - COLMAP camera poses are accurate for sparse reconstruction.
      - MiDaS depth is relative (not perfectly metric) and may have small
        per-frame scale inconsistencies.
      - ICP corrects these residual alignment errors between frames by finding
        the rigid transformation that minimizes point-to-plane distance.

    HOW IT WORKS:
      1. Frame 0 is fixed (the reference / target).
      2. Each subsequent frame is registered to the growing merged cloud.
      3. We use Point-to-Plane ICP (more accurate than Point-to-Point ICP
         because it accounts for surface orientation via normals).
      4. If ICP fitness is very low, we fall back to the original pose
         (no additional correction applied).

    Returns:
        registered_pcds : list of aligned PointCloud objects
        transforms      : list of 4x4 numpy arrays (the ICP corrections)
    """
    print_section("ICP Registration — Aligning Frame Point Clouds")

    if len(frame_pcds) < 2:
        print("  Only one frame — skipping ICP.")
        return frame_pcds, [np.eye(4)]

    def prepare_for_icp(pcd, vox):
        """Downsample + estimate normals (required for Point-to-Plane ICP)."""
        down = pcd.voxel_down_sample(vox)
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30)
        )
        down.orient_normals_towards_camera_location(
            np.array([0.0, 0.0, 0.0])
        )
        return down

    icp_voxel = VOXEL_SIZE * 2          # ICP uses coarser resolution (faster)
    max_dist  = ICP_MAX_CORRESPONDENCE_DISTANCE

    registered_pcds = [frame_pcds[0]]   # Frame 0 = reference
    transforms      = [np.eye(4)]

    # Grow the reference cloud by accumulating registered frames
    target_pcd = copy.deepcopy(frame_pcds[0])

    print(f"  ICP params: voxel={icp_voxel:.4f}, max_dist={max_dist:.4f}, "
          f"max_iter={ICP_MAX_ITERATIONS}")
    print(f"  Frames to register: {len(frame_pcds) - 1}")
    print()

    for i in range(1, len(frame_pcds)):
        source = frame_pcds[i]

        source_down = prepare_for_icp(source, icp_voxel)
        target_down = prepare_for_icp(target_pcd, icp_voxel)

        result = o3d.pipelines.registration.registration_icp(
            source_down,
            target_down,
            max_correspondence_distance=max_dist,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=ICP_MAX_ITERATIONS,
            ),
        )

        T        = result.transformation
        fitness  = result.fitness
        rmse     = result.inlier_rmse
        shift    = np.linalg.norm(T[:3, 3])       # How much ICP moved this frame
        rotation = np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))

        status = "✓" if fitness > 0.3 else "⚠ low fitness"
        print(f"  Frame {i:3d}: fitness={fitness:.4f}  RMSE={rmse:.6f}  "
              f"shift={shift:.4f}m  rot={np.degrees(rotation):.2f}°  {status}")

        if fitness < 0.05:
            print(f"           → ICP did not converge for frame {i}. "
                  f"Using original COLMAP pose only.")
            T = np.eye(4)

        # Apply ICP correction to full-resolution source cloud
        source_aligned = copy.deepcopy(source)
        source_aligned.transform(T)

        registered_pcds.append(source_aligned)
        transforms.append(T)

        # Merge aligned frame into growing reference
        target_pcd = target_pcd + source_aligned
        # Every 5 frames, downsample the growing target to keep it manageable
        if i % 5 == 0:
            target_pcd = target_pcd.voxel_down_sample(icp_voxel * 0.5)

    print(f"\n  ICP complete. {len(registered_pcds)} frames aligned.")
    return registered_pcds, transforms


# ============================================================================ #
# SECTION 9 — POINT CLOUD CLEANING                                           #
# ============================================================================ #

def merge_and_clean(frame_pcds):
    """
    Merge all registered frame point clouds and apply:
      1. Voxel downsampling — reduces redundant points from overlapping frames
      2. Statistical Outlier Removal (SOR) — removes noise/flying points

    Returns: clean merged Open3D PointCloud
    """
    print_section("Merging + Cleaning Point Cloud")

    # Merge all frames
    merged = o3d.geometry.PointCloud()
    for pcd in frame_pcds:
        merged += pcd

    n_raw = len(merged.points)
    print(f"  Raw merged cloud    : {n_raw:,} points")

    # Step 1: Voxel downsampling (remove redundant overlapping points)
    merged = merged.voxel_down_sample(VOXEL_SIZE)
    n_down = len(merged.points)
    print(f"  After voxel downsample ({VOXEL_SIZE}m): {n_down:,} points  "
          f"({100*n_down/n_raw:.1f}% retained)")

    # Step 2: Statistical Outlier Removal
    # For each point: compute mean distance to its N nearest neighbors.
    # Points whose mean distance > (global_mean + std_ratio × global_std) → removed.
    cl, ind = merged.remove_statistical_outlier(
        nb_neighbors=SOR_NB_NEIGHBORS,
        std_ratio=SOR_STD_RATIO,
    )
    n_clean = len(cl.points)
    print(f"  After SOR (nb={SOR_NB_NEIGHBORS}, std={SOR_STD_RATIO:.1f}):  "
          f"{n_clean:,} points  ({100*n_clean/n_down:.1f}% retained)")

    return cl


# ============================================================================ #
# SECTION 10 — TESLA-STYLE VISUALIZATION COLORING                            #
# ============================================================================ #

def apply_tesla_coloring(pcd, mode=None, colormap=None):
    """
    Apply Tesla-inspired visualization coloring to the point cloud.

    Tesla's occupancy network / BEV visualization uses:
      - Height-based plasma colormap (Z-axis → purple → blue → yellow → red)
      - This reveals the 3D structure at a glance — floor vs objects vs ceiling
      - The same visual language is used in Tesla FSD's internal visualization

    COLOR MODES:
      "height"   → Height-based plasma coloring  (Tesla BEV style)
      "rgb"      → Keep original photo colors     (photorealistic)
      "distance" → Distance from scene centroid   (radar/sonar style)
      "normals"  → Surface normal directions       (reveals flat surfaces)

    Args:
        pcd     : Open3D PointCloud
        mode    : Override COLOR_MODE config if provided
        colormap: Override COLOR_MAP config if provided

    Returns: PointCloud with colors applied
    """
    mode     = mode     or COLOR_MODE
    colormap = colormap or COLOR_MAP

    points = np.asarray(pcd.points)
    n      = len(points)

    print(f"\n[COLOR] Applying Tesla-style coloring: mode='{mode}', cmap='{colormap}'")

    if mode == "height":
        # --- Tesla BEV style: color by vertical height (Z-axis) ---
        z = points[:, 2]

        # Clip extreme values so the colormap isn't wasted on outliers
        z_lo = np.percentile(z, HEIGHT_CLIP_LOW)
        z_hi = np.percentile(z, HEIGHT_CLIP_HIGH)
        z_norm = np.clip((z - z_lo) / (z_hi - z_lo + 1e-8), 0.0, 1.0)

        cmap   = plt.get_cmap(colormap)
        colors = cmap(z_norm)[:, :3]  # (N, 4) → take only RGB, drop alpha

    elif mode == "rgb":
        # Keep original image colors if they exist
        if pcd.has_colors():
            print("[COLOR] Using original photo colors (RGB mode)")
            return pcd   # No change needed
        else:
            # Fallback to gray
            colors = np.tile([0.65, 0.65, 0.65], (n, 1))
            print("[COLOR] No original colors found — using gray.")

    elif mode == "distance":
        # --- Radar/sonar style: color by distance from scene center ---
        centroid  = points.mean(axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        d_norm    = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
        cmap      = plt.get_cmap(colormap)
        colors    = cmap(d_norm)[:, :3]

    elif mode == "normals":
        # --- Normal-based: color by surface orientation ---
        # Encodes: Red=X direction, Green=Y direction, Blue=Z direction
        # Horizontal surfaces → blue. Vertical walls → red/green.
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=VOXEL_SIZE * 3, max_nn=30
                )
            )
        normals = np.asarray(pcd.normals)
        colors  = np.abs(normals)  # |nx|, |ny|, |nz| → always [0,1]

    else:
        colors = np.tile([0.5, 0.5, 0.5], (n, 1))

    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def transfer_colors_to_mesh(mesh, pcd):
    """
    Transfer point cloud colors to mesh vertices using nearest-neighbor lookup.

    The Poisson mesh vertices don't automatically carry point colors.
    This function finds the nearest point cloud point for each mesh vertex
    and copies its color.

    Args:
        mesh: Open3D TriangleMesh (vertices without colors)
        pcd : Open3D PointCloud with colors
    Returns: mesh with vertex colors assigned
    """
    from scipy.spatial import cKDTree

    pcd_pts  = np.asarray(pcd.points)
    pcd_cols = np.asarray(pcd.colors)
    mesh_vts = np.asarray(mesh.vertices)

    print(f"[MESH]  Transferring colors: {len(pcd_pts):,} pts → {len(mesh_vts):,} vertices")

    tree = cKDTree(pcd_pts)
    _, indices = tree.query(mesh_vts, k=1, workers=-1)  # k=1: nearest neighbor

    vertex_colors = pcd_cols[indices]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    return mesh


# ============================================================================ #
# SECTION 11 — SURFACE MESH RECONSTRUCTION                                   #
# ============================================================================ #

def reconstruct_surface_mesh(pcd):
    """
    Reconstruct a 3D surface mesh from the cleaned point cloud.

    ALGORITHM: Poisson Surface Reconstruction
      - Fits a smooth implicit surface through the points
      - Produces a watertight (closed) mesh
      - Requires point normals (estimated automatically)
      - Quality controlled by octree depth (POISSON_DEPTH parameter)

    WHY POISSON:
      - Produces the smoothest, cleanest mesh
      - Great for architectural surfaces, products, medical models
      - The "watertight" property is useful for 3D printing, CAD, and
        real-time physics simulation in game engines

    Returns: Open3D TriangleMesh
    """
    print_section("Poisson Surface Mesh Reconstruction")

    # Estimate normals — required for Poisson
    print(f"  Estimating point normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=VOXEL_SIZE * 3,
            max_nn=30,
        )
    )
    # Orient normals consistently (pointing outward)
    pcd.orient_normals_consistent_tangent_plane(100)

    print(f"  Running Poisson reconstruction (depth={POISSON_DEPTH})...")
    print(f"  (Higher depth = more detail. This may take 1–5 minutes.)")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH, linear_fit=False
    )

    n_before = len(mesh.triangles)

    # Remove low-density regions (these are artifacts in space not covered by points)
    if MESH_DENSITY_QUANTILE > 0:
        dens = np.asarray(densities)
        threshold = np.quantile(dens, MESH_DENSITY_QUANTILE)
        mesh.remove_vertices_by_mask(dens < threshold)
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

    n_after = len(mesh.triangles)
    print(f"  Triangles: {n_before:,} → {n_after:,} (after density cleaning)")

    # Transfer Tesla-style colors from point cloud to mesh vertices
    mesh = transfer_colors_to_mesh(mesh, pcd)
    mesh.compute_vertex_normals()

    return mesh


# ============================================================================ #
# SECTION 12 — TESLA BEV VOXEL MESH                                          #
# ============================================================================ #

def create_voxel_mesh(pcd):
    """
    Create a Tesla-style voxel occupancy mesh.

    This builds a grid of colored cubes — one per occupied voxel in the point cloud.
    This is the same visual representation used in Tesla's occupancy network
    (their 3D scene understanding for FSD), and in many AV dashboards.

    APPLICATIONS:
      - Autonomous vehicles: BEV occupancy grid visualization
      - Construction: 3D site model block representation
      - Architectural: room occupancy analysis
      - Robotics: collision map

    The construction is fully vectorized (no Python loops) using NumPy broadcasting.

    Returns: Open3D TriangleMesh of cubes
    """
    print_section("Tesla BEV Voxel Mesh Generation")

    # Downsample to voxel grid
    voxel_pcd = pcd.voxel_down_sample(VOXEL_MESH_CUBE_SIZE)
    voxel_pts = np.asarray(voxel_pcd.points)
    n = len(voxel_pts)
    print(f"  Occupied voxels  : {n:,}  (size = {VOXEL_MESH_CUBE_SIZE}m)")

    # Apply Tesla-style height coloring to voxels
    apply_tesla_coloring(voxel_pcd, mode=COLOR_MODE, colormap=COLOR_MAP)
    voxel_cols = np.asarray(voxel_pcd.colors)

    # Unit cube: 8 vertices centered at origin, half-edge = 0.5
    unit_verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],   # Bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],   # Top face
    ], dtype=np.float64) - 0.5  # Center each cube at its voxel center

    # 12 triangles per cube (2 per face), consistent winding for outward normals
    unit_tris = np.array([
        [0, 2, 1], [0, 3, 2],   # Bottom   (-Z)
        [4, 5, 6], [4, 6, 7],   # Top      (+Z)
        [0, 1, 5], [0, 5, 4],   # Front    (-Y)
        [2, 3, 7], [2, 7, 6],   # Back     (+Y)
        [0, 4, 7], [0, 7, 3],   # Left     (-X)
        [1, 2, 6], [1, 6, 5],   # Right    (+X)
    ], dtype=np.int32)

    # ── Vectorized construction ────────────────────────────────────────────────
    # all_verts: (N_voxels × 8, 3) — each voxel's 8 cube corners
    all_verts = (unit_verts * VOXEL_MESH_CUBE_SIZE) + voxel_pts[:, np.newaxis, :]
    all_verts = all_verts.reshape(-1, 3)  # Flatten: (N×8, 3)

    # all_tris: (N_voxels × 12, 3) — triangle indices with per-voxel offset
    offsets   = (np.arange(n) * 8)[:, np.newaxis, np.newaxis]
    all_tris  = (unit_tris[np.newaxis, :, :] + offsets).reshape(-1, 3)

    # Colors: each of the 8 vertices of a voxel cube gets that voxel's color
    all_cols = np.repeat(voxel_cols, 8, axis=0)   # (N×8, 3)

    # Build Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices      = o3d.utility.Vector3dVector(all_verts)
    mesh.triangles     = o3d.utility.Vector3iVector(all_tris)
    mesh.vertex_colors = o3d.utility.Vector3dVector(all_cols)
    mesh.compute_vertex_normals()

    print(f"  Voxel mesh built : {len(all_verts):,} vertices, {len(all_tris):,} triangles")
    return mesh


# ============================================================================ #
# SECTION 13 — EXPORT (GLB / GLTF / PLY / OBJ)                              #
# ============================================================================ #

def export_pointcloud_ply(pcd, filepath):
    """Export cleaned point cloud as binary PLY file."""
    o3d.io.write_point_cloud(filepath, pcd, write_ascii=False, compressed=True)
    size_mb = os.path.getsize(filepath) / 1e6
    print(f"[EXPORT] PLY  → {filepath}  ({size_mb:.1f} MB)")


def export_mesh_obj(mesh, filepath):
    """Export mesh as .OBJ (compatible with Blender, Maya, 3ds Max)."""
    o3d.io.write_triangle_mesh(filepath, mesh, write_ascii=True)
    size_mb = os.path.getsize(filepath) / 1e6
    print(f"[EXPORT] OBJ  → {filepath}  ({size_mb:.1f} MB)")


def export_mesh_glb(mesh, filepath):
    """
    Export Open3D mesh as .GLB (binary GLTF 2.0).

    GLB is a single-file binary format for 3D scenes.
    Supported by virtually every modern 3D platform:
      • Browsers  : Three.js, Babylon.js, <model-viewer>
      • Game engines: Unity, Unreal, Godot
      • AR/VR     : Apple AR Quick Look, Google Scene Viewer, Meta Spark
      • Design    : Blender, Figma, Sketchfab, Adobe Dimension
      • Mobile    : iOS (USDZ auto-converts), Android (ARCore)

    Args:
        mesh    : Open3D TriangleMesh with vertex colors
        filepath: Output .glb path
    """
    if not TRIMESH_AVAILABLE:
        print("[EXPORT] trimesh not installed. Skipping GLB export.")
        print("         Install with: pip install trimesh")
        return

    import trimesh

    vertices = np.asarray(mesh.vertices)
    faces    = np.asarray(mesh.triangles)
    colors   = np.asarray(mesh.vertex_colors)

    # trimesh requires RGBA uint8 vertex colors
    rgba = np.ones((len(vertices), 4), dtype=np.uint8) * 255
    rgba[:, :3] = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    tm = trimesh.Trimesh(
        vertices     = vertices,
        faces        = faces,
        vertex_colors= rgba,
        process      = False,   # Don't auto-modify the mesh
    )
    tm.fix_normals()
    tm.export(filepath)

    size_mb = os.path.getsize(filepath) / 1e6
    print(f"[EXPORT] GLB  → {filepath}  ({size_mb:.1f} MB)")
    print(f"         View online: https://gltf-viewer.donmccurdy.com  (drag & drop)")
    print(f"         View online: https://sandbox.babylonjs.com       (paste URL or upload)")


def export_mesh_gltf(mesh, filepath):
    """Export mesh as .GLTF (JSON text format — same data as GLB, larger but human-readable)."""
    if not TRIMESH_AVAILABLE:
        return

    import trimesh

    vertices = np.asarray(mesh.vertices)
    faces    = np.asarray(mesh.triangles)
    colors   = np.asarray(mesh.vertex_colors)
    rgba = np.ones((len(vertices), 4), dtype=np.uint8) * 255
    rgba[:, :3] = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    scene = trimesh.Scene(
        trimesh.Trimesh(vertices=vertices, faces=faces,
                        vertex_colors=rgba, process=False)
    )
    scene.export(filepath, file_type="gltf")
    size_mb = os.path.getsize(filepath) / 1e6
    print(f"[EXPORT] GLTF → {filepath}  ({size_mb:.1f} MB)")


# ============================================================================ #
# SECTION 14 — MAIN PIPELINE                                                 #
# ============================================================================ #

def main():
    t_start = time.time()

    print("=" * 70)
    print("  Multi-Image SfM 3D Reconstruction Pipeline")
    print("=" * 70)

    # ── Apply application preset ──────────────────────────────────────────────
    apply_preset(APPLICATION_MODE)

    # ── Setup output directories ──────────────────────────────────────────────
    setup_directories()

    # ── Load image list ───────────────────────────────────────────────────────
    image_paths = load_image_paths(IMAGE_FOLDER)

    # ── COLMAP SfM ────────────────────────────────────────────────────────────
    colmap_available = check_colmap(COLMAP_EXECUTABLE)

    if colmap_available:
        txt_folder = run_colmap_sfm(
            image_folder    = IMAGE_FOLDER,
            output_folder   = OUTPUT_FOLDER,
            colmap_exe      = COLMAP_EXECUTABLE,
            matcher         = COLMAP_MATCHER,
            use_gpu         = COLMAP_USE_GPU,
            image_quality   = COLMAP_IMAGE_QUALITY,
        )
        colmap_data = parse_colmap_reconstruction(txt_folder)
    else:
        print("\n[WARN] COLMAP not found at:", COLMAP_EXECUTABLE)
        print("       Falling back to sparse-only mode using COLMAP output if available.")
        print("       Install COLMAP: https://colmap.github.io/install.html")
        # Try to load an existing COLMAP reconstruction if it exists
        existing = os.path.join(OUTPUT_FOLDER, "colmap_sparse", "0_txt")
        if not os.path.isdir(existing):
            raise FileNotFoundError(
                "COLMAP not installed and no existing reconstruction found. "
                "Please install COLMAP and re-run."
            )
        colmap_data = parse_colmap_reconstruction(existing)

    # Show the sparse COLMAP point cloud
    print_section("COLMAP Sparse Point Cloud")
    sparse_pcd = o3d.geometry.PointCloud()
    sparse_pcd.points = o3d.utility.Vector3dVector(colmap_data["points3d_xyz"])
    sparse_pcd.colors = o3d.utility.Vector3dVector(colmap_data["points3d_rgb"])
    print(f"  Sparse cloud: {len(sparse_pcd.points):,} points")

    if VISUALIZE_INTERMEDIATE:
        safe_visualize([sparse_pcd], "COLMAP Sparse Reconstruction")

    # ── MiDaS Depth Estimation ────────────────────────────────────────────────
    if TORCH_AVAILABLE:
        midas_model, midas_transform, midas_device = load_midas(DEPTH_MODEL, DEPTH_DEVICE)
    else:
        midas_model = midas_transform = midas_device = None
        print("[WARN] Skipping depth estimation (PyTorch not available)")

    # ── Per-Frame Dense Point Clouds ──────────────────────────────────────────
    if midas_model is not None:
        frame_pcds = generate_frame_pointclouds(
            image_paths    = image_paths,
            colmap_data    = colmap_data,
            midas_model    = midas_model,
            midas_transform= midas_transform,
            midas_device   = midas_device,
            max_frames     = MAX_IMAGES_FOR_DENSE,
            depth_max_output_resolution = DEPTH_MAX_OUTPUT_RESOLUTION,
        )
    else:
        # Use the COLMAP sparse cloud as a single "frame"
        print("[WARN] Using COLMAP sparse cloud only (no depth estimation).")
        frame_pcds = [sparse_pcd]

    if not frame_pcds:
        raise RuntimeError("No frame point clouds generated. Check your images and COLMAP output.")

    # ── ICP Registration ──────────────────────────────────────────────────────
    if len(frame_pcds) > 1:
        # Visualize BEFORE ICP: color each frame differently to see misalignment
        if VISUALIZE_INTERMEDIATE:
            print_section("Before ICP — Colored by Frame (each color = one frame)")
            vis_before = []
            rng = np.random.RandomState(42)
            for i, pcd in enumerate(frame_pcds[:min(8, len(frame_pcds))]):
                tmp = copy.deepcopy(pcd)
                c = rng.rand(3)
                tmp.paint_uniform_color(c.tolist())
                vis_before.append(tmp)
            safe_visualize(vis_before, "BEFORE ICP — Each frame in a different color")

        registered_pcds, transforms = icp_register_frames(frame_pcds)

        # Visualize AFTER ICP
        if VISUALIZE_INTERMEDIATE:
            print_section("After ICP — Colored by Frame (alignment should be tighter)")
            vis_after = []
            rng = np.random.RandomState(42)
            for i, pcd in enumerate(registered_pcds[:min(8, len(registered_pcds))]):
                tmp = copy.deepcopy(pcd)
                c = rng.rand(3)
                tmp.paint_uniform_color(c.tolist())
                vis_after.append(tmp)
            safe_visualize(vis_after, "AFTER ICP — Frames should now align cleanly")
    else:
        registered_pcds = frame_pcds

    # ── Merge + Clean ─────────────────────────────────────────────────────────
    clean_pcd = merge_and_clean(registered_pcds)

    # ── Tesla-Style Coloring ──────────────────────────────────────────────────
    apply_tesla_coloring(clean_pcd)

    if VISUALIZE_INTERMEDIATE:
        safe_visualize([clean_pcd], f"Clean Point Cloud — Tesla '{COLOR_MAP}' Coloring")

    # ── Surface Mesh ──────────────────────────────────────────────────────────
    surface_mesh = reconstruct_surface_mesh(copy.deepcopy(clean_pcd))

    if VISUALIZE_FINAL:
        safe_visualize([surface_mesh], "Poisson Surface Mesh")

    # ── Voxel Mesh (Tesla BEV) ────────────────────────────────────────────────
    if GENERATE_VOXEL_MESH:
        voxel_mesh = create_voxel_mesh(clean_pcd)
        if VISUALIZE_FINAL:
            safe_visualize([voxel_mesh], "Tesla-Style Voxel Occupancy Mesh")

    # ── Export ────────────────────────────────────────────────────────────────
    print_section("Exporting Results")

    base = os.path.join(OUTPUT_FOLDER, OUTPUT_NAME)

    if EXPORT_PLY:
        export_pointcloud_ply(clean_pcd, base + "_pointcloud.ply")

    if EXPORT_GLB:
        export_mesh_glb(surface_mesh, base + "_surface.glb")
        if GENERATE_VOXEL_MESH:
            export_mesh_glb(voxel_mesh, base + "_voxel_bev.glb")

    if EXPORT_OBJ:
        export_mesh_obj(surface_mesh, base + "_surface.obj")

    if EXPORT_GLTF:
        export_mesh_gltf(surface_mesh, base + "_surface.gltf")

    # ── Summary ───────────────────────────────────────────────────────────────
    t_total = time.time() - t_start
    print_section("Pipeline Complete")
    print(f"  Total time        : {t_total/60:.1f} minutes")
    print(f"  Output folder     : {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"  Point cloud       : {len(clean_pcd.points):,} points")
    print(f"  Surface triangles : {len(surface_mesh.triangles):,}")
    if GENERATE_VOXEL_MESH:
        print(f"  Voxel cubes       : {len(voxel_mesh.triangles) // 12:,}")
    print()
    print("  Files created:")
    for f in os.listdir(OUTPUT_FOLDER):
        if OUTPUT_NAME in f:
            fpath = os.path.join(OUTPUT_FOLDER, f)
            print(f"    {f:40s}  {os.path.getsize(fpath)/1e6:.1f} MB")

    print()
    print("  View .GLB files online:")
    print("    https://gltf-viewer.donmccurdy.com  ← drag & drop your .glb here")
    print("    https://sandbox.babylonjs.com")


if __name__ == "__main__":
    main()
