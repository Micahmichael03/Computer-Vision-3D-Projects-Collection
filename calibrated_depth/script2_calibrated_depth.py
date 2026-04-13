#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 SCRIPT 2: Camera Calibration + Depth Estimation 3D Reconstruction Pipeline
 Written for: Michael Micah — Professional Computer Vision Pipeline
================================================================================

 PIPELINE OVERVIEW
 -----------------
  Mode A — Calibrated Single/Multi-Image Reconstruction:
    Stage 1 → Camera calibration from checkerboard images  (or load saved)
    Stage 2 → Undistort scene images using calibration
    Stage 3 → MiDaS depth estimation on undistorted images
    Stage 4 → Back-project to metric 3D using calibrated intrinsics
    Stage 5 → Merge + SOR cleaning
    Stage 6 → Tesla-style coloring + Poisson mesh
    Stage 7 → Export GLB / GLTF

  Mode B — Stitched Panorama Reconstruction:
    Stage 1 → Load scene images (no calibration required)
    Stage 2 → Stitch overlapping images into a panorama (OpenCV Stitcher)
    Stage 3 → Estimate virtual camera intrinsics for the panorama
    Stage 4 → MiDaS depth estimation on the panorama
    Stage 5 → Back-project to 3D
    Stage 6 → Merge + SOR cleaning
    Stage 7 → Tesla-style coloring + Poisson mesh
    Stage 8 → Export GLB / GLTF

 WHEN TO USE EACH MODE
 ---------------------
  USE CALIBRATION MODE when:
    - You want metric-accurate reconstruction (for medical, AV, construction)
    - You have a checkerboard calibration target
    - Accuracy and undistortion matter
    - E-commerce products, medical scans, AV camera rigs

  USE STITCHING MODE when:
    - You want a panoramic overview of a scene
    - You have overlapping wide-angle shots of a room or landscape
    - Architectural walkthroughs, site surveys, real estate photography

 HOW TO RUN
 ----------
  1. Install dependencies:
       pip install opencv-python numpy open3d torch torchvision
       pip install trimesh matplotlib pillow scipy

  2. For Calibration Mode (MODE = "calibrate"):
       a. Print a checkerboard target:
            https://calib.io/pages/camera-calibration-tools  (free download)
       b. Photograph it from 15–25 different angles, covering the full frame
       c. Put calibration images in CALIBRATION_IMAGE_FOLDER
       d. Put your scene images in SCENE_IMAGE_FOLDER

  3. For Stitching Mode (MODE = "stitch"):
       a. Put overlapping scene images in SCENE_IMAGE_FOLDER
       b. Images must have 30–50% overlap for stitching to work

  4. Run:
       python script2_calibrated_depth.py

  5. View results at: https://gltf-viewer.donmccurdy.com

================================================================================
"""

# ============================================================================ #
# SECTION 0 — CONFIGURATION (EDIT ONLY THIS SECTION)                         #
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# 0.1  Pipeline Mode                                                           #
# ---------------------------------------------------------------------------- #
# "calibrate" → Use camera calibration + undistortion + per-image depth
# "stitch"    → Stitch overlapping images into panorama + single depth pass
# "direct"    → Skip calibration and stitching. Just run depth estimation
#               directly on your images (uses a default pinhole assumption)
PIPELINE_MODE = "calibrate"

# ---------------------------------------------------------------------------- #
# 0.2  Application Mode                                                        #
# ---------------------------------------------------------------------------- #
# Controls color, mesh quality, and export settings.
# "general" | "architectural" | "medical" | "ecommerce" | "autonomous" | "construction"
APPLICATION_MODE = "general"

# ---------------------------------------------------------------------------- #
# 0.3  Paths                                                                   #
# ---------------------------------------------------------------------------- #
# Folder containing checkerboard calibration images
# (Only used when PIPELINE_MODE = "calibrate")
CALIBRATION_IMAGE_FOLDER = "calibration_images/"

# Folder containing scene images to reconstruct
SCENE_IMAGE_FOLDER = "scene_images/"

# Where to save all outputs
OUTPUT_FOLDER = "output_script2/"

# Filename for saving/loading camera calibration (avoids re-calibrating every run)
CALIBRATION_SAVE_FILE = "camera_calibration.npz"

# ---------------------------------------------------------------------------- #
# 0.4  Camera Calibration Settings                                             #
# ---------------------------------------------------------------------------- #
# Checkerboard inner corners (not outer corners, not total squares).
# For a standard 9×6 board: 8 inner corners horizontally, 5 vertically.
# Count: CHECKERBOARD_COLS × CHECKERBOARD_ROWS = inner corners
CHECKERBOARD_COLS = 9    # Number of inner corners along the long side
CHECKERBOARD_ROWS = 6    # Number of inner corners along the short side

# Physical size of one square on your printed checkerboard (in millimeters)
# Measure carefully — this directly affects metric depth accuracy.
SQUARE_SIZE_MM = 25.0    # 25mm = 2.5cm squares

# Minimum calibration images to accept (more = better accuracy; aim for 15–20)
MIN_CALIBRATION_IMAGES = 10

# If True, show each calibration image with detected corners drawn on it
SHOW_CALIBRATION_CORNERS = False

# ---------------------------------------------------------------------------- #
# 0.5  Stitching Settings (PIPELINE_MODE = "stitch")                          #
# ---------------------------------------------------------------------------- #
# Stitching mode: "panorama" for full 360/wide-angle | "scan" for flatbed-style
STITCH_MODE = "panorama"

# Maximum number of images to stitch (too many can be slow and memory-heavy)
MAX_STITCH_IMAGES = 30

# JPEG compression for the output panorama (0–100)
PANORAMA_JPEG_QUALITY = 95

# ---------------------------------------------------------------------------- #
# 0.6  Depth Estimation                                                        #
# ---------------------------------------------------------------------------- #
# MiDaS model variant (all work on CPU):
# "MiDaS_small" → Fastest (recommended for first tests)
# "DPT_Hybrid"  → Better quality
# "DPT_Large"   → Best quality, slowest
DEPTH_MODEL = "MiDaS_small"

# Maximum number of scene images to run depth estimation on
# Set None to process all images
MAX_DEPTH_IMAGES = 10

# ---------------------------------------------------------------------------- #
# 0.7  Point Cloud Processing                                                  #
# ---------------------------------------------------------------------------- #
VOXEL_SIZE       = 0.02
SOR_NB_NEIGHBORS = 20
SOR_STD_RATIO    = 2.0

# ---------------------------------------------------------------------------- #
# 0.8  Coloring + Mesh                                                         #
# ---------------------------------------------------------------------------- #
COLOR_MODE    = "height"     # "height" | "rgb" | "distance" | "normals"
COLOR_MAP     = "plasma"     # "plasma" | "turbo" | "viridis" | "inferno"
HEIGHT_CLIP_LOW  = 5
HEIGHT_CLIP_HIGH = 95
POISSON_DEPTH    = 9
MESH_DENSITY_QUANTILE = 0.01

# ---------------------------------------------------------------------------- #
# 0.9  Export                                                                  #
# ---------------------------------------------------------------------------- #
EXPORT_PLY  = True
EXPORT_GLB  = True
EXPORT_OBJ  = False
OUTPUT_NAME = "calibrated_reconstruction"

# ---------------------------------------------------------------------------- #
# 0.10  Visualization                                                          #
# ---------------------------------------------------------------------------- #
VISUALIZE_INTERMEDIATE = True
VISUALIZE_FINAL        = True

# ============================================================================ #
# END OF CONFIGURATION                                                         #
# ============================================================================ #


# ============================================================================ #
# SECTION 1 — IMPORTS                                                         #
# ============================================================================ #

import os
import sys
import glob
import copy
import time
import warnings

import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from PIL import Image

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not installed. Depth estimation unavailable.")
    print("       Install: pip install torch torchvision")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")


# ============================================================================ #
# SECTION 2 — UTILITY FUNCTIONS                                               #
# ============================================================================ #

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def setup_directories():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"[SETUP] Output folder: {os.path.abspath(OUTPUT_FOLDER)}")


def load_image_paths(folder, exts=("*.jpg", "*.jpeg", "*.png", "*.PNG")):
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def safe_visualize(geometries, window_name="Open3D"):
    if not (VISUALIZE_INTERMEDIATE or VISUALIZE_FINAL):
        return
    try:
        o3d.visualization.draw_geometries(
            geometries, window_name=window_name, width=1280, height=720
        )
    except Exception as e:
        print(f"[VIZ]  Could not open viewer: {e}")


def build_camera_matrix(fx, fy, cx, cy):
    """Build a 3x3 OpenCV camera intrinsic matrix."""
    return np.array([[fx, 0,  cx],
                     [0,  fy, cy],
                     [0,  0,  1 ]], dtype=np.float64)


# ============================================================================ #
# SECTION 3 — CAMERA CALIBRATION                                              #
# ============================================================================ #

def calibrate_camera(calibration_folder):
    """
    Compute camera intrinsics using multiple checkerboard images.

    THEORY:
      A real camera lens introduces distortion — straight lines in the world
      appear curved in images (barrel/pincushion distortion). Camera calibration
      finds:
        • camera_matrix K = [[fx, 0,  cx],
                              [0,  fy, cy],
                              [0,  0,  1 ]]
          where fx, fy = focal lengths (pixels), cx, cy = principal point (pixels)
        • dist_coeffs = [k1, k2, p1, p2, k3]
          radial (k) and tangential (p) distortion coefficients

      These are found by solving a system of equations from checkerboard corners
      whose 3D positions are exactly known (regular grid × SQUARE_SIZE_MM).

    WHY THIS MATTERS:
      - Without calibration: depth back-projection uses wrong intrinsics → 3D
        coordinates are warped
      - With calibration: images are undistorted → metric-accurate back-projection
      - For AV, medical, and construction: accuracy can be life-critical

    Args:
        calibration_folder: folder of checkerboard images

    Returns:
        camera_matrix (3x3), dist_coeffs (1x5), image_size (w, h)
    """
    print_section("Camera Calibration")
    print(f"  Checkerboard     : {CHECKERBOARD_COLS} × {CHECKERBOARD_ROWS} inner corners")
    print(f"  Square size      : {SQUARE_SIZE_MM} mm")
    print(f"  Images folder    : {calibration_folder}")

    cal_paths = load_image_paths(calibration_folder)
    if not cal_paths:
        raise FileNotFoundError(
            f"No calibration images found in '{calibration_folder}'.\n"
            f"Take 15–25 photos of a printed checkerboard from different angles."
        )

    print(f"  Found {len(cal_paths)} calibration images.")

    # 3D object points for the checkerboard (Z=0 because board lies in a flat plane)
    # objp[i] = (col_i * SQUARE_SIZE, row_i * SQUARE_SIZE, 0)
    objp = np.zeros((CHECKERBOARD_COLS * CHECKERBOARD_ROWS, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[
        0:CHECKERBOARD_COLS, 0:CHECKERBOARD_ROWS
    ].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM   # Convert to millimetres

    # Subpixel corner refinement parameters
    subpix_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,   # Max iterations
        0.001 # Epsilon (accuracy)
    )

    objpoints  = []   # 3D points in real-world space (same for every image)
    imgpoints  = []   # 2D points in image plane (different per image)
    image_size = None

    print("\n  Detecting checkerboard corners...")
    successful = 0

    for idx, path in enumerate(cal_paths):
        img  = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (width, height)

        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            (CHECKERBOARD_COLS, CHECKERBOARD_ROWS),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                  cv2.CALIB_CB_NORMALIZE_IMAGE  +
                  cv2.CALIB_CB_FAST_CHECK
        )

        status = "✓" if ret else "✗ corners not found"
        print(f"    [{idx+1:3d}/{len(cal_paths)}] {os.path.basename(path):35s}  {status}")

        if ret:
            # Refine corners to sub-pixel accuracy
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)

            objpoints.append(objp)
            imgpoints.append(corners_sub)
            successful += 1

            if SHOW_CALIBRATION_CORNERS:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, (CHECKERBOARD_COLS, CHECKERBOARD_ROWS),
                                          corners_sub, ret)
                cv2.imshow(f"Corners — {os.path.basename(path)}", vis)
                cv2.waitKey(300)

    if SHOW_CALIBRATION_CORNERS:
        cv2.destroyAllWindows()

    if successful < MIN_CALIBRATION_IMAGES:
        raise RuntimeError(
            f"Only {successful}/{len(cal_paths)} calibration images were usable. "
            f"Minimum required: {MIN_CALIBRATION_IMAGES}.\n"
            f"  Tips:\n"
            f"  - Ensure the full checkerboard is visible (not clipped by frame edges)\n"
            f"  - Use good lighting — avoid glare and shadows on the board\n"
            f"  - Cover different angles: flat, 45°, rotated, corners of the frame\n"
        )

    print(f"\n  Usable calibration images: {successful} / {len(cal_paths)}")
    print(f"  Running camera calibration...")

    # OpenCV camera calibration
    # Flags: CALIB_FIX_PRINCIPAL_POINT and CALIB_ZERO_TANGENT_DIST can be added
    # for simpler models (trading accuracy for robustness)
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,   # Initial camera matrix (None = auto)
        None,   # Initial dist_coeffs
    )

    # Compute reprojection error (how well the model fits the data)
    # Good calibration: < 1.0 pixel. Excellent: < 0.5 pixels.
    total_error = 0
    for i in range(len(objpoints)):
        imgpts_reprojected, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(imgpoints[i], imgpts_reprojected, cv2.NORM_L2) / len(imgpts_reprojected)
        total_error += error

    mean_reprojection_error = total_error / len(objpoints)

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    print(f"\n  ── Calibration Results ──────────────────────────────────")
    print(f"  Image size        : {image_size[0]} × {image_size[1]} px")
    print(f"  Focal length      : fx={fx:.2f} px,  fy={fy:.2f} px")
    print(f"  Principal point   : cx={cx:.2f} px,  cy={cy:.2f} px")
    print(f"  Distortion coeffs : {dist_coeffs.ravel()[:5]}")
    print(f"  Reprojection error: {mean_reprojection_error:.4f} px", end="  ")
    if mean_reprojection_error < 0.5:
        print("← Excellent! ✓")
    elif mean_reprojection_error < 1.0:
        print("← Good ✓")
    else:
        print("← Consider adding more calibration images for better accuracy")

    return camera_matrix, dist_coeffs, image_size


def save_calibration(camera_matrix, dist_coeffs, image_size, filepath):
    """Save calibration data to .npz file."""
    np.savez(filepath,
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             image_size=np.array(image_size))
    print(f"[CAL]   Calibration saved → {filepath}")


def load_calibration(filepath):
    """Load previously saved calibration data."""
    if not os.path.exists(filepath):
        return None, None, None

    data = np.load(filepath)
    camera_matrix = data["camera_matrix"]
    dist_coeffs   = data["dist_coeffs"]
    image_size    = tuple(data["image_size"])

    print(f"[CAL]   Loaded calibration from {filepath}")
    print(f"        fx={camera_matrix[0,0]:.1f}  fy={camera_matrix[1,1]:.1f}  "
          f"cx={camera_matrix[0,2]:.1f}  cy={camera_matrix[1,2]:.1f}")
    return camera_matrix, dist_coeffs, image_size


# ============================================================================ #
# SECTION 4 — IMAGE UNDISTORTION                                              #
# ============================================================================ #

def undistort_images(image_paths, camera_matrix, dist_coeffs, output_folder):
    """
    Remove lens distortion from each scene image using the calibrated parameters.

    After undistortion:
      - Straight lines in the world appear straight in the image
      - The camera model is now a perfect pinhole camera (no distortion)
      - Depth back-projection gives metric-accurate results

    The optimal new camera matrix is computed to ensure no valid pixels are lost
    (alpha=1.0 keeps all pixels; alpha=0.0 crops to valid region only).

    Args:
        image_paths   : list of scene image paths
        camera_matrix : 3x3 intrinsic matrix from calibration
        dist_coeffs   : distortion coefficients from calibration
        output_folder : where to save undistorted images

    Returns:
        undistorted_paths    : list of undistorted image file paths
        optimal_camera_matrix: new camera matrix (with updated cx, cy after undistortion)
    """
    print_section("Image Undistortion")
    print(f"  Undistorting {len(image_paths)} images...")

    undist_folder = os.path.join(output_folder, "undistorted")
    os.makedirs(undist_folder, exist_ok=True)

    undistorted_paths   = []
    optimal_camera_matrix = None

    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"  [SKIP] Could not load: {path}")
            continue

        h, w = img.shape[:2]

        # Compute the optimal new camera matrix.
        # alpha=1.0: retain all pixels (black borders may appear at edges)
        # alpha=0.0: crop to only the valid (non-black) region
        opt_mtx, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), alpha=1.0
        )

        if optimal_camera_matrix is None:
            optimal_camera_matrix = opt_mtx

        # Undistort using the calibrated model
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, opt_mtx)

        # Optionally crop to valid ROI (roi = [x, y, w, h])
        # x, y, w_roi, h_roi = roi
        # if all(v > 0 for v in [w_roi, h_roi]):
        #     undistorted = undistorted[y:y+h_roi, x:x+w_roi]

        out_path = os.path.join(undist_folder, f"undistorted_{idx:04d}.jpg")
        cv2.imwrite(out_path, undistorted, [cv2.IMWRITE_JPEG_QUALITY, 95])
        undistorted_paths.append(out_path)

        print(f"  [{idx+1:3d}/{len(image_paths)}] {os.path.basename(path)} → undistorted")

    print(f"\n  {len(undistorted_paths)} images undistorted. Saved to: {undist_folder}")
    return undistorted_paths, optimal_camera_matrix


# ============================================================================ #
# SECTION 5 — IMAGE STITCHING (PANORAMA MODE)                                #
# ============================================================================ #

def stitch_images(image_paths, output_folder, max_images=None):
    """
    Stitch multiple overlapping images into a single panorama using OpenCV.

    HOW IT WORKS (internally, OpenCV handles all of this):
      1. Detect SIFT/ORB keypoints in each image
      2. Match keypoints between adjacent image pairs
      3. Estimate homographies (2D projective transforms) between images
      4. Warp all images to a common cylindrical or spherical projection
      5. Blend seams using multi-band blending

    REQUIREMENTS:
      - Images must overlap 30–50%
      - The scene should have enough texture for feature matching
      - All images taken from approximately the same viewpoint
        (rotating on a tripod = ideal; moving camera = imperfect stitching)

    IMPORTANT LIMITATION:
      - Stitched panoramas lose per-image camera poses
      - We estimate a virtual pinhole camera for the stitched image
      - Depth will be approximate, not metric
      - For metric accuracy: use calibration mode instead

    Returns:
        panorama_path    : path to the saved panorama image
        virtual_intrinsics: dict {fx, fy, cx, cy, width, height} (estimated)
    """
    print_section("Image Stitching (Panorama Mode)")

    if max_images and len(image_paths) > max_images:
        image_paths = image_paths[:max_images]
        print(f"  Using first {max_images} images (MAX_STITCH_IMAGES)")

    print(f"  Stitching {len(image_paths)} images...")

    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
        else:
            print(f"  [SKIP] Could not load: {path}")

    if len(images) < 2:
        raise ValueError("Need at least 2 images for stitching.")

    # Choose stitcher mode
    mode = cv2.Stitcher_PANORAMA if STITCH_MODE == "panorama" else cv2.Stitcher_SCANS
    stitcher = cv2.Stitcher_create(mode)

    print("  Running OpenCV Stitcher... (may take 1–3 minutes)")
    t0     = time.time()
    status, panorama = stitcher.stitch(images)
    elapsed = time.time() - t0

    status_messages = {
        cv2.Stitcher_OK            : "Success",
        cv2.Stitcher_ERR_NEED_MORE_IMGS: "Not enough images",
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed — not enough overlap",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera params adjustment failed",
    }

    if status != cv2.Stitcher_OK:
        raise RuntimeError(
            f"Stitching failed: {status_messages.get(status, 'Unknown error')}\n"
            f"  Tips:\n"
            f"  - Ensure images have 30–50% overlap\n"
            f"  - Images should be taken from the same viewpoint (rotating camera)\n"
            f"  - Use images with good texture (avoid plain walls/sky)\n"
        )

    print(f"  Stitching succeeded in {elapsed:.1f}s")
    print(f"  Panorama size: {panorama.shape[1]} × {panorama.shape[0]} px")

    # Save panorama
    os.makedirs(output_folder, exist_ok=True)
    pano_path = os.path.join(output_folder, "panorama.jpg")
    cv2.imwrite(pano_path, panorama, [cv2.IMWRITE_JPEG_QUALITY, PANORAMA_JPEG_QUALITY])
    print(f"  Saved panorama → {pano_path}")

    # Estimate virtual intrinsics for the panorama.
    # Assumption: field of view ≈ 90° horizontal for a panorama segment,
    # which gives fx = w / (2 * tan(FOV/2)) ≈ w / 2 for 90° FOV.
    # This is a rough estimate; true metric accuracy requires calibration.
    ph, pw = panorama.shape[:2]
    hfov_rad = np.radians(90.0)                    # Assumed horizontal FOV
    fx_est   = pw / (2 * np.tan(hfov_rad / 2))
    fy_est   = fx_est                               # Assume square pixels
    cx_est   = pw / 2.0
    cy_est   = ph / 2.0

    virtual_intrinsics = {
        "fx": fx_est, "fy": fy_est,
        "cx": cx_est, "cy": cy_est,
        "width": pw,  "height": ph,
        "dist_coeffs": np.zeros(5),
    }

    print(f"  Estimated virtual intrinsics:")
    print(f"    fx={fx_est:.1f} fy={fy_est:.1f} cx={cx_est:.1f} cy={cy_est:.1f}")
    print(f"    (Approximate — use calibration mode for metric accuracy)")

    return pano_path, virtual_intrinsics


# ============================================================================ #
# SECTION 6 — DEPTH ESTIMATION (MiDaS — CPU Compatible)                     #
# ============================================================================ #

def load_midas(model_type):
    """
    Load MiDaS depth estimation model.
    Downloads model weights on first run (~100–400 MB from internet cache).
    Returns (model, transform_fn, device).
    """
    if not TORCH_AVAILABLE:
        return None, None, None

    print(f"\n[DEPTH] Loading MiDaS: {model_type}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEPTH] Device: {device}")

    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    midas.to(device).eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform  = (transforms.dpt_transform
                  if model_type in ("DPT_Large", "DPT_Hybrid")
                  else transforms.small_transform)

    print("[DEPTH] Model ready.")
    return midas, transform, device


def estimate_depth_single(image_bgr, midas_model, midas_transform, device):
    """
    Run MiDaS depth estimation on a single OpenCV BGR image.

    Returns normalized depth map [0, 1] where:
      0.0 = very close to camera
      1.0 = very far from camera
    (The inversion of MiDaS's raw disparity output)
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w      = image_rgb.shape[:2]

    input_batch = midas_transform(image_rgb).to(device)

    with torch.no_grad():
        pred = midas_model(input_batch)

    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth = pred.cpu().numpy().astype(np.float32)

    # Invert disparity → depth. MiDaS outputs: high = close.
    # We want: high = far.
    depth = 1.0 / (depth + 1e-8)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    return depth


def visualize_depth_comparison(image_bgr, depth_map, save_path=None):
    """
    Side-by-side visualization of original image and depth map.
    Useful for debugging and understanding the depth quality.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    im = axes[1].imshow(depth_map, cmap="plasma")
    axes[1].set_title("MiDaS Depth Map (plasma colormap)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], label="Relative Depth (0=close, 1=far)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[VIZ]  Depth comparison saved → {save_path}")

    if VISUALIZE_INTERMEDIATE:
        plt.show()

    plt.close()


# ============================================================================ #
# SECTION 7 — POINT CLOUD GENERATION FROM CALIBRATED DEPTH                  #
# ============================================================================ #

def depth_to_pointcloud_calibrated(depth_map, image_bgr, camera_matrix,
                                    dist_coeffs=None,
                                    depth_scale=1.0,
                                    max_depth_norm=0.95):
    """
    Back-project a depth map to a 3D point cloud using EXACT calibrated intrinsics.

    Advantages over Script 1's approach:
      - Camera matrix comes from physical calibration → accurate intrinsics
      - Undistortion has been applied → straight-line geometry preserved
      - Depth back-projection is more accurate

    The pinhole model:
      x_cam = (u - cx) * depth / fx
      y_cam = (v - cy) * depth / fy
      z_cam = depth

    Since we're working with a single image (or panorama), all points are
    returned in camera coordinates (not world coordinates). For multi-image,
    you would need to apply a pose transform per image.

    Args:
        depth_map     : (H, W) float32 normalized [0, 1] depth
        image_bgr     : (H, W, 3) uint8 BGR image
        camera_matrix : 3x3 calibrated intrinsic matrix
        dist_coeffs   : unused here (image already undistorted)
        depth_scale   : scale relative depth to approximate meters (default 1.0)
        max_depth_norm: discard pixels with depth > this threshold (0.95 → 95%)

    Returns:
        points (N, 3) float32 in camera coordinates
        colors (N, 3) float32 RGB [0, 1]
    """
    h, w = depth_map.shape

    # Extract calibrated intrinsics
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # Scale intrinsics if depth map has different resolution
    img_h, img_w = image_bgr.shape[:2]
    sx = w / img_w
    sy = h / img_h
    fx_s = fx * sx
    fy_s = fy * sy
    cx_s = cx * sx
    cy_s = cy * sy

    # Pixel coordinate grids
    u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))

    # Scale depth to metric
    depth_metric = depth_map * depth_scale

    # Filter: remove background (very far) and invalid pixels
    valid = (depth_map > 0.01) & (depth_map < max_depth_norm)

    # Back-projection using calibrated intrinsics
    x_cam = (u[valid] - cx_s) * depth_metric[valid] / fx_s
    y_cam = (v[valid] - cy_s) * depth_metric[valid] / fy_s
    z_cam = depth_metric[valid]

    points = np.stack([x_cam, y_cam, z_cam], axis=-1)

    # Colors (BGR → RGB, normalize)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if (h, w) != (img_h, img_w):
        image_rgb = cv2.resize(image_rgb, (w, h))
    colors = image_rgb[valid].astype(np.float32) / 255.0

    return points.astype(np.float32), colors


def depth_to_pointcloud_assumed(depth_map, image_bgr,
                                 width=None, height=None,
                                 depth_scale=1.0, hfov_deg=60.0):
    """
    Back-project depth to 3D using an ASSUMED camera model (no calibration).

    Used in PIPELINE_MODE = "direct" where we have no calibration data.
    Assumes a pinhole camera with a given horizontal field of view (HFOV).

    HFOV guidance:
      - Smartphone camera : 60–75°
      - Wide-angle lens   : 90–120°
      - Drone camera      : 70–85°
      - Webcam            : 60–70°

    Args:
        depth_map  : (H, W) float32 normalized [0, 1] depth
        image_bgr  : (H, W, 3) uint8 BGR image
        hfov_deg   : horizontal field of view in degrees (your camera's specification)

    Returns:
        points (N, 3) float32
        colors (N, 3) float32
    """
    h, w = image_bgr.shape[:2] if width is None else (height, width)
    depth_h, depth_w = depth_map.shape

    # Construct assumed camera matrix
    fx = depth_w / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
    fy = fx   # Assume square pixels
    cx = depth_w / 2.0
    cy = depth_h / 2.0

    assumed_matrix = build_camera_matrix(fx, fy, cx, cy)

    return depth_to_pointcloud_calibrated(
        depth_map, image_bgr, assumed_matrix,
        depth_scale=depth_scale
    )


def multi_image_to_pointcloud(image_paths, camera_matrix, midas_model,
                               midas_transform, midas_device, max_images=None):
    """
    Run depth estimation on multiple calibrated/undistorted images and
    create one point cloud per image, then merge.

    For multi-image without camera poses, we treat each image independently
    (camera coordinate frame). This gives a "fan" of point clouds that can
    be roughly aligned using ICP (see Script 1 for full ICP pipeline).

    For a single image (panorama mode), only one cloud is generated.

    Returns: merged Open3D PointCloud
    """
    print_section("Multi-Image Depth Estimation + Point Cloud Generation")

    if max_images:
        image_paths = image_paths[:max_images]

    print(f"  Processing {len(image_paths)} images...")

    all_pcds = []
    depth_scale = estimate_single_scale(camera_matrix)

    for idx, path in enumerate(image_paths):
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            print(f"  [{idx+1}] SKIP: could not read {path}")
            continue

        # Depth estimation
        t0    = time.time()
        depth = estimate_depth_single(image_bgr, midas_model, midas_transform, midas_device)
        dt    = time.time() - t0

        # Save depth visualization for the first image
        if idx == 0:
            vis_path = os.path.join(OUTPUT_FOLDER, "depth_visualization.png")
            visualize_depth_comparison(image_bgr, depth, save_path=vis_path)

        # Back-project to 3D
        pts, cols = depth_to_pointcloud_calibrated(
            depth, image_bgr, camera_matrix,
            depth_scale=depth_scale,
        )

        if len(pts) < 50:
            print(f"  [{idx+1:3d}] SKIP: too few valid points")
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)

        # Light per-frame downsample
        pcd = pcd.voxel_down_sample(VOXEL_SIZE * 0.5)

        all_pcds.append(pcd)
        print(f"  [{idx+1:3d}/{len(image_paths)}]  {os.path.basename(path):35s}  "
              f"{len(pcd.points):>8,} pts  (depth: {dt:.1f}s)")

    if not all_pcds:
        raise RuntimeError("No point clouds generated. Check your images.")

    # Merge all clouds (simple concatenation — no pose alignment in this mode)
    print(f"\n  Merging {len(all_pcds)} point clouds...")
    merged = o3d.geometry.PointCloud()
    for pcd in all_pcds:
        merged += pcd

    print(f"  Merged: {len(merged.points):,} points")
    return merged


def estimate_single_scale(camera_matrix):
    """
    Estimate a rough metric depth scale from the camera focal length.

    For a calibrated camera:
      - Depth in pixels × pixel_size_mm / focal_length_mm ≈ metric depth
      - We approximate pixel_size as 0.0034 mm (typical smartphone/mirrorless)
    This gives a rough approximation to metric scale. For exact metric scale
    you need a known reference distance in the scene.
    """
    fx = camera_matrix[0, 0]
    # Approximate: assume sensor width ≈ 36mm full-frame equivalent
    # pixel_pitch ≈ sensor_width / image_width (in mm)
    # Metric scale ≈ focal_length_pixels × pixel_pitch / focal_length_mm
    # Without physical sensor specs, this is a best-effort approximation
    scale = fx / 1000.0   # Very rough: 1000px focal length ≈ 1.0m median depth
    return max(0.1, min(scale, 10.0))  # Clamp for safety


# ============================================================================ #
# SECTION 8 — POINT CLOUD CLEANING                                           #
# ============================================================================ #

def clean_pointcloud(pcd):
    """
    Voxel downsample + Statistical Outlier Removal.

    SOR explanation:
      For each point P:
        1. Find its NB_NEIGHBORS nearest neighbors
        2. Compute the mean distance d_mean to those neighbors
        3. Compute global mean and std of all d_means
        4. If d_mean > global_mean + STD_RATIO × global_std: remove P

    This removes "flying pixels" (depth estimation artifacts at depth
    discontinuities / object edges) and random noise.
    """
    print_section("Point Cloud Cleaning")

    n_raw = len(pcd.points)
    print(f"  Raw points: {n_raw:,}")

    # Voxel downsample
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    print(f"  After voxel downsample ({VOXEL_SIZE}m): {len(pcd.points):,}")

    # SOR
    cl, _ = pcd.remove_statistical_outlier(
        nb_neighbors=SOR_NB_NEIGHBORS,
        std_ratio=SOR_STD_RATIO,
    )
    print(f"  After SOR  (nb={SOR_NB_NEIGHBORS}, σ={SOR_STD_RATIO}):  {len(cl.points):,}  "
          f"({100*len(cl.points)/len(pcd.points):.1f}% retained)")

    return cl


# ============================================================================ #
# SECTION 9 — TESLA-STYLE COLORING + SURFACE MESH                           #
# ============================================================================ #

def apply_coloring(pcd):
    """Apply Tesla-style coloring (see Script 1 Section 10 for full explanation)."""
    points = np.asarray(pcd.points)
    n      = len(points)

    print(f"\n[COLOR] mode='{COLOR_MODE}', cmap='{COLOR_MAP}'")

    if COLOR_MODE == "height":
        z    = points[:, 2]
        z_lo = np.percentile(z, HEIGHT_CLIP_LOW)
        z_hi = np.percentile(z, HEIGHT_CLIP_HIGH)
        norm = np.clip((z - z_lo) / (z_hi - z_lo + 1e-8), 0, 1)
        colors = plt.get_cmap(COLOR_MAP)(norm)[:, :3]

    elif COLOR_MODE == "rgb":
        if pcd.has_colors():
            return pcd
        colors = np.tile([0.65, 0.65, 0.65], (n, 1))

    elif COLOR_MODE == "distance":
        centroid  = points.mean(axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        norm      = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
        colors    = plt.get_cmap(COLOR_MAP)(norm)[:, :3]

    elif COLOR_MODE == "normals":
        if not pcd.has_normals():
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 3, max_nn=30)
            )
        colors = np.abs(np.asarray(pcd.normals))

    else:
        colors = np.tile([0.5, 0.5, 0.5], (n, 1))

    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def reconstruct_mesh(pcd):
    """
    Estimate normals + Poisson surface reconstruction.
    Transfer colors from point cloud to mesh vertices using nearest-neighbor.
    """
    print_section("Poisson Surface Mesh Reconstruction")

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 3, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(100)

    print(f"  Poisson depth={POISSON_DEPTH}  ({len(pcd.points):,} input points)")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH, linear_fit=False
    )

    if MESH_DENSITY_QUANTILE > 0:
        dens = np.asarray(densities)
        mask = dens < np.quantile(dens, MESH_DENSITY_QUANTILE)
        mesh.remove_vertices_by_mask(mask)
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

    print(f"  Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    # Color transfer
    pcd_pts  = np.asarray(pcd.points)
    pcd_cols = np.asarray(pcd.colors)
    mesh_vts = np.asarray(mesh.vertices)
    tree     = cKDTree(pcd_pts)
    _, idx   = tree.query(mesh_vts, k=1, workers=-1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(pcd_cols[idx])
    mesh.compute_vertex_normals()

    return mesh


# ============================================================================ #
# SECTION 10 — EXPORT                                                        #
# ============================================================================ #

def export_results(pcd, mesh, base_name):
    """Export point cloud and mesh in configured formats."""
    print_section("Exporting Results")

    if EXPORT_PLY:
        ply_path = base_name + "_pointcloud.ply"
        o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False)
        print(f"[EXPORT] PLY  → {ply_path}")

    if EXPORT_GLB:
        _export_glb(mesh, base_name + "_surface.glb")

    if EXPORT_OBJ:
        obj_path = base_name + "_surface.obj"
        o3d.io.write_triangle_mesh(obj_path, mesh, write_ascii=True)
        print(f"[EXPORT] OBJ  → {obj_path}")


def _export_glb(mesh, filepath):
    if not TRIMESH_AVAILABLE:
        print("[EXPORT] trimesh not available. Install: pip install trimesh")
        return

    import trimesh

    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    c = np.asarray(mesh.vertex_colors)

    rgba = np.ones((len(v), 4), dtype=np.uint8) * 255
    rgba[:, :3] = (np.clip(c, 0, 1) * 255).astype(np.uint8)

    tm = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=rgba, process=False)
    tm.fix_normals()
    tm.export(filepath)

    size_mb = os.path.getsize(filepath) / 1e6
    print(f"[EXPORT] GLB  → {filepath}  ({size_mb:.1f} MB)")
    print(f"         View: https://gltf-viewer.donmccurdy.com")


# ============================================================================ #
# SECTION 11 — MAIN PIPELINE                                                 #
# ============================================================================ #

def main():
    t_start = time.time()

    print("=" * 70)
    print("  Camera Calibration + Depth Estimation 3D Reconstruction")
    print(f"  Pipeline mode: {PIPELINE_MODE}")
    print(f"  Application  : {APPLICATION_MODE}")
    print("=" * 70)

    setup_directories()

    # ── Load MiDaS ────────────────────────────────────────────────────────────
    midas_model, midas_transform, midas_device = load_midas(DEPTH_MODEL)

    if midas_model is None:
        raise RuntimeError(
            "MiDaS model could not be loaded. "
            "Install PyTorch: pip install torch torchvision"
        )

    # ── Branch on pipeline mode ───────────────────────────────────────────────
    base = os.path.join(OUTPUT_FOLDER, OUTPUT_NAME)

    if PIPELINE_MODE == "calibrate":
        # ── Mode A: Camera Calibration ─────────────────────────────────────
        print_section("Mode A: Camera Calibration Pipeline")

        # Try to load existing calibration first (avoids re-calibrating)
        cal_file = os.path.join(OUTPUT_FOLDER, CALIBRATION_SAVE_FILE)
        cam_mtx, dist_coeffs, img_size = load_calibration(cal_file)

        if cam_mtx is None:
            print("[CAL] No saved calibration found. Running calibration...")
            cam_mtx, dist_coeffs, img_size = calibrate_camera(CALIBRATION_IMAGE_FOLDER)
            save_calibration(cam_mtx, dist_coeffs, img_size, cal_file)
        else:
            print("[CAL] Using loaded calibration. Delete .npz to recalibrate.")

        # Load and undistort scene images
        scene_paths = load_image_paths(SCENE_IMAGE_FOLDER)
        print(f"\n[SCENE] Found {len(scene_paths)} scene images.")

        if not scene_paths:
            raise FileNotFoundError(f"No scene images in '{SCENE_IMAGE_FOLDER}'")

        undist_paths, opt_cam_mtx = undistort_images(
            scene_paths, cam_mtx, dist_coeffs, OUTPUT_FOLDER
        )

        # Use the optimal (post-undistortion) camera matrix
        effective_matrix = opt_cam_mtx if opt_cam_mtx is not None else cam_mtx

        # Generate point cloud from undistorted images
        merged_pcd = multi_image_to_pointcloud(
            undist_paths, effective_matrix,
            midas_model, midas_transform, midas_device,
            max_images=MAX_DEPTH_IMAGES,
        )

    elif PIPELINE_MODE == "stitch":
        # ── Mode B: Image Stitching ────────────────────────────────────────
        print_section("Mode B: Image Stitching Pipeline")

        scene_paths = load_image_paths(SCENE_IMAGE_FOLDER)
        if not scene_paths:
            raise FileNotFoundError(f"No scene images in '{SCENE_IMAGE_FOLDER}'")

        pano_path, virtual_intr = stitch_images(
            scene_paths, OUTPUT_FOLDER, max_images=MAX_STITCH_IMAGES
        )

        # Run depth on the panorama
        pano_img = cv2.imread(pano_path)
        print_section("Depth Estimation on Panorama")

        t0    = time.time()
        depth = estimate_depth_single(pano_img, midas_model, midas_transform, midas_device)
        print(f"  Depth estimated in {time.time()-t0:.1f}s")

        vis_path = os.path.join(OUTPUT_FOLDER, "panorama_depth_vis.png")
        visualize_depth_comparison(pano_img, depth, save_path=vis_path)

        # Build camera matrix from virtual intrinsics
        cam_mtx = build_camera_matrix(
            virtual_intr["fx"], virtual_intr["fy"],
            virtual_intr["cx"], virtual_intr["cy"],
        )

        # Point cloud from panorama
        depth_scale = virtual_intr["fx"] / 1000.0
        pts, cols   = depth_to_pointcloud_calibrated(
            depth, pano_img, cam_mtx, depth_scale=depth_scale
        )

        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(pts)
        merged_pcd.colors = o3d.utility.Vector3dVector(cols)
        print(f"  Panorama point cloud: {len(merged_pcd.points):,} points")

    elif PIPELINE_MODE == "direct":
        # ── Mode C: No Calibration / Direct ───────────────────────────────
        print_section("Mode C: Direct Depth (assumed camera model)")
        print("  No calibration or stitching. Using assumed pinhole camera.")
        print("  Set HFOV_DEG to match your camera's horizontal field of view.")

        HFOV_DEG = 65.0   # Adjust for your specific camera

        scene_paths = load_image_paths(SCENE_IMAGE_FOLDER)
        if not scene_paths:
            raise FileNotFoundError(f"No scene images in '{SCENE_IMAGE_FOLDER}'")

        all_pcds = []
        for idx, path in enumerate(scene_paths[:MAX_DEPTH_IMAGES or len(scene_paths)]):
            img   = cv2.imread(path)
            depth = estimate_depth_single(img, midas_model, midas_transform, midas_device)
            pts, cols = depth_to_pointcloud_assumed(depth, img, hfov_deg=HFOV_DEG)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols)
            all_pcds.append(pcd)
            print(f"  [{idx+1}] {os.path.basename(path)}: {len(pcd.points):,} points")

        merged_pcd = o3d.geometry.PointCloud()
        for p in all_pcds:
            merged_pcd += p

    else:
        raise ValueError(f"Unknown PIPELINE_MODE: '{PIPELINE_MODE}'. "
                         f"Choose 'calibrate', 'stitch', or 'direct'.")

    # ── Shared post-processing ─────────────────────────────────────────────────

    # Clean
    clean_pcd = clean_pointcloud(merged_pcd)

    if VISUALIZE_INTERMEDIATE:
        safe_visualize([clean_pcd], "Cleaned Point Cloud")

    # Color
    apply_coloring(clean_pcd)

    if VISUALIZE_INTERMEDIATE:
        safe_visualize([clean_pcd], f"Tesla-Style Coloring ({COLOR_MAP})")

    # Mesh
    surface_mesh = reconstruct_mesh(copy.deepcopy(clean_pcd))

    if VISUALIZE_FINAL:
        safe_visualize([surface_mesh], "Reconstructed Surface Mesh")

    # Export
    export_results(clean_pcd, surface_mesh, base)

    # ── Summary ────────────────────────────────────────────────────────────────
    t_total = time.time() - t_start
    print_section("Complete")
    print(f"  Mode          : {PIPELINE_MODE}")
    print(f"  Total time    : {t_total/60:.1f} min")
    print(f"  Points        : {len(clean_pcd.points):,}")
    print(f"  Triangles     : {len(surface_mesh.triangles):,}")
    print(f"  Output folder : {os.path.abspath(OUTPUT_FOLDER)}")
    print()
    print("  View GLB at: https://gltf-viewer.donmccurdy.com")


if __name__ == "__main__":
    main()
