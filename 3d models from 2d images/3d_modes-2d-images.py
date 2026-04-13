#%% 1. libraries import
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline') # 'TkAgg' 
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation

#%% 2. Getting model

feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

#%% 3. Loading and resizing the image

image = Image.open(r"C:\Users\user\Downloads\1774184978714.jpeg")
new_height = 480 if image.height > 480 else image.height
new_height -= (new_height % 32)
new_width = int(new_height * image.width / image.height)
diff = new_width % 32

new_width = new_width - diff if diff < 16 else new_width + 32 - diff
new_size = (new_width, new_height)
image = image.resize(new_size)

#%% 4. Preparing the image for the model

inputs = feature_extractor(images=image, return_tensors="pt")

#%% 5. Getting the predication from the model

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth
    
#%% 6. Post-processing

pad = 16
output = predicted_depth.squeeze().cpu().numpy() * 1000.0
output = output[pad:-pad, pad:-pad]
image = image.crop((pad, pad, image.width - pad, image.height -pad))

# visualize the prediction
fig, ax = plt.subplots(1, 2)

# Left image
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# Right image (Depth Map)
ax[1].imshow(output, cmap='plasma')
ax[1].set_title("Predicted Depth")
ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

plt.tight_layout()
plt.show()

#%% 7. Importing the libraries 

import numpy as np
import open3d as o3d

#%% 8. Preparing the depth image for open3d

# 1. Convert color to numpy (Open3D needs uint8 for color)
color_np = np.array(image).astype(np.uint8)

# 2. Keep Depth as Float (Crucial for sharpness!)
# We divide by 1000 to get meters. 
depth_np = (output / 1000.0).astype(np.float32)

# 3. Create Open3D Images
color_o3d = o3d.geometry.Image(color_np)
depth_o3d = o3d.geometry.Image(depth_np)

# 4. Create RGBD - depth_trunc=5.0 ignores far-away background noise
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d, 
    depth_o3d, 
    depth_scale=1.0, 
    depth_trunc=5.0, 
    convert_rgb_to_intensity=False
)


#%% 9. Creating a Camera

# Make sure we have the dimensions from the current image
# This prevents the NameError if you run this cell independently
width, height = image.size if hasattr(image, 'size') else (image.shape[1], image.shape[0])

camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
# Standard focal length (500) and principal point (center of image)
camera_intrinsic.set_intrinsics(width, height, 500.0, 500.0, width/2.0, height/2.0)


#%% 10. Creating o3d point cloud

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

# Flip it so it's not upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# --- CLEANING THE NOISE ---
# cl is the cleaned cloud, ind is the index of points kept
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd = cl 

o3d.visualization.draw_geometries([pcd])

#%% 11. Exporting the Result

# Save as .ply (Standard 3D format for point clouds)
# This will save the file in your project folder
file_name = "Chris_Point_Cloud.ply"

# Save the 'pcd' variable that you just cleaned in Cell 10
o3d.io.write_point_cloud(file_name, pcd)

print(f"Successfully exported: {file_name}")

#o3d.io.write_triangle_mesh('../Results/Teasla.ply', mesh)

# %%
