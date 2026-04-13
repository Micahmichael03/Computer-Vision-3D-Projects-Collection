#Base libraries
import numpy as np

#3D library
import open3d as o3d

#%% 2. Data Loading (PLY, OBJ, ASCII)

# Loading your point cloud
pcd = o3d.io.read_point_cloud('../DATA/Naavis_EXTERIOR.ply')

#%% Preprocessing operations

# Convert Open3d Object to Numpy for processing stages
input_points = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)
colors = np.asarray(pcd.colors)

# Centering Simple Translation
center = pcd.get_center()
pcd.translate(-center)

#%% 3D Visualisation and Exploration
o3d.visualization.draw_geometries([pcd])

#%% Execute Processing
import time
import region_growing_sample as rg

t0 = time.time()
segments_n = rg.rg_normals(input_points, normals, 0.1, np.pi/6)
t1 = time.time()

labels , colors_1 = rg.coloring_segments(input_points, segments_n)

print(f"Region Growing Successful in: {round(t1-t0,3)} seconds")

#%% Visualization
pcd.colors = o3d.utility.Vector3dVector(colors_1)
o3d.visualization.draw_geometries([pcd])

#%% Export

# If only the Geometry + Normals + Color is relevant
o3d.io.write_point_cloud("../RESULTS/Naavis_EXTERIOR_segmented.ply", pcd)

# If you need to add features
pcd_segmented = np.hstack((input_points, np.atleast_2d(labels).T))
np.savetxt("../RESULTS/Naavis_EXTERIOR_segmented.xyz", pcd_segmented, fmt='%1.6f', delimiter=';', header='X;Y;Z;Segment')