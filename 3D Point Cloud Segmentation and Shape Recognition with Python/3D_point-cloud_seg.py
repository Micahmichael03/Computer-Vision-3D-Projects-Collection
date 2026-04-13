
#%% 1. Environment setup
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


#%% 2. Point Cloud Data Preparation
DATANAME = o3d.data.PLYPointCloud() 
pcd = o3d.io.read_point_cloud(DATANAME.path)


#%%  3. Data Pre-Processing
pcd_center = pcd.get_center()
pcd.translate(-pcd_center)


#%%  3.1. Statistical outlier filter
nn = 16
std_multiplier = 10

filtered_pcd = pcd.remove_statistical_outlier(nn, std_multiplier)

outliers = pcd.select_by_index(filtered_pcd[1], invert=True)
outliers.paint_uniform_color([1, 0, 0])
filtered_pcd = filtered_pcd[0]

o3d.visualization.draw_geometries([filtered_pcd, outliers])


#%%  3.2. Voxel downsampling
voxel_size = 0.01

pcd_downsampled = filtered_pcd.voxel_down_sample(voxel_size=voxel_size)
o3d.visualization.draw_geometries([pcd_downsampled])


#%%  3.3. Estimating normals
nn_distance = np.mean(pcd.compute_nearest_neighbor_distance())

radius_normals = nn_distance * 4

pcd_downsampled.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_normals,
        max_nn=16
    ),
    fast_normal_computation=True
)

pcd_downsampled.paint_uniform_color([0.6, 0.6, 0.6])
o3d.visualization.draw_geometries([pcd_downsampled, outliers])


#%% 4. Extracting and Setting Parameters (usage)
front = [0.9681980579027637, 0.24767928031788963, -0.0353198205149242]
lookat = [0.0084975577023191553, -0.2087872942101065, -0.48907046814855626]
up = [0.011867566069102271, 0.095549553792428299, 0.99535392890581653]
zoom = 0.23999999999999969

pcd = pcd_downsampled
o3d.visualization.draw_geometries(
    [pcd],
    zoom=zoom,
    front=front,
    lookat=lookat,
    up=up
)


#%%  5. RANSAC Planar Segmentation
pt_to_plane_dist = 0.02

plane_model, inliers = pcd.segment_plane(
    distance_threshold=pt_to_plane_dist,
    ransac_n=3,
    num_iterations=1000
)

[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)

inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])

o3d.visualization.draw_geometries(
    [inlier_cloud, outlier_cloud],
    zoom=zoom,
    front=front,
    lookat=lookat,
    up=up
)

#%%  6. Multi-order RANSAC
max_plane_idx = 6
pt_to_plane_dist = 0.02

segment_models = {}
segments = {}
rest = pcd

for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    
    # FIXED: num_iterations must be one word and on the same line or properly escaped
    segment_models[i], inliers = rest.segment_plane(
        distance_threshold=pt_to_plane_dist, 
        ransac_n=3, 
        num_iterations=1000
    )
    
    segments[i] = rest.select_by_index(inliers)
    segments[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True)
    print("pass", i + 1, "/", max_plane_idx, "done.")

o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+
[rest],zoom=zoom, front=front, lookat=lookat,up=up)


#%%  7. DBSCAN sur rest

labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=5))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
rest.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+
[rest],zoom=zoom, front=front, lookat=lookat,up=up)

# Save as .ply (Standard 3D format for point clouds)
# This will save the file in your project folder
file_name = "_Point_Cloud.ply"

# Save the 'pcd' variable that you just cleaned in Cell 10
o3d.io.write_point_cloud(file_name, pcd)

print(f"Successfully exported: {file_name}")

# %%
