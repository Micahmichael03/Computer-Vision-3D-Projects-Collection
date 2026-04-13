#%% this works fine, use it often

import numpy as np 
import pyvista as pv

#%% 1. Load Mesh (with Texture)

mesh = pv.examples.load_globe()
texture = pv.examples.load_globe_texture()

pl = pv.Plotter()
pl.add_mesh(mesh, texture=texture, smooth_shading=True)
pl.show()

#%% 2. Load + visualize a 3D Point cloud

cloud = pv.read(r"C:\Users\user\OneDrive\Documents\Computer Vision\3D_projects\Open3D-Tutorial-main\happy_recon\happy_vrip.ply")
# Rotate the point cloud to make it upright
cloud.rotate_x(80, inplace=True)
scalars = np.linalg.norm(cloud.points - cloud.center, axis=1)

pl = pv.Plotter()
pl.add_mesh(cloud)
pl.show()

#%% 3. Experiment: Adjust the parameters

pl = pv.Plotter(off_screen=False)
pl.add_mesh (
    cloud,
    style='Points',
    render_points_as_spheres=True,
    emissive=False,
    color='#ff7c2',
    scalars=scalars,
    opacity=1,
    point_size=5.0,
    show_scalar_bar=False
    )

pl.add_text('test', color='b')
pl.background_color = 'k'
pl.enable_eye_dome_lighting()
pl.show()

#%% 4. Generate an orbital GIF

pl = pv.Plotter(off_screen=True, image_scale=1)
pl.add_mesh (
    cloud,
    style='Points',
    render_points_as_spheres=True,
    emissive=False,
    color='#ff7c2',
    scalars=scalars,
    opacity=1,
    point_size=5.0,
    show_scalar_bar=False
    )

pl.background_color = 'k'
pl.enable_eye_dome_lighting()
pl.show(auto_close=False)

viewup = [0.2, 0.2, 1]

path = pl.generate_orbital_path(n_points=40, shift=cloud.length,
viewup=viewup, factor=3.0)
pl.open_gif("orbit_cloud.gif")
pl.orbit_on_path(path, write_frames=True, viewup=viewup)
pl.close()

#%% 5. Generate an orbital MP4

pl = pv.Plotter(off_screen=True, image_scale=1)
pl.add_mesh(
    cloud,
    style='points_gaussian',
    render_points_as_spheres=True,
    emissive=False,
    color='#ff7c2',
    scalars=scalars,
    opacity=1,
    point_size=5.0,
    show_scalar_bar=False
)

pl.background_color = 'k'
pl.show(auto_close=False)

viewup = [0.2, 0.2, 1]

path = pl.generate_orbital_path(n_points=40, shift=cloud.length,
                                viewup=viewup, factor=3.0)

# Use 'framerate' instead of 'fps'
pl.open_movie("orbit_cloud.mp4", framerate=24)
pl.orbit_on_path(path, write_frames=True, viewup=viewup)
pl.close()

#%% 6. Create a function

def cloudgify(input_path):
    cloud = pv.read(input_path)
    # Rotate the point cloud to make it upright
    cloud.rotate_x(-90, inplace=True)
    scalars = np.linalg.norm(cloud.points - cloud.center, axis=1)
    pl = pv.Plotter(off_screen=True, image_scale=1)
    pl.add_mesh(
        cloud,
        style='points',
        render_points_as_spheres=True,
        emissive=False,
        color='#ff7c2',
        scalars=scalars,
        opacity=1,
        point_size=5.0,
        show_scalar_bar=False
    )

    pl.background_color = 'k'
    pl.enable_eye_dome_lighting()
    pl.show(auto_close=False)

    viewup = [0, 0, 1]
    
    path = pl.generate_orbital_path(n_points=40, shift=cloud.length,
viewup=viewup, factor=3.0)
    
    pl.open_gif(input_path.split('.')[0]+'.gif')
    pl.orbit_on_path(path, write_frames=True, viewup=viewup)
    pl.close()
    
    path = pl.generate_orbital_path(n_points=100, shift=cloud.length,
viewup=viewup, factor=3.0)
    
    pl.open_movie(input_path.split('.')[0]+'.mp4')
    pl.orbit_on_path(path, write_frames=True, viewup=viewup)
    pl.close()
    
    return

#%% 7. Multi-dataset testtest with other datasets

dataset_paths= [" lots/of/datasets/in .ply"]

for pcd in dataset_paths:
    cloudgify(pcd)