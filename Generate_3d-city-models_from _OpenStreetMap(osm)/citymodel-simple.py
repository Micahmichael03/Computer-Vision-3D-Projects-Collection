# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 00:13:15 

@author: user
"""

#%% 1. Environment + library setup

import osmnx as ox
import numpy as np
import pyvista as pv
from shapely.geometry import Polygon, MultiPolygon
import random

#%%  2. Extract OSM data

def extract_osm_data(location, radius=500):
    """
    Extract building footprints and street data from OpenStreetMap.
    """
    print(f"Downloading OSM data for {location}...")

    # Get the latitude and longitude for the location
    center_point = ox.geocoder.geocode(location)

    # Download building footprints using features_from_point
    buildings = ox.features_from_point(
        center_point,
        tags={'building': True},
        dist=radius
    )

    # Download street footprints using graph_from_point
    streets = ox.graph_from_point(
        center_point,
        dist=radius,
        network_type='drive',
        simplify=False
    )

    # Convert to GeoDataFrame with projection
    buildings = buildings.to_crs(epsg=2154)
    streets = ox.project_graph(streets, to_crs='epsg:2154')

    print(f"Downloaded {len(buildings)} buildings")
    print(f"Downloaded {len(streets.edges)} street segments")

    return buildings, streets



#%% 3. Generate building footprints

def generate_footprints(buildings):
    """
    Convert GeoDataFrame geometries to Shapely polygons.
    """
    footprints = []

    for geom in buildings.geometry:
        if isinstance(geom, MultiPolygon):
            # Split multipolygons into individual polygons
            footprints.extend(list(geom.geoms))
        elif isinstance(geom, Polygon):
            # Add single polygons
            footprints.append(geom)
        # Skip points and other geometry types

    return footprints


#%% 4. Generate 3D Buildings

def create_watertight_building(coords, height):
    """
    Create a watertight building mesh with base, walls, and roof.
    """
    # Remove duplicate last point from coords
    coords = coords[:-1]
    n_points = len(coords)

    # Create points for base and top
    base_points = np.column_stack((coords, np.zeros(len(coords))))
    top_points = np.column_stack((coords, np.full(len(coords), height)))

    # Combine all points
    points = np.vstack((base_points, top_points))

    # Create faces
    faces = []

    # Add base (as triangle fan)
    base_face = [n_points] + list(range(n_points))
    faces.extend(base_face)

    # Add roof (as triangle fan)
    roof_indices = list(range(n_points, n_points * 2))
    roof_face = [n_points] + roof_indices
    faces.extend(roof_face)

    # Add walls (as quads)
    for i in range(n_points):
        next_i = (i + 1) % n_points
        wall_face = [
            4,  # quad
            i, next_i,  # bottom points
            n_points + next_i, n_points + i  # top points
        ]
        faces.extend(wall_face)

    return points, faces


# ------------------------------------------------------------
# Utility: Generate random color
# ------------------------------------------------------------
def generate_random_color():
    """
    Generate a random RGB color.
    """
    return [random.random() for _ in range(3)]


# ------------------------------------------------------------
# Extrude buildings into 3D visualization
# ------------------------------------------------------------
def extrude_buildings(footprints):
    """
    Extrude 2D footprints into 3D buildings using PyVista.
    """
    print("Extruding buildings...")

    # Create empty PyVista mesh for the final city
    city_mesh = pv.PolyData()
    instances_building = []

    for footprint in footprints:
        # Get coordinates from footprint
        coords = np.array(footprint.exterior.coords)

        # Generate random building height (between 10 and 50 meters)
        height = random.uniform(10, 50)

        # Create watertight building geometry
        points, faces = create_watertight_building(coords, height)

        # Create building mesh
        building = pv.PolyData(points, np.array(faces))

        # Generate and apply random color
        color = generate_random_color()
        building['color'] = np.tile(color, (building.n_points, 1))
        instances_building.append(building)

        # Add to city mesh
        if city_mesh.n_points == 0:
            city_mesh = building
        else:
            city_mesh = city_mesh.merge(building, merge_points=False)

    print("Building extrusion complete")
    return city_mesh, instances_building

#%% 5. 3D Visualization: Test #1

location = "New York, USA"
radius = 500

buildings, streets = extract_osm_data(location, radius)
footprints = generate_footprints(buildings)
mesh, bd_instances = extrude_buildings(footprints)

pl = pv.Plotter(border=False)
pl.add_mesh(mesh, scalars=mesh['color'], cmap="tab20", show_edges=False)
pl.remove_scalar_bar()
pl.show(title='(c) Florent Poux - 3D Tech')

# 3D Professionals: Education Materials + Courses at 3D Geodata Academy.

#%% 6. Building Export Function: The Building

def save_to_obj(mesh, output_path):
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {output_path}")
    mesh.save(output_path)
    print("Export complete")


def streetGraph_to_pyvista(st_graph):
    nodes, edges = ox.graph_to_gdfs(st_graph)
    pts_list = edges['geometry'].apply(
        lambda g: np.column_stack((g.xy[0], g.xy[1], np.zeros(len(g.xy[0]))))
    ).tolist()
    vertices = np.concatenate(pts_list)

    lines = []
    j = 0
    for pts in pts_list:
        vertex_length = len(pts)
        vertex_start = j
        vertex_end = j + vertex_length - 1
        vertex_arr = [vertex_length] + list(range(vertex_start, vertex_end + 1))
        lines.append(vertex_arr)
        j += vertex_length

    return pv.PolyData(vertices, lines=np.hstack(lines))

#%% 7. Cloudify GIF Export

def cloudgify(location, mesh, street_mesh, file_path):
    pl = pv.Plotter(off_screen=True, image_scale=1)
    pl.background_color = 'k'
    pl.add_text(location, position='upper_left', color='Lightgrey',
                shadow=True, font_size=26)
    pl.add_mesh(mesh, scalars=mesh['color'], cmap="tab20", show_edges=False)
    pl.add_mesh(street_mesh)
    pl.remove_scalar_bar()
    pl.show(auto_close=False)

    viewup = [0, 0, 1]
    output_dir = Path(file_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = pl.generate_orbital_path(n_points=40, shift=mesh.length,
                                    viewup=viewup, factor=3.0)
    pl.open_gif(output_dir/"model.gif")
    pl.orbit_on_path(path, write_frames=True, viewup=viewup)
    pl.close()
    print("Export of GIF successful")
    return

#%% 8. Single Location Experiment

def single_location_experiment():
    location = "Aachen, Germany"
    radius = 500

    buildings, streets = extract_osm_data(location, radius)
    footprints = generate_footprints(buildings)
    mesh, bd_instances = extrude_buildings(footprints)
    street_mesh = streetGraph_to_pyvista(streets)

    output_dir = "output/" + location.split(",")[0]
    cloudgify(location, mesh, street_mesh, output_dir)

    save_to_obj(mesh, output_dir + "/buildings.obj")
    save_to_obj(street_mesh, output_dir + "/streets.obj")

#%% 9. Script Automation (100%)

def automate_pipeline():
    exporting = False
    visualizing = False
    gification = True

    locations = [
        "Toulouse, France",
        "Calgary, Canada",
        "Aachen, Germany",
        "Enschede, Netherlands",
        "Angleur, Belgium",
    ]
    radius = 500

    for location in locations:
        buildings, streets = extract_osm_data(location, radius)
        footprints = generate_footprints(buildings)
        mesh, bd_instances = extrude_buildings(footprints)
        street_mesh = streetGraph_to_pyvista(streets)

        if exporting:
            output_dir = "output/" + location.split(",")[0]
            save_to_obj(mesh, output_dir + "/buildings.obj")
            save_to_obj(street_mesh, output_dir + "/streets.obj")

        if visualizing:
            pl = pv.Plotter(border=False)
            pl.background_color = 'lightgrey'
            pl.add_text(location, position='upper_left', color='black',
                        shadow=True, font_size=26)
            pl.add_mesh(mesh, scalars=mesh['color'], cmap='tab20', show_edges=False)
            pl.add_mesh(street_mesh)
            pl.remove_scalar_bar()
            pl.show(title='(c) Florent Poux - 3D Tech')

        if gification:
            output_dir = "output/" + location.split(",")[0]
            cloudgify(location, mesh, street_mesh, output_dir)

#%% 10. Going Beyond

print(buildings.columns.tolist())
    # 2. take the height from other datasets (e.g., LiDAR)
    # 3. integrate in a 3D World
