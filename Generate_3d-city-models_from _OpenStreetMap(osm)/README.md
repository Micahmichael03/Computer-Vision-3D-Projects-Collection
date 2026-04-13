# 🏙️ OSM 3D City Model Generator

> Turn any city on Earth into a fully textured, animated 3D model — powered entirely by open data.

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![OSMnx](https://img.shields.io/badge/OSMnx-1.x-green)](https://osmnx.readthedocs.io)
[![PyVista](https://img.shields.io/badge/PyVista-0.43+-orange)](https://pyvista.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What This Does

This pipeline pulls real building footprints and street networks from OpenStreetMap, extrudes them into watertight 3D meshes, and exports them as `.obj` files or animated `.gif` flyovers — all without touching a paid dataset.

**One script. Any city. Full 3D output.**

```
OSM API → Building Footprints → 3D Mesh (watertight) → .OBJ / Animated GIF
                     └── Street Network → PyVista Polylines
```

---

## Pipeline Architecture

```
extract_osm_data()
  ├── ox.geocoder.geocode()         # Resolve city name → lat/lon
  ├── ox.features_from_point()      # Fetch building footprints (tags: building=True)
  ├── ox.graph_from_point()         # Fetch driveable street graph
  └── .to_crs(epsg=2154)            # Reproject to Lambert-93 for metric accuracy

generate_footprints()
  └── Unpack MultiPolygon → flat list of Shapely Polygons

create_watertight_building()
  ├── Base face  (polygon fan of ground ring)
  ├── Roof face  (polygon fan of top ring)
  └── Wall faces (quad per edge pair: base[i] → base[i+1] → top[i+1] → top[i])

extrude_buildings()
  ├── Random height: uniform(10m, 50m) per building
  ├── Per-building random RGB color → stored as point scalar
  └── pv.PolyData.merge() → single unified city mesh

streetGraph_to_pyvista()
  └── Edge geometries → PyVista polylines (Z = 0)

cloudgify()
  └── Orbital camera path (40 frames) → .gif export

automate_pipeline()
  └── Iterates across N cities with flags: exporting | visualizing | gification
```

---

## Requirements

```bash
pip install osmnx pyvista shapely numpy
```

| Library   | Purpose                                      | Minimum Version |
|-----------|----------------------------------------------|-----------------|
| `osmnx`   | OSM data download, geocoding, graph handling | 1.0.0           |
| `pyvista` | 3D mesh creation, rendering, GIF export      | 0.43.0          |
| `shapely` | Polygon and MultiPolygon geometry handling   | 2.0.0           |
| `numpy`   | Array ops for coordinates and face arrays    | 1.21.0          |

> **Note:** PyVista requires a display backend for interactive rendering. On headless servers (CI, Docker), rendering falls back to `off_screen=True` automatically in `cloudgify()`.

---

## Installation

```bash
git clone https://github.com/Micahmichael03/osm-3d-city-model.git
cd osm-3d-city-model
pip install -r requirements.txt
```

---

## Usage

### Quick Start — Single City (Interactive Viewer)

```python
from citymodel_simple import extract_osm_data, generate_footprints, extrude_buildings
import pyvista as pv

location = "Aachen, Germany"
radius = 500  # meters

buildings, streets = extract_osm_data(location, radius)
footprints = generate_footprints(buildings)
mesh, _ = extrude_buildings(footprints)

pl = pv.Plotter()
pl.add_mesh(mesh, scalars=mesh['color'], cmap="tab20")
pl.show()
```

### Export as .OBJ

```python
from citymodel_simple import single_location_experiment
single_location_experiment()
# Outputs: output/Aachen/buildings.obj + output/Aachen/streets.obj
```

### Animated GIF (Orbital Flyover)

```python
from citymodel_simple import automate_pipeline
# Set gification = True inside automate_pipeline(), then:
automate_pipeline()
# Outputs: output//model.gif per location
```

### Batch — Multiple Cities

```python
# Inside automate_pipeline(), configure:
locations = [
    "Toulouse, France",
    "Calgary, Canada",
    "Aachen, Germany",
    "Enschede, Netherlands",
    "Angleur, Belgium",
]
exporting    = True   # Save .obj files
visualizing  = False  # Skip interactive window
gification   = True   # Generate animated GIF
```

---

## Output Structure

```
output/
├── Toulouse/
│   ├── buildings.obj
│   ├── streets.obj
│   └── model.gif
├── Calgary/
│   └── ...
└── Aachen/
    └── ...
```

---

## Design Decisions & Trade-offs

| Decision | Why |
|----------|-----|
| EPSG:2154 (Lambert-93) projection | Converts geographic degrees to meters for accurate distance/extrusion math |
| Random building heights (10–50m) | No height attribute exists in base OSM tags; LiDAR integration is the next step |
| Watertight mesh (base + walls + roof) | Required for valid `.obj` export and clean rendering in downstream 3D tools |
| `merge_points=False` on city merge | Prevents shared vertices between buildings from corrupting face winding |
| Per-point color scalars | Enables `cmap="tab20"` colorization per building without separate material files |

---

## Known Limitations & Next Steps

- [ ] Replace random heights with real LiDAR or OSM `height` tag data where available
- [ ] Add texture mapping (roof/wall materials per building type)
- [ ] Export to `.glb` / `.gltf` for web-based 3D viewers
- [ ] Integrate with a 3D world engine (e.g., CesiumJS, Unity)
- [ ] Add `Path` import to fix `save_to_obj()` — currently requires `from pathlib import Path`

---

## Author

Built by **Michael Chukwuemeka Micah**

- GitHub: [@Micahmichael03](https://github.com/Micahmichael03)
- LinkedIn: [michael-micah003](https://linkedin.com/in/michael-micah003)
- Email: makoflash05@gmail.com
- Substack: [@michaelchukwuemekamicah](https://substack.com/@michaelchukwuemekamicah)

---

*Inspired by [Florent Poux – 3D Tech](https://learngeodata.eu). Data © OpenStreetMap contributors.*