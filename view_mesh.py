import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import trimesh
import numpy as np

# Launch the visualizer
vis = meshcat.Visualizer().open()

# Load the mesh
mesh = trimesh.load("/home/yufeiyang/Documents/BundleSDF/your_mesh_centered.obj")  # Replace with your path

# Diagnostic 1: Check mesh content
print("Vertices:", mesh.vertices.shape)
print("Faces:", mesh.faces.shape)

# If it's a Scene instead of a Trimesh, get the geometry
if isinstance(mesh, trimesh.Scene):
    print("Loaded a Scene. Converting to Trimesh...")
    mesh = mesh.dump().sum()

# Normalize mesh size and center
mesh.apply_translation(-mesh.centroid)
mesh.apply_scale(1.0 / mesh.scale)

# Diagnostic 2: Print centroid and scale
print("Centroid:", mesh.centroid)
print("Scale:", mesh.scale)

# Convert mesh to Meshcat format
meshcat_geom = g.TriangularMeshGeometry(
    vertices=mesh.vertices.astype(np.float32),
    faces=mesh.faces.astype(np.uint32)
)

# Add mesh to scene
material = g.MeshLambertMaterial(color=0x00ff00)
vis["model"].set_object(meshcat_geom, material)

# Optional: adjust view
vis["model"].set_transform(tf.translation_matrix([0, 0, 0]))
