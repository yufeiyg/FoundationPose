import numpy as np
import trimesh
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import os
import time
import glob

code_dir = os.path.dirname(os.path.realpath(__file__))

# === 1. Load the OBJ mesh ===
mesh_path = f'/home/yufeiyang/Documents/BundleSDF/D_shape.obj'
# mesh_path = "/home/yufeiyang/Documents/FoundationPose/demo_data/push_t_data/mesh/push_t_bicolor.obj"
scene_or_mesh = trimesh.load(mesh_path)

# Handle both Trimesh and Scene types
if isinstance(scene_or_mesh, trimesh.Scene):
    # Merge all geometries in the scene
    trimesh_mesh = trimesh.util.concatenate(
        [geom for geom in scene_or_mesh.geometry.values()]
    )
else:
    trimesh_mesh = scene_or_mesh

# === 2. Create a MeshCat visualizer ===
vis = meshcat.Visualizer().open()
vis.delete()  # Clear the scene

# Create a MeshCat mesh object from Trimesh geometry
vertices = trimesh_mesh.vertices.astype(np.float32)
faces = trimesh_mesh.faces.astype(np.uint32)
meshcat_mesh = g.TriangularMeshGeometry(vertices, faces)

# Set the object in MeshCat
vis["object"].set_object(meshcat_mesh, g.MeshLambertMaterial(color=0x00FF00))
vis["object"].set_transform(np.eye(4))

# === 3. Load ob_in_cam poses ===
pose_folder = "/home/yufeiyang/Documents/BundleSDF/foundationPose/D_shape/ob_in_cam"
import re
def numerical_sort(value):
    # Extract the first number found in the filename
    match = re.search(r'(\d+)', os.path.basename(value))
    return int(match.group(1)) if match else -1

pose_files = sorted(glob.glob(os.path.join(pose_folder, "*.txt")), key=numerical_sort)

def load_matrix_from_txt(path):
    data = np.loadtxt(path)
    if data.size != 16:
        raise ValueError(f"File {path} does not contain a 4x4 matrix.")
    return data.reshape(4, 4)

# Load all poses first
poses = [load_matrix_from_txt(pf) for pf in pose_files]

# === 4. Rebase poses relative to first pose ===
T0 = poses[0]
T0_inv = np.linalg.inv(T0)
adjusted_poses = [T0_inv @ T for T in poses]

# === 5. Animate the mesh using rebased poses ===
for i, T in enumerate(adjusted_poses):
    vis["object"].set_transform(T)
    print(f"Showing frame {i} from file: {pose_files[i]}")
    time.sleep(0.1)  # Adjust playback speed here