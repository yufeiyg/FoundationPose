import trimesh
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import triangulate
# Load the mesh from the specified file path
path = "/home/yufeiyang/Documents/FoundationPose/output/ob_000000k/model/model.obj"
mesh = trimesh.load(path)
mesh.fill_holes()

components = mesh.split(only_watertight=False)
mesh = max(components, key=lambda m: len(m.faces))  # Or use `m.volume` if watertight
print("cleaned mesh")

face_normals = mesh.face_normals

direction = np.array([0, 0, -1])
threshold = 0.8  # cosine similarity threshold
dot_products = face_normals @ direction
facing_indices = np.where(dot_products > threshold)[0]  # faces "facing" X

face_subset = mesh.faces[facing_indices]
verts_idx = np.unique(face_subset)
verts_subset = mesh.vertices[verts_idx]

index_map = {old: i for i, old in enumerate(verts_idx)}
faces_mapped = np.array([[index_map[idx] for idx in face] for face in face_subset])

top_face = trimesh.Trimesh(vertices=verts_subset, faces=faces_mapped, process=False)# export the top face mesh
top_face.fill_holes()
top_face.export("/home/yufeiyang/Documents/FoundationPose/output/ob_000000k/model/top_face.obj")

# # Clean the top face
# # components = top_face.split(only_watertight=False)
# # top_face = max(components, key=lambda m: len(m.faces))

# # Find mean Z of the top surface
# top_z = np.mean(verts_subset[:, 2])
# print("top mean")
# # Target bottom Z
# target_z = np.min(mesh.vertices[:, 2])

# # Compute accurate offset downward
# offset_z = target_z - top_z

# # Apply translation downward
# top_face.apply_translation([0, 0, offset_z])

# # Combine with original mesh
# mesh_combined = trimesh.util.concatenate([mesh, top_face])
# mesh_combined.fill_holes()

# # Export
# mesh_combined.export("/home/yufeiyang/Documents/FoundationPose/output/ob_000000k/model/subset_mesh.obj")

