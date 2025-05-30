from pathlib import Path
import numpy as np
import trimesh
import torch

# subsample the vertices of the mesh to the given number of points using farthest point sampling
def subsample_ply(ply_path, num_points=4096):
    print(f'Subsampling {ply_path} to {num_points} points')
    mesh = trimesh.load_mesh(ply_path)
    vertices = mesh.vertices
    num_vertices = vertices.shape[0]
    if num_vertices <= num_points:
        selected_vertices = vertices
    else:
        # use cuda-based farthest point sampling
        vertices = torch.tensor(vertices).cuda()
        selected_indices = torch.zeros(num_points, dtype=torch.long).cuda()
        selected_indices[0] = torch.randint(num_vertices, (1,))
        distances = torch.norm(vertices[selected_indices[0]].unsqueeze(0) - vertices, dim=1)
        for i in range(1, num_points):
            selected_indices[i] = torch.argmax(distances)
            distances = torch.min(distances, torch.norm(vertices[selected_indices[i]].unsqueeze(0) - vertices, dim=1))
        selected_vertices = vertices[selected_indices].cpu().numpy()
    # export the subsampled points to a new ply file
    trimesh.Trimesh(selected_vertices).export(ply_path.parent / f'points_{num_points}.ply')


dataset_dir = Path('./data/scene_mesh_4render/')
for scene_dir in dataset_dir.iterdir():
    if not scene_dir.is_dir():
        continue
    scene_ply = scene_dir / 'mesh_floor_zup.ply'
    subsample_ply(scene_ply, num_points=8192 * 2)
    # break