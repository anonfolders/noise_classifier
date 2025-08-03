import numpy as np

def project_pointcloud(pc, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
    x, y, z, intensity = pc[:,0], pc[:,1], pc[:,2], pc[:,3]
    depth = np.linalg.norm(pc[:, :3], 2, axis=1)
    depth = np.clip(depth, a_min=1e-6, a_max=None)

    yaw = np.arctan2(y, x)
    pitch = np.arcsin(np.clip(z / depth, -1.0, 1.0))  # Clip to valid asin range

    fov = np.radians(fov_up - fov_down)
    proj_x = 0.5 * (yaw / np.pi + 1.0) * W
    proj_y = (1.0 - (pitch - np.radians(fov_down)) / fov) * H

    proj_x = np.clip(np.floor(proj_x), 0, W - 1).astype(np.int32)
    proj_y = np.clip(np.floor(proj_y), 0, H - 1).astype(np.int32)

    range_img = np.zeros((H, W, 2), dtype=np.float32)
    range_img[proj_y, proj_x, 0] = depth
    range_img[proj_y, proj_x, 1] = intensity
    return range_img
