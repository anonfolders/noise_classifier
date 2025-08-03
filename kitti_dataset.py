import torch
from torch.utils.data import Dataset
from flatten_pcd import project_pointcloud


class KITTIDataset(Dataset):
    def __init__(self, pointclouds, labels):
        """
        Args:
            pointclouds: list of N x 4 numpy arrays (XYZI)
            labels: list of integers representing scene condition (e.g., 0 = clear)
        """
        self.pointclouds = pointclouds
        self.labels = labels

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        pc = self.pointclouds[idx]
        label = self.labels[idx]

        img = project_pointcloud(pc)  # shape: (H, W, 2)
        img = torch.tensor(img).permute(2, 0, 1)  # [2, H, W] format for CNN

        return img, torch.tensor(label, dtype=torch.long)
