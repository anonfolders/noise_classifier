import os
import numpy as np


def read_kitti_bin(bin_path):
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)


def load_all_bins_in_folder(folder, cls_idx):
    pcds = []
    labels = []

    bin_files = [f for f in os.listdir(folder) if f.endswith(".bin")]
    for file in bin_files:
        file_path = os.path.join(folder, file)
        pc = read_kitti_bin(file_path)
        pcds.append(pc)
        labels.append(cls_idx)
    
    return pcds, labels


def load_kitti_bin_from_class_folders(root_dirs):
    """
    Loads .bin files from a folder structure where each subfolder is a class.
    
    Args:
        root_dirs (Dict[str, str]): path to top-level dataset directory

    Returns:
        pointclouds (List[np.ndarray])
        labels (List[int])
        label_map (Dict[str, int])  # e.g. {"clear": 0, "fog": 1, ...}
    """
    pointclouds = []
    labels = []
    label_map = {}

    for class_idx, class_name in enumerate(root_dirs.keys()):
        class_folder_all = root_dirs[class_name]
        label_map[class_name] = class_idx
        for sev, class_folder in class_folder_all.items():
            if os.path.isdir(class_folder):
                # bin_files = [f for f in os.listdir(class_folder) if f.endswith(".bin")]
                # for file in bin_files:
                #     file_path = os.path.join(class_folder, file)
                #     pc = read_kitti_bin(file_path)
                #     pointclouds.append(pc)
                #     labels.append(class_idx)
                pcs, lbls = load_all_bins_in_folder(class_folder, class_idx)
                pointclouds.extend(pcs)
                labels.extend(lbls)

    return pointclouds, labels, label_map

