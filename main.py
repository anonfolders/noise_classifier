from kitti_dataset import KITTIDataset
from torch.utils.data import DataLoader
from dataloader import load_kitti_bin_from_class_folders
from torch.utils.data import random_split
import torch
from cnn_classifier import NoiseClassifierCNN
from train_model import train_model
from collections import defaultdict, Counter
import numpy as np
from data_cfg import KITTI_ODO_08, KITTI_ODO_09, KITTI_ODO_10
import argparse
import sys


def train(roots):
    # Provide multiple dataset roots (each with class folders)
    # roots = KITTI_ODO_10

    pcs, labels, label_map = load_kitti_bin_from_class_folders(roots)

    print("Class-to-label mapping:", label_map)

    # Use your dataset class
    dataset = KITTIDataset(pcs, labels)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    # Model
    num_classes = len(label_map)
    model = NoiseClassifierCNN(num_classes)

    # Train
    train_model(model, train_loader, val_loader, epochs=10, lr=1e-3)

    torch.save(model.state_dict(), "noise_classifier_weights.pth")


def eval(roots):
    # load data
    # roots = KITTI_ODO_09

    pcs, labels, label_map = load_kitti_bin_from_class_folders(roots)
    dataset = KITTIDataset(pcs, labels)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # load model
    num_classes = len(label_map)  # from dataset
    model = NoiseClassifierCNN(num_classes)
    model.load_state_dict(torch.load("noise_classifier_weights.pth"))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # prediction results
    correct_per_class = np.zeros(num_classes, dtype=np.int32)
    total_per_class = np.zeros(num_classes, dtype=np.int32)
    misclassified_as = defaultdict(list)  # true_class: [pred1, pred2, ...]


    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)

            for label, pred in zip(y, preds):
                total_per_class[label.item()] += 1
                if label.item() == pred.item():
                    correct_per_class[label.item()] += 1
                else:
                    misclassified_as[label.item()].append(pred.item())

    # Accuracy and misclassification summary
    idx_to_class = {v: k for k, v in label_map.items()}

    # Compute per-class accuracy
    for class_name, class_idx in label_map.items():
        total = total_per_class[class_idx]
        correct = correct_per_class[class_idx]
        acc = correct / total if total > 0 else 0.0
        print(f"Class '{class_name}': Accuracy = {acc:.2%} ({correct}/{total})")

        if misclassified_as[class_idx]:
            counter = Counter(misclassified_as[class_idx])
            top3 = counter.most_common(3)
            print("  Top-3 misclassified as:")
            for pred_idx, count in top3:
                pred_name = idx_to_class[pred_idx]
                print(f"    → {pred_name} ({count} times)")
        else:
            print("  No misclassifications.")

# Map parameter to function
function_map = {
    'eval': eval,
    'train': train
}

data_map = {
    '08': KITTI_ODO_08,
    '09': KITTI_ODO_09,
    '10': KITTI_ODO_10
}


def main():
    parser = argparse.ArgumentParser(description="Function dispatcher")
    parser.add_argument('function', choices=function_map.keys(), help="Function to call")
    parser.add_argument('dataset', choices=data_map.keys(), help="Dataset to use")

    try:
        args = parser.parse_args()
        print(f'{args.function}ing on {args.dataset}')
        function_map[args.function](data_map[args.dataset])
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()




