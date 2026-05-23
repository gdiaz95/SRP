""" mazes_data.py
    Maze related dataloaders
"""
import os
import torch
from torch.utils import data
from easy_to_hard_data import MazeDataset

# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, W0611

# Project root is three levels up from this file (deepthinking/utils/mazes_data.py)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_path(path):
    """Return absolute path: pass through if already absolute, else resolve from project root."""
    if os.path.isabs(path):
        return path
    return os.path.join(_PROJECT_ROOT, path)


def prepare_maze_loader(train_batch_size, test_batch_size, train_data, test_data,
                        data_type, train_data_path, shuffle=True):

    base_dir_train = _resolve_path(train_data_path)
    base_dir_test = os.path.join(_PROJECT_ROOT, "data", data_type)

    train_dataset = MazeDataset(base_dir_train, train=True, size=train_data, download=False)
    testset = MazeDataset(base_dir_test, train=False, size=test_data, download=False)

    train_split = int(0.8 * len(train_dataset))

    trainset, valset = torch.utils.data.random_split(
        train_dataset,
        [train_split, len(train_dataset) - train_split],
        generator=torch.Generator().manual_seed(42),
    )

    trainloader = data.DataLoader(trainset, num_workers=0, batch_size=train_batch_size,
                                  shuffle=shuffle, drop_last=True)
    valloader = data.DataLoader(valset, num_workers=0, batch_size=test_batch_size,
                                shuffle=False, drop_last=False)
    testloader = data.DataLoader(testset, num_workers=0, batch_size=test_batch_size,
                                 shuffle=False, drop_last=False)

    return {"train": trainloader, "test": testloader, "val": valloader}
