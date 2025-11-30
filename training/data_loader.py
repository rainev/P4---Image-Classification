# training/data_loader.py

import torch
import torchvision
import torchvision.transforms as transforms


def get_mnist_loader(
    data_dir: str = "./data",
    batch_size: int = 16,
    num_workers: int = 2,
):
    """
    Returns a DataLoader for the MNIST training set.
    Downloads the dataset into data_dir if not yet present.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # => float tensor [0,1], shape (1, 28, 28)
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return train_loader