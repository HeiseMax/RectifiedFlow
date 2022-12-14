import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from torchvision.transforms import transforms
import torchvision.datasets as datasets

########## helpers ##########


def one_hot_image(labels, device):
    shape = (labels.shape[0], 1, 32, 32)
    labels_img = torch.zeros(shape).to(device)
    labels_img[np.arange(labels.shape[0]), 0, labels] = 1
    return labels_img

########## Plots ##########


def show_samples(rectified_flow, get_samples, rows, columns, channels, img_size, num_steps, device, conditional=False):
    rectified_flow.v_model.eval()
    img_init = get_samples(
        (rows*columns, channels, img_size, img_size), device=device)
    img = rectified_flow.sample_ode(img_init, 100)

    size = (4*columns, 4*rows)
    fig, ax = plt.subplots(rows, columns, figsize=(size),
                           sharex=True, sharey=True)

    for row in range(rows):
        for column in range(columns):
            ax[row, column].imshow(
                img[-1][row*columns + column, 0].detach().cpu().numpy())
    plt.show()


def show_trajectories(rectified_flow, get_samples, img_size, num_steps, device):
    rectified_flow.v_model.eval()
    img_init = get_samples((20, 1, img_size, img_size), device=device)
    img = rectified_flow.sample_ode(img_init, 100)
    s = torch.zeros(20, len(img), 1, 32, 32)
    for i in range(len(img)):
        s[:, i] = img[i]
    for i in range(20):
        plt.plot(s[i, :, 0, 15, 15].cpu().numpy())
    plt.plot()

########## DataLoaders ##########


def load_MNIST(batchsize, classes=None):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_dataset = datasets.MNIST(
        "./data",
        download=True,
        train=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        "./data",
        download=True,
        train=False,
        transform=transform,
    )

    if classes != None:
        train_dataset.data = train_dataset.data[np.isin(
            train_dataset.targets, classes)]
        train_dataset.targets = train_dataset.targets[np.isin(
            train_dataset.targets, classes)]

        test_dataset.data = test_dataset.data[np.isin(
            test_dataset.targets, classes)]
        test_dataset.targets = test_dataset.targets[np.isin(
            test_dataset.targets, classes)]
    else:
        classes = np.unique(train_dataset.targets)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    return train_dataset, test_dataset, train_dataloader, test_dataloader, classes


def load_FashionMNIST(batchsize, classes=None):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_dataset = datasets.FashionMNIST(
        "./data",
        download=True,
        train=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        "./data",
        download=True,
        train=False,
        transform=transform,
    )

    if classes != None:
        train_dataset.data = train_dataset.data[np.isin(
            train_dataset.targets, classes)]
        train_dataset.targets = train_dataset.targets[np.isin(
            train_dataset.targets, classes)]

        test_dataset.data = test_dataset.data[np.isin(
            test_dataset.targets, classes)]
        test_dataset.targets = test_dataset.targets[np.isin(
            test_dataset.targets, classes)]
    else:
        classes = np.unique(train_dataset.targets)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    return train_dataset, test_dataset, train_dataloader, test_dataloader, classes


def load_Cifar10(batchsize, classes=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(
        "./data",
        download=True,
        train=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        "./data",
        download=True,
        train=False,
        transform=transform,
    )

    if classes != None:
        train_dataset.data = train_dataset.data[np.isin(
            train_dataset.targets, classes)]
        train_dataset.targets = train_dataset.targets[np.isin(
            train_dataset.targets, classes)]

        test_dataset.data = test_dataset.data[np.isin(
            test_dataset.targets, classes)]
        test_dataset.targets = test_dataset.targets[np.isin(
            test_dataset.targets, classes)]
    else:
        classes = np.unique(train_dataset.targets)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    return train_dataset, test_dataset, train_dataloader, test_dataloader, classes
