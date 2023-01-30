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
    plt.rcParams['figure.dpi'] = 120
    rectified_flow.v_model.eval()
    img_init = get_samples(
        (rows*columns, channels, img_size, img_size), device=device)
    img = rectified_flow.sample_ode(img_init, num_steps)[-1]

    size = (4*columns, 4*rows)
    fig, ax = plt.subplots(rows, columns, figsize=(size),
                           sharex=True, sharey=True)

    for row in range(rows):
        for column in range(columns):
            i = row * columns + column
            min = torch.min(img[i])
            img[i] = img[i] - min
            max = torch.max(img[i])
            img[i] = img[i] / max
            ax[row, column].imshow(
                img[i].permute(1,2,0).detach().cpu().numpy())
    
    plt.xticks([])
    plt.yticks([])
    plt.show()

def show_samples_array(rectified_flow, samples, rows, columns, num_steps, device, conditional=False):
    plt.rcParams['figure.dpi'] = 120
    rectified_flow.v_model.eval()

    size = (4*columns, 4*rows)
    fig, ax = plt.subplots(rows, columns, figsize=(size),
                           sharex=True, sharey=True)

    for row in range(rows):
        for column in range(columns):
            i = row * columns + column
            img = rectified_flow.sample_ode(samples[i], num_steps)[-1]
            min = torch.min(img[0])
            img[0] = img[0] - min
            max = torch.max(img[0])
            img[0] = img[0] / max
            ax[row, column].imshow(
                img[0].permute(1,2,0).detach().cpu().numpy())
    
    plt.xticks([])
    plt.yticks([])
    plt.show()

def show_sample_temperature(rectified_flow, get_samples, rows, columns, channels, img_size, num_steps, device, conditional=False):
    plt.rcParams['figure.dpi'] = 120
    rectified_flow.v_model.eval()
    img_init = get_samples(
        (1, channels, img_size, img_size), device=device)

    size = (4*columns, 4*rows)
    fig, ax = plt.subplots(rows, columns, figsize=(size),
                           sharex=True, sharey=True)

    n = rows * columns

    for row in range(rows):
        for column in range(columns):
            i = row * columns + column
            img = rectified_flow.sample_ode(img_init * (1/n) * i, num_steps)[-1]
            min = torch.min(img[0])
            img[0] = img[0] - min
            max = torch.max(img[0])
            img[0] = img[0] / max
            ax[row, column].imshow(
                img[0].permute(1,2,0).detach().cpu().numpy())
    
    plt.xticks([])
    plt.yticks([])
    plt.show()

def show_sample_flows(rectified_flows, get_samples, num_samples, channels, img_size, num_steps, device, conditional=False):
    plt.rcParams['figure.dpi'] = 120

    rows = len(rectified_flows)
    columns = num_samples
    img_init = get_samples(
        (num_samples, channels, img_size, img_size), device=device)

    size = (4*columns, 4*rows)
    fig, ax = plt.subplots(rows, columns, figsize=(size),
                           sharex=True, sharey=True)

    for row in range(rows):
        rectified_flow = rectified_flows[row]
        rectified_flow.v_model.eval()
        for column in range(columns):
            img = rectified_flow.sample_ode(img_init, num_steps)[-1]
            min = torch.min(img[column])
            img[column] = img[column] - min
            max = torch.max(img[column])
            img[column] = img[column] / max
            ax[row, column].imshow(
                img[column].permute(1,2,0).detach().cpu().numpy())
    
    plt.xticks([])
    plt.yticks([])
    plt.show()

def show_sample_steps(rectified_flow, get_samples, rows, columns, channels, img_size, device, conditional=False):
    plt.rcParams['figure.dpi'] = 120
    rectified_flow.v_model.eval()
    img_init = get_samples(
        (1, channels, img_size, img_size), device=device)

    size = (4*columns, 4*rows)
    fig, ax = plt.subplots(rows, columns, figsize=(size),
                           sharex=True, sharey=True)

    for row in range(rows):
        for column in range(columns):
            i = row * columns + column
            img = rectified_flow.sample_ode(img_init, i +1)[-1]
            min = torch.min(img[0])
            img[0] = img[0] - min
            max = torch.max(img[0])
            img[0] = img[0] / max
            ax[row, column].imshow(
                img[0].permute(1,2,0).detach().cpu().numpy())
    
    plt.xticks([])
    plt.yticks([])
    plt.show()

def show_trajecrory_samples(rectified_flow, sample1, sample2, rows, columns, num_steps, conditional=False):
    plt.rcParams['figure.dpi'] = 120
    rectified_flow.v_model.eval()

    num_images = rows * columns

    dif = sample2 - sample1

    size = (4*columns, 4*rows)
    fig, ax = plt.subplots(rows, columns, figsize=(size),
        sharex=True, sharey=True)
    for row in range(rows):
        for column in range(columns):
            i = row*columns + column
            img_init = sample1 + i*(1/num_images)*dif
            img = rectified_flow.sample_ode(img_init, num_steps)[-1]
            min = torch.min(img[0])
            img[0] = img[0] - min
            max = torch.max(img[-1][0])
            img[0] = img[0] / max

            ax[row, column].imshow(img[0].permute(1,2,0).detach().cpu().numpy())

    plt.xticks([])
    plt.yticks([])
    plt.show()

def show_samples_translation(rectified_flow, get_samples, columns, channels, img_size, num_steps, device, conditional=False):
    plt.rcParams['figure.dpi'] = 120
    rectified_flow.v_model.eval()
    img_init = get_samples(
        (columns, channels, img_size, img_size), device=device)[:columns]
    img = rectified_flow.sample_ode(img_init, num_steps)[-1]

    size = (4*columns, 4*2)
    fig, ax = plt.subplots(2, columns, figsize=(size),
                           sharex=True, sharey=True)

    
    for column in range(columns):
        i = column
        min = torch.min(img[i])
        img[i] = img[i] - min
        max = torch.max(img[i])
        img[i] = img[i] / max
        ax[1, column].imshow(
            img[i].permute(1,2,0).detach().cpu().numpy())

        ax[0, column].imshow(
            img_init[i].permute(1,2,0).detach().cpu().numpy())

    plt.xticks([])
    plt.yticks([])
    plt.show()

def show_trajectory_translation(rectified_flow, get_samples, rows, columns, channels, img_size, num_steps, device, conditional=False):
    plt.rcParams['figure.dpi'] = 120
    rectified_flow.v_model.eval()
    img_init = get_samples(
        (rows*columns, channels, img_size, img_size), device=device)
    img = rectified_flow.sample_ode(img_init, num_steps)

    size = (4*columns, 4*rows)
    fig, ax = plt.subplots(rows, columns, figsize=(size),
                           sharex=True, sharey=True)

    f = num_steps / (rows * columns -1)

    for row in range(rows):
        for column in range(columns):
            i = row * columns + column
            min = torch.min(img[i*f])
            img[i*f] = img[i*f] - min
            max = torch.max(img[i*f])
            img[i*f] = img[i*f] / max
            ax[row, column].imshow(
                img[i*f].permute(1,2,0).detach().cpu().numpy())
    
    plt.xticks([])
    plt.yticks([])
    plt.show()

def show_trajectories(rectified_flow, get_samples, img_size, num_steps, x, y, device):
    rectified_flow.v_model.eval()
    plt.rcParams['figure.dpi'] = 120
    img_init = get_samples((20, 1, img_size, img_size), device=device)
    img = rectified_flow.sample_ode(img_init, num_steps)
    s = torch.zeros(20, len(img), 1, 32, 32)
    for i in range(len(img)):
        s[:, i] = img[i]
    for i in range(20):
        plt.plot(s[i, :, 0, x, y].cpu().numpy())
    plt.ylim(-1.8, 1.8)
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
        drop_last=True
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
    train_dataset.targets = np.array(train_dataset.targets)
    test_dataset.targets = np.array(test_dataset.targets)

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
