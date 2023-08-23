import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import Module, MSELoss
from torchvision.transforms import RandomHorizontalFlip

from scipy.integrate import RK45

from util import show_samples


class Toy_RectifiedFlow(Module):
    def __init__(self, v_model, device):
        super().__init__()
        self.v_model = v_model.to(device)
        self.device = device

        self.loss_curve = []

    def get_train_tuple(self, z0, z1):
        # random times
        t = torch.rand((z0.shape[0], 1)).to(self.device)
        # linear interpolation
        z_t = t * z1 + (1-t) * z0
        # connection line
        target = z1 - z0

        return z_t, t, target

    def get_train_tuple_non_linear(self, z0, z1):
        # random times
        t = torch.rand((z0.shape[0], 1)).to(self.device)
        # linear interpolation
        z_t = t^2 * z1 + (1-t)^2 * z0
        # connection line
        target = 2* t *z1 - 2* (1-t) *z0

        return z_t, t, target

    @torch.no_grad()
    def sample_ode(self, z0, num_steps):
        trajectory = []
        trajectory.append(z0.detach().clone())

        batchsize = z0.shape[0]

        dt = 1./num_steps
        z = z0.detach().clone()
        for i in range(num_steps):
            t = torch.ones((batchsize, 1)).to(self.device) * i / num_steps
            pred = self.v_model(z, t)
            z = z.detach().clone() + pred * dt

            trajectory.append(z.detach().clone())

        return trajectory

    @torch.no_grad()
    def reverse_sample_ode(self, z1, num_steps):
        trajectory = []
        trajectory.append(z1.detach().clone())

        batchsize = z1.shape[0]

        dt = 1./num_steps
        z = z1.detach().clone()
        for i in range(num_steps):
            t = torch.ones((batchsize, 1)).to(self.device) * i / num_steps
            pred = self.v_model(z, 1 - t)
            z = z.detach().clone() - (pred * dt)

            trajectory.append(z.detach().clone())

        return trajectory


def train_toy_rectified_flow(rectified_flow, optimizer, pairs, batchsize, inner_iters):
    loss_curve = rectified_flow.loss_curve
    rectified_flow.v_model.train()
    for i in range(inner_iters + 1):
        optimizer.zero_grad()
        indeces = torch.randperm(len(pairs))[:batchsize]
        batch = pairs[indeces]
        z0 = batch[:, 0].detach().clone()
        z1 = batch[:, 1].detach().clone()
        z_t, t, target = rectified_flow.get_train_tuple(z0, z1)

        pred = rectified_flow.v_model(z_t, t)
        loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
        loss = loss.mean()
        loss.backward()

        optimizer.step()
        loss_curve.append(loss.item())

    rectified_flow.loss_curve = loss_curve
    return rectified_flow

def train_toy_rectified_flow_non_linear(rectified_flow, optimizer, pairs, batchsize, inner_iters):
    loss_curve = rectified_flow.loss_curve
    rectified_flow.v_model.train()
    for i in range(inner_iters + 1):
        optimizer.zero_grad()
        indeces = torch.randperm(len(pairs))[:batchsize]
        batch = pairs[indeces]
        z0 = batch[:, 0].detach().clone()
        z1 = batch[:, 1].detach().clone()
        z_t, t, target = rectified_flow.get_train_tuple_non_linear(z0, z1)

        pred = rectified_flow.v_model(z_t, t)
        loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
        loss = loss.mean()
        loss.backward()

        optimizer.step()
        loss_curve.append(loss.item())

    rectified_flow.loss_curve = loss_curve
    return rectified_flow


def draw_toy_plot(rectified_flow, z0, z1, num_steps, dimension=2):
    rectified_flow.v_model.eval()
    traj = rectified_flow.sample_ode(z0=z0, num_steps=num_steps)

    if dimension == 2:
        plt.figure(figsize=(4, 4))
        plt.xlim(0, 8)
        plt.ylim(0, 8)

        plt.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu(
        ).numpy(), label=r'$\pi_1$', alpha=0.15)
        plt.scatter(traj[0][:, 0].cpu().numpy(), traj[0]
                    [:, 1].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)
        plt.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1]
                    [:, 1].cpu().numpy(), label='Generated', alpha=0.15)
        plt.legend()
        plt.title('Distribution')
        # plt.tight_layout()

        traj_particles = torch.stack(traj)
        # plt.figure(figsize=(4,4))
        plt.xlim(0, 8)
        plt.ylim(0, 8)
        plt.axis('equal')
        for i in range(30):
            plt.plot(traj_particles[:, i, 0].cpu().numpy(),
                     traj_particles[:, i, 1].cpu().numpy())
        plt.title('Transport Trajectory')
        plt.tight_layout()

    if dimension == 3:
        fig = plt.figure(figsize=(10, 10), dpi=80)
        ax = fig.add_subplot(111, projection='3d')

        # plt.figure(figsize=(4,4))
        # plt.xlim(0,8)
        # plt.ylim(0,8)

        ax.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(),
                   z1[:, 2].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
        ax.scatter(traj[0][:, 0].cpu().numpy(), traj[0][:, 1].cpu().numpy(
        ), traj[0][:, 2].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)
        ax.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(),
                   traj[-1][:, 2].cpu().numpy(), label='Generated', alpha=0.3)
        ax.legend()
        # ax.title('Distribution')
        # ax.tight_layout()

        traj_particles = torch.stack(traj)
        # ax.figure(figsize=(4,4))
        # ax.xlim(0,8)
        # ax.ylim(0,8)
        ax.axis('auto')
        ax.view_init(0., 100)
        for i in range(120):
            ax.plot(traj_particles[:, i, 0].cpu().numpy(), traj_particles[:, i, 1].cpu(
            ).numpy(), traj_particles[:, i, 2].cpu().numpy())
        # ax.title('Transport Trajectory')
        # ax.tight_layout()

###############################################


class RectifiedFlow(Module):
    def __init__(self, v_model, device):
        super().__init__()
        self.v_model = v_model.to(device)
        self.device = device

        self.loss_curve = []
        self.train_loss_curve = []

    def get_train_tuple(self, z0, z1):
        # random times
        t = torch.rand((z0.shape[0], 1)).to(self.device)
        # linear interpolation
        z_t = t * z1 + (1-t) * z0
        # connection line
        target = z1 - z0

        return z_t, t, target

    @torch.no_grad()
    def sample_ode(self, z0, num_steps):
        trajectory = []
        trajectory.append(z0.detach().clone())

        batchsize = z0.shape[0]

        dt = 1./num_steps
        z = z0.detach().clone()
        for i in range(num_steps):
            t = torch.ones((batchsize, 1)).to(self.device) * i / num_steps
            pred = self.v_model(z, t)
            z = z.detach().clone() + pred * dt

            trajectory.append(z.detach().clone())

        return trajectory

    @torch.no_grad()
    def reverse_sample_ode(self, z1, num_steps):
        trajectory = []
        trajectory.append(z1.detach().clone())

        batchsize = z1.shape[0]

        dt = 1./num_steps
        z = z1.detach().clone()
        for i in range(num_steps):
            t = torch.ones((batchsize, 1)).to(self.device) * i / num_steps
            pred = self.v_model(z, 1 - t)
            z = z.detach().clone() - (pred * dt)

            trajectory.append(z.detach().clone())

        return trajectory


def train_rectified_flow(rectified_flow, optimizer, scheduler, dataloader, device, epochs):
    loss_curve = rectified_flow.loss_curve

    rectified_flow.v_model.train()
    for epoch in range(epochs):
        for batch_ in dataloader:
            optimizer.zero_grad()

            batch = batch_[0].reshape(-1, 28*28).to(device)
            indeces = torch.randperm(batch.shape[0])
            batch = batch[indeces]
            z0 = torch.randn(batch.shape).to(device)
            t = torch.rand((batch.shape[0], 1)).to(device)
            perturbed_data = t * batch + (1.-t) * z0

            target = batch - z0
            score = rectified_flow.v_model(perturbed_data, t)
            category = MSELoss()
            loss = category(score, target)

            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
            scheduler.step(loss.item())

    rectified_flow.loss_curve = loss_curve
    return rectified_flow

####### U-Net Rectified Flow ########


class RectifiedFlow_Unet(Module):
    def __init__(self, v_model, device):
        super().__init__()
        self.v_model = v_model.to(device)
        self.device = device

        self.loss_curve = []
        self.train_loss_curve = []

    def get_train_tuple(self, z0, z1):
        # random times
        t = torch.rand((z0.shape[0], 1)).to(self.device)
        # linear interpolation
        z_t = t * z1 + (1-t) * z0
        # target gradient
        target = z1 - z0

        return z_t, t, target

    @torch.no_grad()
    def sample_ode(self, z0, num_steps):
        trajectory = []
        trajectory.append(z0.detach().clone())

        batchsize = z0.shape[0]

        dt = 1./num_steps
        z = z0.detach().clone()
        for i in range(num_steps):
            t = torch.ones(batchsize).to(self.device) * i / num_steps

            pred = self.v_model(t, z)
            z = z.detach().clone() + pred * dt

            trajectory.append(z.detach().clone())

        return trajectory

    @torch.no_grad()
    def sample_ode_RK(self, z0, num_steps):  # not functional
        trajectory = []
        trajectory.append(z0.detach().clone())

        batchsize = z0.shape[0]

        z = z0.detach().clone()

        solver = RK45(self.v_model.cpu(), 0.0,
                      z0.flatten().detach().clone().cpu().numpy(), 1.0)
        solver.step()
        sol = solver.dense_output()
        print(sol)

        return trajectory

    @torch.no_grad()
    def sample_ode_cond(self, z0, labels, num_steps):
        trajectory = []
        trajectory.append(z0.detach().clone())

        batchsize = z0.shape[0]

        dt = 1./num_steps
        z = z0.detach().clone()
        for i in range(num_steps):
            t = torch.ones(batchsize).to(self.device) * i / num_steps

            pred = self.v_model(t, z, labels)
            z = z.detach().clone() + pred * dt

            trajectory.append(z.detach().clone())

        return trajectory

    @torch.no_grad()
    def reverse_sample_ode(self, z1, num_steps):
        trajectory = []
        trajectory.append(z1.detach().clone())

        batchsize = z1.shape[0]

        dt = 1./num_steps
        z = z1.detach().clone()
        for i in range(num_steps):
            t = torch.ones(batchsize).to(self.device) * i / num_steps

            pred = self.v_model(1 - t, z)
            z = z.detach().clone() - (pred * dt)

            trajectory.append(z.detach().clone())

        return trajectory

    @torch.no_grad()
    def reverse_sample_ode_cond(self, z1, labels, num_steps):
        trajectory = []
        trajectory.append(z1.detach().clone())

        batchsize = z1.shape[0]

        dt = 1./num_steps
        z = z1.detach().clone()
        for i in range(num_steps):
            t = torch.ones(batchsize).to(self.device) * i / num_steps

            pred = self.v_model(1-t, z, labels)
            z = z.detach().clone() - (pred * dt)

            trajectory.append(z.detach().clone())

        return trajectory


def train_rectified_flow_Unet(rectified_flow, optimizer, scheduler, dataloader, device, epochs, noise_factor=0):
    loss_curve = rectified_flow.loss_curve

    rectified_flow.v_model.train()
    for epoch in range(epochs):
        for batch_ in dataloader:
            optimizer.zero_grad()

            batch = batch_[0].to(device)
            indeces = torch.randperm(batch.shape[0])
            batch = batch[indeces]
            if noise_factor != 0:
                batch = batch + (torch.randn(batch.shape).to(device) * noise_factor) - \
                    (torch.randn(batch.shape).to(device) * noise_factor)
            z0 = torch.randn(batch.shape).to(device)
            t = torch.rand(batch.shape[0]).to(device)
            t_expand = t.view(-1, 1, 1, 1).repeat(1,
                                                  batch.shape[1], batch.shape[2], batch.shape[3])
            perturbed_data = t_expand * batch + (1.-t_expand) * z0

            target = batch - z0
            score = rectified_flow.v_model(t, perturbed_data)
            category = MSELoss()
            loss = category(score, target)

            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
            scheduler.step(loss.item())

    rectified_flow.loss_curve = loss_curve
    return rectified_flow


def train_rectified_flow_Unet_2peak(rectified_flow, optimizer, scheduler, dataloader, device, epochs, noise_factor=0):
    loss_curve = rectified_flow.loss_curve

    rectified_flow.v_model.train()
    for epoch in range(epochs):
        for batch_ in dataloader:
            optimizer.zero_grad()

            batch = batch_[0].to(device)
            indeces = torch.randperm(batch.shape[0])
            batch = batch[indeces]
            if noise_factor != 0:
                batch = batch + (torch.randn(batch.shape).to(device) * noise_factor) - \
                    (torch.randn(batch.shape).to(device) * noise_factor)
            z0 = torch.randn(batch.shape).to(device)
            z0[:int(batch.shape[0] / 2)] = z0[:int(batch.shape[0] / 2)] * 0.5 + 1
            z0[int(batch.shape[0] / 2):] = z0[int(batch.shape[0] / 2):] * 0.5 - 1
            t = torch.rand(batch.shape[0]).to(device)
            t_expand = t.view(-1, 1, 1, 1).repeat(1,
                                                  batch.shape[1], batch.shape[2], batch.shape[3])
            perturbed_data = t_expand * batch + (1.-t_expand) * z0

            target = batch - z0
            score = rectified_flow.v_model(t, perturbed_data)
            category = MSELoss()
            loss = category(score, target)

            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
            scheduler.step(loss.item())

    rectified_flow.loss_curve = loss_curve
    return rectified_flow


def train_rectified_flow_Unet_shifted(rectified_flow, optimizer, scheduler, dataloader, device, epochs, noise_factor=0):
    loss_curve = rectified_flow.loss_curve

    rectified_flow.v_model.train()
    for epoch in range(epochs):
        for batch_ in dataloader:
            optimizer.zero_grad()

            batch = batch_[0].to(device)
            indeces = torch.randperm(batch.shape[0])
            batch = batch[indeces]
            if noise_factor != 0:
                batch = batch + (torch.randn(batch.shape).to(device) * noise_factor) - \
                    (torch.randn(batch.shape).to(device) * noise_factor)
            z0 = torch.randn(batch.shape).to(device) + 2
            t = torch.rand(batch.shape[0]).to(device)
            t_expand = t.view(-1, 1, 1, 1).repeat(1,
                                                  batch.shape[1], batch.shape[2], batch.shape[3])
            perturbed_data = t_expand * batch + (1.-t_expand) * z0

            target = batch - z0
            score = rectified_flow.v_model(t, perturbed_data)
            category = MSELoss()
            loss = category(score, target)

            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
            scheduler.step(loss.item())

    rectified_flow.loss_curve = loss_curve
    return rectified_flow


def train_rectified_flow_Unet_cond(rectified_flow, optimizer, scheduler, dataloader, device, epochs, noise_factor=0):
    loss_curve = rectified_flow.loss_curve

    rectified_flow.v_model.train()
    for epoch in range(epochs):
        for batch_ in dataloader:
            optimizer.zero_grad()

            batch, labels = batch_
            indeces = torch.randperm(batch.shape[0])
            batch = batch[indeces].to(device)
            if noise_factor != 0:
                batch = batch + (torch.randn(batch.shape).to(device) * noise_factor) - \
                    (torch.randn(batch.shape).to(device) * noise_factor)
            z0 = torch.randn(batch.shape).to(device)
            t = torch.rand(batch.shape[0]).to(device)
            t_expand = t.view(-1, 1, 1, 1).repeat(1,
                                                  batch.shape[1], batch.shape[2], batch.shape[3])
            perturbed_data = t_expand * batch + (1.-t_expand) * z0

            labels = np.where(labels == 1, 0, 1)
            labels = torch.tensor(labels).to(device)
            labels = labels[indeces]
            #labels_expand = labels.view(-1,1,1,1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])

            target = batch - z0
            score = rectified_flow.v_model(t, perturbed_data, labels)
            category = MSELoss()
            loss = category(score, target)

            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
            scheduler.step(loss.item())

    rectified_flow.loss_curve = loss_curve
    return rectified_flow


def one_hot_image(labels, device):
    shape = (labels.shape[0], 1, 32, 32)
    labels_img = torch.zeros(shape).to(device)
    labels_img[np.arange(labels.shape[0]), 0, labels] = 1
    return labels_img


def train_rectified_flow_Unet_cond_10classes(rectified_flow, optimizer, scheduler, dataloader, device, epochs, noise_factor=0):
    loss_curve = rectified_flow.loss_curve

    rectified_flow.v_model.train()
    for epoch in range(epochs):
        for batch_ in dataloader:
            optimizer.zero_grad()

            batch, labels = batch_
            indeces = torch.randperm(batch.shape[0])
            batch = batch[indeces].to(device)
            if noise_factor != 0:
                batch = batch + (torch.randn(batch.shape).to(device) * noise_factor) - \
                    (torch.randn(batch.shape).to(device) * noise_factor)
            z0 = torch.randn(batch.shape).to(device)
            t = torch.rand(batch.shape[0]).to(device)
            t_expand = t.view(-1, 1, 1, 1).repeat(1,
                                                  batch.shape[1], batch.shape[2], batch.shape[3])
            perturbed_data = t_expand * batch + (1.-t_expand) * z0

            labels = torch.tensor(labels[indeces].clone().detach()).to(device)
            labels = one_hot_image(labels, device)

            target = batch - z0
            score = rectified_flow.v_model(t, perturbed_data, labels)
            category = MSELoss()
            loss = category(score, target)

            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
            scheduler.step(loss.item())

    rectified_flow.loss_curve = loss_curve
    return rectified_flow


def train_rectified_flow_Unet_cond_10classes2(rectified_flow, optimizer, scheduler, dataloader, device, epochs, noise_factor=0):
    loss_curve = rectified_flow.loss_curve

    rectified_flow.v_model.train()
    for epoch in range(epochs):
        for batch_ in dataloader:
            optimizer.zero_grad()

            batch, labels = batch_
            indeces = torch.randperm(batch.shape[0])
            batch = batch[indeces].to(device)
            if noise_factor != 0:
                batch = batch + (torch.randn(batch.shape).to(device) * noise_factor) - \
                    (torch.randn(batch.shape).to(device) * noise_factor)
            z0 = torch.randn(batch.shape).to(device)
            t = torch.rand(batch.shape[0]).to(device)
            t_expand = t.view(-1, 1, 1, 1).repeat(1,
                                                  batch.shape[1], batch.shape[2], batch.shape[3])
            perturbed_data = t_expand * batch + (1.-t_expand) * z0

            labels = torch.tensor(labels).to(device)
            labels = labels[indeces] / 10

            target = batch - z0
            score = rectified_flow.v_model(t, perturbed_data, labels)
            category = MSELoss()
            loss = category(score, target)

            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
            scheduler.step(loss.item())

    rectified_flow.loss_curve = loss_curve
    return rectified_flow


def train_rectified_flow_Unet_cond2(rectified_flow, optimizer, scheduler, dataloader, device, epochs, noise_factor=0):
    loss_curve = rectified_flow.loss_curve

    rectified_flow.v_model.train()
    for epoch in range(epochs):
        for batch_ in dataloader:
            optimizer.zero_grad()

            batch, labels = batch_
            indeces = torch.randperm(batch.shape[0])
            batch = batch[indeces].to(device)
            if noise_factor != 0:
                batch = batch + (torch.randn(batch.shape).to(device) * noise_factor) - \
                    (torch.randn(batch.shape).to(device) * noise_factor)

            blob1 = torch.randn(batch.shape).to(device) * 0.5 + 1
            blob2 = torch.randn(batch.shape).to(device) * 0.5 - 1

            labels = torch.tensor(labels).to(device)[indeces]
            c_expand = labels.view(-1, 1, 1, 1).repeat(1,
                                                       1, batch.shape[2], batch.shape[3])
            z0 = torch.where(c_expand == 1, blob1, blob2)

            t = torch.rand(batch.shape[0]).to(device)
            t_expand = t.view(-1, 1, 1, 1).repeat(1,
                                                  batch.shape[1], batch.shape[2], batch.shape[3])
            perturbed_data = t_expand * batch + (1.-t_expand) * z0

            target = batch - z0
            score = rectified_flow.v_model(t, perturbed_data)
            category = MSELoss()
            loss = category(score, target)

            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
            scheduler.step(loss.item())

    rectified_flow.loss_curve = loss_curve
    return rectified_flow


def train_rectified_flow_Unet_reverse(rectified_flow, optimizer, scheduler, dataloader, device, epochs):
    loss_curve = rectified_flow.loss_curve

    rectified_flow.v_model.train()
    for epoch in range(epochs):
        for batch_ in dataloader:
            optimizer.zero_grad()

            batch = batch_[0].to(device)
            indeces = torch.randperm(batch.shape[0])
            batch = batch[indeces]
            z0 = batch
            batch = torch.randn(batch.shape).to(device)
            t = torch.rand(batch.shape[0]).to(device)
            t_expand = t.view(-1, 1, 1, 1).repeat(1,
                                                  batch.shape[1], batch.shape[2], batch.shape[3])
            perturbed_data = t_expand * batch + (1.-t_expand) * z0

            target = batch - z0
            score = rectified_flow.v_model(torch.stack(
                [perturbed_data, t_expand], axis=1).reshape(-1, 2, 32, 32))
            category = MSELoss()
            loss = category(score, target)

            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
            scheduler.step(loss.item())

    rectified_flow.loss_curve = loss_curve
    return rectified_flow


def train_rectified_flow_Unet_cond_reverse(rectified_flow, optimizer, scheduler, dataloader, device, epochs):
    loss_curve = rectified_flow.loss_curve

    rectified_flow.v_model.train()
    for epoch in range(epochs):
        for batch_ in dataloader:
            optimizer.zero_grad()

            batch, labels = batch_
            indeces = torch.randperm(batch.shape[0])
            batch = batch[indeces].to(device)
            z0 = batch
            batch = torch.randn(batch.shape).to(device)
            t = torch.rand(batch.shape[0]).to(device)
            t_expand = t.view(-1, 1, 1, 1).repeat(1,
                                                  batch.shape[1], batch.shape[2], batch.shape[3])
            labels = np.where(labels == 4, 0, 1)
            labels = torch.tensor(labels).to(device)
            labels_expand = labels.view(-1, 1, 1, 1).repeat(1,
                                                            batch.shape[1], batch.shape[2], batch.shape[3])
            perturbed_data = t_expand * batch + (1.-t_expand) * z0

            target = batch - z0
            score = rectified_flow.v_model(torch.stack(
                [perturbed_data, t_expand, labels_expand], axis=1).reshape(-1, 3, 32, 32))
            category = MSELoss()
            loss = category(score, target)

            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
            scheduler.step(loss.item())

    rectified_flow.loss_curve = loss_curve
    return rectified_flow


def train(rectified_flow, conditional, optimizer, scheduler, dataloader, get_samples, device, epochs, noise_factor=0, flip=False):
    loss_curve = rectified_flow.loss_curve

    rectified_flow.v_model.train()
    for epoch in range(epochs):
        for batch_ in dataloader:

            optimizer.zero_grad()

            batch, labels = batch_
            indeces = torch.randperm(batch.shape[0])
            batch = batch[indeces].to(device)
            if flip:
                hflipper = RandomHorizontalFlip(p=0.5)
                batch = hflipper(batch)

            if noise_factor != 0:
                batch = batch + (torch.randn(batch.shape).to(device) * noise_factor) - \
                    (torch.randn(batch.shape).to(device) * noise_factor)

            z0 = get_samples(batch.shape, device)
            t = torch.rand(batch.shape[0]).to(device)
            t_expand = t.view(-1, 1, 1, 1).repeat(1,
                                                  batch.shape[1], batch.shape[2], batch.shape[3])
            perturbed_data = t_expand * batch + (1.-t_expand) * z0

            target = batch - z0
            if conditional:
                labels = torch.tensor(
                    labels[indeces].clone().detach()).to(device)
                score = rectified_flow.v_model(t, perturbed_data, labels)
            else:
                score = rectified_flow.v_model(t, perturbed_data)
            category = MSELoss() # try reduction sum?
            loss = category(score, target)

            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
            if scheduler != None:
                scheduler.step(loss.item())

        if epoch % 60 == 0:
            print(epoch)
            show_samples(rectified_flow, get_samples, 2, 5, 1, 32, 100, device, False)

    rectified_flow.loss_curve = loss_curve
    return rectified_flow

def train_cond(rectified_flow, optimizer, scheduler, dataloader, get_samples, encoder, device, epochs, noise_factor=0, flip=False):
    loss_curve = rectified_flow.loss_curve

    rectified_flow.v_model.train()
    for epoch in range(epochs):
        for batch_ in dataloader:

            optimizer.zero_grad()

            batch, labels = batch_
            indeces = torch.randperm(batch.shape[0])
            batch = batch[indeces].to(device)
            if flip:
                hflipper = RandomHorizontalFlip(p=0.5)
                batch = hflipper(batch)

            if noise_factor != 0:
                batch = batch + (torch.randn(batch.shape).to(device) * noise_factor) - \
                    (torch.randn(batch.shape).to(device) * noise_factor)

            z0 = get_samples(batch.shape, device)
            t = torch.rand(batch.shape[0]).to(device)
            t_expand = t.view(-1, 1, 1, 1).repeat(1,
                                                  batch.shape[1], batch.shape[2], batch.shape[3])
            perturbed_data = t_expand * batch + (1.-t_expand) * z0

            encoder.eval()
            encoder.requires_grad_(False)
            c = encoder(batch[:, :1, 2:-2, 2:-2])

            target = batch - z0
            score = rectified_flow.v_model(t, perturbed_data, c)
            category = MSELoss()
            loss = category(score, target)

            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
            if scheduler != None:
                scheduler.step(loss.item())

        # if epoch % 60 == 0:
        #     print(epoch)
        #     show_samples(rectified_flow, get_samples, 2, 5, 1, 32, 100, device, False)

    rectified_flow.loss_curve = loss_curve
    return rectified_flow

def train_translation(rectified_flow, conditional, optimizer, scheduler, dataloader, get_samples, device, epochs, noise_factor=0, flip=False):
    loss_curve = rectified_flow.loss_curve

    rectified_flow.v_model.train()
    for epoch in range(epochs):
        for batch_ in dataloader:

            optimizer.zero_grad()

            batch, labels = batch_
            indeces = torch.randperm(batch.shape[0])
            batch = batch[indeces].to(device)
            if flip:
                hflipper = RandomHorizontalFlip(p=0.5)
                batch = hflipper(batch)

            if noise_factor != 0:
                batch = batch + (torch.randn(batch.shape).to(device) * noise_factor) - \
                    (torch.randn(batch.shape).to(device) * noise_factor)

            z0 = get_samples(batch.shape, device)
            if noise_factor != 0:
                z0 = z0 + (torch.randn(z0.shape).to(device) * noise_factor) - \
                    (torch.randn(z0.shape).to(device) * noise_factor)
            t = torch.rand(batch.shape[0]).to(device)
            t_expand = t.view(-1, 1, 1, 1).repeat(1,
                                                  batch.shape[1], batch.shape[2], batch.shape[3])
            perturbed_data = t_expand * batch + (1.-t_expand) * z0

            target = batch - z0
            if conditional:
                labels = torch.tensor(
                    labels[indeces].clone().detach()).to(device)
                score = rectified_flow.v_model(t, perturbed_data, labels)
            else:
                score = rectified_flow.v_model(t, perturbed_data)
            category = MSELoss() # try reduction sum?
            loss = category(score, target)

            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
            if scheduler != None:
                scheduler.step(loss.item())

    rectified_flow.loss_curve = loss_curve
    return rectified_flow
    
def train_reflow(rectified_flow, conditional, optimizer, scheduler, z0_initial, z1, batch_size, device, epochs, noise_factor=0, flip=False):
    loss_curve = rectified_flow.loss_curve

    rectified_flow.v_model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        indeces = torch.randperm(batch_size)

        z0, batch = torch.tensor(z0_initial[indeces]).to(device).float(), torch.tensor(z1[indeces]).to(device).float()
        if flip:
            hflipper = RandomHorizontalFlip(p=0.5)
            batch = hflipper(batch)

        if noise_factor != 0:
            batch = batch + (torch.randn(batch.shape).to(device) * noise_factor) - \
                (torch.randn(batch.shape).to(device) * noise_factor)

        t = torch.rand(batch.shape[0]).to(device).float()
        t_expand = t.view(-1, 1, 1, 1).repeat(1,
                                                batch.shape[1], batch.shape[2], batch.shape[3])
        perturbed_data = t_expand * batch + (1.-t_expand) * z0

        target = batch - z0
        if conditional:
            labels = torch.tensor(
                labels[indeces].clone().detach()).to(device)
            score = rectified_flow.v_model(t, perturbed_data, labels)
        else:
            score = rectified_flow.v_model(t, perturbed_data)
        category = MSELoss() # try reduction sum?
        loss = category(score, target)

        loss.backward()
        optimizer.step()
        loss_curve.append(loss.item())
        if scheduler != None:
            scheduler.step(loss.item())

    rectified_flow.loss_curve = loss_curve
    return rectified_flow