import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import Module, MSELoss

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
        z_t = t* z1 + (1-t) * z0
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
            t = torch.ones((batchsize, 1)).to(self.device) * i /num_steps
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
            t = torch.ones((batchsize, 1)).to(self.device) * i /num_steps
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
            loss = (target- pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
            loss = loss.mean()
            loss.backward()

            optimizer.step()
            loss_curve.append(loss.item())

        rectified_flow.loss_curve = loss_curve
        return rectified_flow

def draw_toy_plot(rectified_flow, z0, z1, num_steps, dimension=2):
    rectified_flow.v_model.eval()
    traj = rectified_flow.sample_ode(z0=z0, num_steps=num_steps)

    if dimension ==2:
        plt.figure(figsize=(4,4))
        plt.xlim(0,8)
        plt.ylim(0,8)
            
        plt.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
        plt.scatter(traj[0][:, 0].cpu().numpy(), traj[0][:, 1].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)
        plt.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), label='Generated', alpha=0.15)
        plt.legend()
        plt.title('Distribution')
        #plt.tight_layout()

        traj_particles = torch.stack(traj)
        #plt.figure(figsize=(4,4))
        plt.xlim(0,8)
        plt.ylim(0,8)
        plt.axis('equal')
        for i in range(30):
            plt.plot(traj_particles[:, i, 0].cpu().numpy(), traj_particles[:, i, 1].cpu().numpy())
        plt.title('Transport Trajectory')
        plt.tight_layout()

    if dimension == 3:
        fig = plt.figure(figsize=(10, 10), dpi=80)
        ax = fig.add_subplot(111, projection='3d')
        
        # plt.figure(figsize=(4,4))
        # plt.xlim(0,8)
        # plt.ylim(0,8)
            
        ax.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), z1[:, 2].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
        ax.scatter(traj[0][:, 0].cpu().numpy(), traj[0][:, 1].cpu().numpy(), traj[0][:, 2].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)
        ax.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), traj[-1][:, 2].cpu().numpy(), label='Generated', alpha=0.3)
        ax.legend()
        # ax.title('Distribution')
        # ax.tight_layout()

        traj_particles = torch.stack(traj)
        # ax.figure(figsize=(4,4))
        # ax.xlim(0,8)
        # ax.ylim(0,8)
        ax.axis('auto')
        ax.view_init(0.,100)
        for i in range(120):
            ax.plot(traj_particles[:, i, 0].cpu().numpy(), traj_particles[:, i, 1].cpu().numpy(), traj_particles[:, i, 2].cpu().numpy())
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
        z_t = t* z1 + (1-t) * z0
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
            t = torch.ones((batchsize, 1)).to(self.device) * i /num_steps
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
            t = torch.ones((batchsize, 1)).to(self.device) * i /num_steps
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
        z_t = t* z1 + (1-t) * z0
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
            t = torch.ones(batchsize).to(self.device) * i /num_steps
            t_expand = t.view(-1, 1, 1, 1).repeat(1, z.shape[1], z.shape[2], z.shape[3])

            pred = self.v_model(torch.stack([z, t_expand], axis = 1).reshape(-1, 2, 32,32))
            z = z.detach().clone() + pred * dt

            trajectory.append(z.detach().clone())

        return trajectory

    @torch.no_grad()
    def sample_ode_cond(self, z0, labels, num_steps):
        trajectory = []
        trajectory.append(z0.detach().clone())

        batchsize = z0.shape[0]

        dt = 1./num_steps
        z = z0.detach().clone()
        for i in range(num_steps):
            t = torch.ones(batchsize).to(self.device) * i /num_steps
            t_expand = t.view(-1, 1, 1, 1).repeat(1, z.shape[1], z.shape[2], z.shape[3])
            labels_expand = labels.view(-1, 1, 1, 1).repeat(1, z.shape[1], z.shape[2], z.shape[3])

            pred = self.v_model(torch.stack([z, t_expand, labels_expand], axis = 1).reshape(-1, 3, 32,32))
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
            t = torch.ones(batchsize).to(self.device) * i /num_steps
            t_expand = t.view(-1, 1, 1, 1).repeat(1, z.shape[1], z.shape[2], z.shape[3])

            pred = self.v_model(torch.stack([z, 1 -t_expand], axis = 1).reshape(-1, 2, 32,32))
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
            t = torch.ones(batchsize).to(self.device) * i /num_steps
            t_expand = t.view(-1, 1, 1, 1).repeat(1, z.shape[1], z.shape[2], z.shape[3])
            labels_expand = labels.view(-1, 1, 1, 1).repeat(1, z.shape[1], z.shape[2], z.shape[3])

            pred = self.v_model(torch.stack([z, 1 -t_expand, labels_expand], axis = 1).reshape(-1, 3, 32,32))
            z = z.detach().clone() - (pred * dt)

            trajectory.append(z.detach().clone())

        return trajectory

def train_rectified_flow_Unet(rectified_flow, optimizer, scheduler, dataloader, device, epochs):
    loss_curve = rectified_flow.loss_curve

    rectified_flow.v_model.train()
    for epoch in range(epochs):
        for batch_ in dataloader:
            optimizer.zero_grad()

            batch = batch_[0].to(device)            
            indeces = torch.randperm(batch.shape[0])
            batch = batch[indeces]
            z0 = torch.randn(batch.shape).to(device)
            t = torch.rand(batch.shape[0]).to(device)
            t_expand = t.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
            perturbed_data = t_expand * batch + (1.-t_expand) * z0

            target = batch - z0        
            score = rectified_flow.v_model(torch.stack([perturbed_data, t_expand], axis=1).reshape(-1,2,32,32))
            category = MSELoss()
            loss = category(score, target)

            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
            scheduler.step(loss.item())

    rectified_flow.loss_curve = loss_curve
    return rectified_flow

def train_rectified_flow_Unet_init(rectified_flow, optimizer, scheduler, dataloader, device, epochs):
    loss_curve = rectified_flow.loss_curve

    rectified_flow.v_model.train()
    for epoch in range(epochs):
        for batch_ in dataloader:
            optimizer.zero_grad()

            batch = batch_[0].to(device)            
            indeces = torch.randperm(batch.shape[0])
            batch = batch[indeces]
            z0 = torch.randn(batch.shape).to(device)
            t = torch.rand(batch.shape[0]).to(device)
            t_expand = t.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
            perturbed_data = t_expand * batch + (1.-t_expand) * z0

            target = batch - z0        
            score = rectified_flow.v_model(torch.stack([perturbed_data, t_expand], axis=1).reshape(-1,2,32,32))
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
            t_expand = t.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
            perturbed_data = t_expand * batch + (1.-t_expand) * z0

            target = batch - z0        
            score = rectified_flow.v_model(torch.stack([perturbed_data, t_expand], axis=1).reshape(-1,2,32,32))
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
            t_expand = t.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
            labels = np.where(labels == 4, 0, 1)
            labels = torch.tensor(labels).to(device)
            labels_expand = labels.view(-1,1,1,1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
            perturbed_data = t_expand * batch + (1.-t_expand) * z0

            target = batch - z0        
            score = rectified_flow.v_model(torch.stack([perturbed_data, t_expand, labels_expand], axis=1).reshape(-1,3,32,32))
            category = MSELoss()
            loss = category(score, target)

            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
            scheduler.step(loss.item())

    rectified_flow.loss_curve = loss_curve
    return rectified_flow

def train_rectified_flow_Unet_cond(rectified_flow, optimizer, scheduler, dataloader, device, epochs):
    loss_curve = rectified_flow.loss_curve

    rectified_flow.v_model.train()
    for epoch in range(epochs):
        for batch_ in dataloader:
            optimizer.zero_grad()

            batch, labels = batch_
            indeces = torch.randperm(batch.shape[0])
            batch = batch[indeces].to(device) 
            z0 = torch.randn(batch.shape).to(device)
            t = torch.rand(batch.shape[0]).to(device)
            t_expand = t.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
            labels = np.where(labels == 1, 0, 1)
            labels = torch.tensor(labels).to(device)
            labels_expand = labels.view(-1,1,1,1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
            perturbed_data = t_expand * batch + (1.-t_expand) * z0

            target = batch - z0        
            score = rectified_flow.v_model(torch.stack([perturbed_data, t_expand, labels_expand], axis=1).reshape(-1,3,32,32))
            category = MSELoss()
            loss = category(score, target)

            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
            scheduler.step(loss.item())

    rectified_flow.loss_curve = loss_curve
    return rectified_flow