import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily

from RectifiedFlow import Toy_RectifiedFlow, train_toy_rectified_flow
from models import Toy_MLP, Toy_MLP_distill

from sklearn.datasets import make_moons, make_circles, make_swiss_roll

# parent class for toy problems
class Toy_problem():
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.name = type(self).__name__

        self.input_dim = 2
        self.num_piles_init = 1
        self.num_piles_target = 1
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/self.num_piles_init for i in range(self.num_piles_init)]))
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_init = MultivariateNormal(torch.tensor(
            [[[0, 0], [0, 0]]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2)]))
        self.distribution_target = MultivariateNormal(torch.tensor(
            [[[2, 0], [0, 0]]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2)]))

        self.info = "No info provided!"
        # add gaussian params to info / and parameter?

        self.flows = []
        self.distill = None #have to be saved and loaded

    def save(self, save_string_appendage=None):
        for i, flow in enumerate(self.flows):
            if save_string_appendage:
                torch.save(flow, f"models/ToyExamples/{self.name}_flow_{i}_{save_string_appendage}.pth")
            else:
                torch.save(flow, f"models/ToyExamples/{self.name}_flow_{i}.pth")
        if self.distill:
            if save_string_appendage:
                torch.save(self.distill, f"models/ToyExamples/{self.name}_distill_{save_string_appendage}.pth")
            else:
                torch.save(self.distill, f"models/ToyExamples/{self.name}_distill.pth")

    def load(self, save_string_appendage=None):
        flows = []
        for i in range(10):
            if save_string_appendage:
                fname = f"models/ToyExamples/{self.name}_flow_{i}_{save_string_appendage}.pth"
            else:
                fname = f"models/ToyExamples/{self.name}_flow_{i}.pth"
            if os.path.isfile(fname):
                flows.append(torch.load(fname))
            else:
                break
        self.flows = flows

        if save_string_appendage:
            fname = f"models/ToyExamples/{self.name}_distill_{save_string_appendage}.pth"
        else:
            fname = f"models/ToyExamples/{self.name}_distill.pth"
        if os.path.isfile(fname):
            self.distill = (torch.load(fname))
        else:
            self.distill = None

    def get_info(self):
        return self.info

    def show_distribution(self, num_samples=1000, show_pairs=False, num_pairs=100, elev=30, azim=-45, roll=0):
        pairs = self.get_pairs(num_samples)
        samples_init = pairs[:, 0]
        samples_target = pairs[:, 1]

        if self.input_dim == 2:
            plt.scatter(samples_init[:, 0].cpu().numpy(),
                        samples_init[:, 1].cpu().numpy())
            plt.scatter(samples_target[:, 0].cpu().numpy(),
                        samples_target[:, 1].cpu().numpy())

            if show_pairs:
                for pair in range(num_pairs):
                    plt.plot([pairs[pair][0][0].cpu().numpy(), pairs[pair][1][0].cpu().numpy()], [
                            pairs[pair][0][1].cpu().numpy(), pairs[pair][1][1].cpu().numpy()])
            plt.show()

        if self.input_dim == 3:
            fig = plt.figure(figsize=(10, 10), dpi=120)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(samples_init[:,0].cpu().numpy(), samples_init[:,1].cpu().numpy(),samples_init[:,2].cpu().numpy(), alpha=0.1)
            ax.scatter(samples_target[:,0].cpu().numpy(), samples_target[:,1].cpu().numpy(), samples_target[:,2].cpu().numpy(), alpha=0.1)

            if show_pairs:
                for pair in range(num_pairs):
                    plt.plot([pairs[pair][0][0].cpu().numpy(), pairs[pair][1][0].cpu().numpy()], [pairs[pair][0][1].cpu().numpy(), pairs[pair][1][1].cpu().numpy()], [pairs[pair][0][2].cpu().numpy(), pairs[pair][1][2].cpu().numpy()])
            ax.view_init(elev,azim,roll)
            plt.show()

    def get_samplers(self):
        return self.get_sampler_init(), self.get_sampler_target()

    def get_sampler_init(self):
        return MixtureSameFamily(self.pile_sampler_init, self.distribution_init)

    def get_sampler_target(self):
        return MixtureSameFamily(self.pile_sampler_target, self.distribution_target)

    def get_samples_init(self, num_samples):
        return self.get_sampler_init().sample([num_samples])[torch.randperm(num_samples)].to(self.device)

    def get_samples_target(self, num_samples):
        return self.get_sampler_target().sample([num_samples])[torch.randperm(num_samples)].to(self.device)

    def get_samples(self, num_samples):
        return self.get_samples_init(num_samples), self.get_samples_target(num_samples)

    def get_pairs(self, num_samples):
        samples = self.get_samples(num_samples)
        return torch.stack([samples[0], samples[1]], axis=1)

    def rectified_flow(self, pairs, layers=1, hidden_num=100, batchsize=2048, iterations=10000, p_drop=0.0):
        v_model = Toy_MLP(self.input_dim, layers, hidden_num, p_drop=p_drop)
        rectified_flow = Toy_RectifiedFlow(v_model, self.device)

        optimizer = torch.optim.Adam(
            rectified_flow.v_model.parameters(), lr=5e-3)

        rectified_flow = train_toy_rectified_flow(
            rectified_flow, optimizer, pairs, batchsize, iterations)
        return rectified_flow

    def train_flows(self, num_samples, num_reflows, layers=1, hidden_num=100, batchsize=2048, iterations=10000, p_drop=0.0):
        pairs = self.get_pairs(num_samples)
        samples_init = self.get_samples_init(num_samples)
        for reflow in range(num_reflows + 1):
            self.flows.append(self.rectified_flow(
                pairs, layers, hidden_num, batchsize, iterations, p_drop=p_drop))
            del pairs
            flow = self.flows[-1].sample_ode(samples_init, num_samples)[-1]
            if reflow < num_reflows:
                pairs = torch.stack([samples_init, flow], axis=1)
            del flow
            torch.cuda.empty_cache()

    def show_flows(self, num_samples, num_connections, num_steps=100, elev=30, azim=-45, roll=1):
        rows = 1
        columns = len(self.flows) + 1
        size = (20, 4)
        dimension = self.input_dim
        if dimension == 2:
            fig, ax = plt.subplots(
                rows, columns + 1, figsize=(size), sharex=True, sharey=True)
        if dimension == 3:
            fig, ax = plt.subplots(
                rows, columns + 1, figsize=(size), sharex=True, sharey=True, subplot_kw={'projection': '3d'})

        z0, z1 = self.get_samples(num_samples)

        if dimension == 2:
            ax[0].scatter(z1[:, 0].cpu().numpy(),
                          z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
            ax[0].scatter(z0[:, 0].cpu().numpy(),
                          z0[:, 1].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)

            for pair in range(num_connections):
                ax[0].plot([z0[pair][0].cpu().numpy(), z1[pair][0].cpu().numpy()], [
                           z0[pair][1].cpu().numpy(), z1[pair][1].cpu().numpy()])

        if dimension == 3:
            ax[0].view_init(elev, azim, roll)
            ax[0].scatter(z1[:, 0].cpu().numpy(),
                          z1[:, 1].cpu().numpy(), z1[:, 2].cpu().numpy(),label=r'$\pi_1$', alpha=0.15)
            ax[0].scatter(z0[:, 0].cpu().numpy(),
                          z0[:, 1].cpu().numpy(), z0[:, 2].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)


            for pair in range(num_connections):
                ax[0].plot([z0[pair][0].cpu().numpy(), z1[pair][0].cpu().numpy()], [
                           z0[pair][1].cpu().numpy(), z1[pair][1].cpu().numpy()], [z0[pair][2].cpu().numpy(), z1[pair][2].cpu().numpy()])

        ax[0].set_title("Initial Matching")
        ax[0].legend(fontsize=10, loc="lower left")

        for column in range(1, columns):
            flow = self.flows[column - 1]
            flow.v_model.eval()
            traj = flow.sample_ode(z0=z0, num_steps=num_steps)

            if dimension == 2:
                ax[column].scatter(z1[:, 0].cpu().numpy(
                ), z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
                ax[column].scatter(z0[:, 0].cpu().numpy(
                ), z0[:, 1].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)
                ax[column].scatter(traj[-1][:, 0].cpu().numpy(), traj[-1]
                                   [:, 1].cpu().numpy(), label='Generated', alpha=0.15)

                traj_particles = torch.stack(traj)
                for i in range(num_connections):
                    ax[column].plot(traj_particles[:, i, 0].cpu(
                    ).numpy(), traj_particles[:, i, 1].cpu().numpy())

            if dimension == 3:
                ax[column].view_init(elev, azim, roll)
                ax[column].scatter(z1[:, 0].cpu().numpy(
                ), z1[:, 1].cpu().numpy(), z1[:, 2].cpu().numpy(),label=r'$\pi_1$', alpha=0.05)
                ax[column].scatter(z0[:, 0].cpu().numpy(
                ), z0[:, 1].cpu().numpy(),z0[:, 2].cpu().numpy(), label=r'$\pi_0$', alpha=0.05)
                ax[column].scatter(traj[-1][:, 0].cpu().numpy(), traj[-1]
                                   [:, 1].cpu().numpy(),traj[-1][:, 2].cpu().numpy(), label='Generated', alpha=0.2)

                traj_particles = torch.stack(traj)
                for i in range(num_connections):
                    ax[column].plot(traj_particles[:, i, 0].cpu(
                    ).numpy(), traj_particles[:, i, 1].cpu().numpy(), traj_particles[:, i, 2].cpu().numpy())

            ax[column].set_title(f"{column}-Rectified Flow")
            ax[column].legend(fontsize=10, loc="lower left")
        
        flow = self.flows[-1]
        flow.v_model.eval()
        traj = flow.reverse_sample_ode(z1=z1, num_steps=num_steps)

        if dimension == 2:
            ax[-1].scatter(z1[:, 0].cpu().numpy(),
                           z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
            ax[-1].scatter(z0[:, 0].cpu().numpy(),
                           z0[:, 1].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)
            ax[-1].scatter(traj[-1][:, 0].cpu().numpy(), traj[-1]
                           [:, 1].cpu().numpy(), label='Generated', alpha=0.15)

            traj_particles = torch.stack(traj)
            for i in range(num_connections):
                ax[-1].plot(traj_particles[:, i, 0].cpu().numpy(),
                            traj_particles[:, i, 1].cpu().numpy())

        if dimension == 3:
            ax[-1].view_init(elev, azim, roll)
            ax[-1].scatter(z1[:, 0].cpu().numpy(),
                           z1[:, 1].cpu().numpy(), z1[:, 2].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
            ax[-1].scatter(z0[:, 0].cpu().numpy(),
                           z0[:, 1].cpu().numpy(), z0[:, 2].cpu().numpy(), label=r'$\pi_0$', alpha=0.0)
            ax[-1].scatter(traj[-1][:, 0].cpu().numpy(), traj[-1]
                           [:, 1].cpu().numpy(), traj[-1][:, 2].cpu().numpy(), label='Generated', alpha=0.15)

            traj_particles = torch.stack(traj)
            for i in range(num_connections):
                ax[-1].plot(traj_particles[:, i, 0].cpu().numpy(),
                            traj_particles[:, i, 1].cpu().numpy(),
                            traj_particles[:, i, 2].cpu().numpy())

        ax[-1].set_title("Reverse Sampling")
        ax[-1].legend(fontsize=10, loc="lower left")
        plt.show()

    def show_flows_single_images(self, num_samples, num_connections, num_steps=100, elev=30, azim=-45, roll=0, x_lim_low=0, x_lim_up=5, y_lim_low=0, y_lim_up=5, equal=True):
        plt.rcParams['figure.dpi'] = 120
        plt.rcParams["figure.figsize"] = (5,5)
        rows = 1
        columns = len(self.flows) + 1
        size = (20, 4)
        dimension = self.input_dim
        # if dimension == 2:
        #     fig, ax = plt.subplots(
        #         rows, columns + 1, figsize=(size), sharex=True, sharey=True)
        # if dimension == 3:
        #     fig = plt.figure(figsize=(10, 10), dpi=80)
        #     ax = fig.add_subplot(111, projection='3d')
            # fig, ax = plt.subplots(
            #     rows, columns + 1, figsize=(size), sharex=True, sharey=True, subplot_kw={'projection': '3d'})

        z0, z1 = self.get_samples(num_samples)

        if dimension == 2:
            plt.scatter(z1[:, 0].cpu().numpy(),
                          z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
            plt.scatter(z0[:, 0].cpu().numpy(),
                          z0[:, 1].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)

            for pair in range(num_connections):
                plt.plot([z0[pair][0].cpu().numpy(), z1[pair][0].cpu().numpy()], [
                           z0[pair][1].cpu().numpy(), z1[pair][1].cpu().numpy()])

        if dimension == 3:
            fig = plt.figure(figsize=(5, 5), dpi=120)
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev, azim, roll)
            ax.scatter(z1[:, 0].cpu().numpy(),
                          z1[:, 1].cpu().numpy(), z1[:, 2].cpu().numpy(),label=r'$\pi_1$', alpha=0.15)
            ax.scatter(z0[:, 0].cpu().numpy(),
                       z0[:, 1].cpu().numpy(), z0[:, 2].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)


            for pair in range(num_connections):
                ax.plot([z0[pair][0].cpu().numpy(), z1[pair][0].cpu().numpy()], [
                           z0[pair][1].cpu().numpy(), z1[pair][1].cpu().numpy()], [z0[pair][2].cpu().numpy(), z1[pair][2].cpu().numpy()])

        if dimension == 2:
            plt.xlim(x_lim_low, x_lim_up)
            plt.ylim(y_lim_low, y_lim_up)
            plt.xticks([])
            plt.yticks([])
            plt.gca().set_axis_off()
        if dimension == 3:
            ax.axes.set_xlim3d(left=-7, right=11) 
            ax.axes.set_ylim3d(bottom=2, top=20) 
            ax.axes.set_zlim3d(bottom=-7, top=11)
            plt.gca().set_axis_off()
            #ax.set_zticks([])
        if equal:
            plt.gca().set_aspect('equal')
        plt.rcParams['figure.dpi'] = 120
        plt.savefig(f"images/{self.name}_1")
        plt.show()

        for column in range(1, columns):
            flow = self.flows[column - 1]
            flow.v_model.eval()
            traj = flow.sample_ode(z0=z0, num_steps=num_steps)

            if dimension == 2:
                plt.scatter(z1[:, 0].cpu().numpy(
                ), z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
                plt.scatter(z0[:, 0].cpu().numpy(
                ), z0[:, 1].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)
                plt.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1]
                                   [:, 1].cpu().numpy(), label='Generated', alpha=0.15)

                traj_particles = torch.stack(traj)
                for i in range(num_connections):
                    plt.plot(traj_particles[:, i, 0].cpu(
                    ).numpy(), traj_particles[:, i, 1].cpu().numpy())
                plt.xlim(x_lim_low, x_lim_up)
                plt.ylim(y_lim_low, y_lim_up)
                plt.xticks([])
                plt.yticks([])
                #plt.legend(fontsize=10, loc="lower left")
                if equal:
                    plt.gca().set_aspect('equal')
                plt.gca().set_axis_off()
                #plt.savefig(f"images/toy9_{1 + column}")
                plt.savefig(f"images/{self.name}_{1 + column}")
                plt.show()

            if dimension == 3:
                fig = plt.figure(figsize=(5, 5), dpi=120)
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(elev, azim, roll)
                ax.scatter(z1[:, 0].cpu().numpy(
                ), z1[:, 1].cpu().numpy(), z1[:, 2].cpu().numpy(),label=r'$\pi_1$', alpha=0.05)
                ax.scatter(z0[:, 0].cpu().numpy(
                ), z0[:, 1].cpu().numpy(),z0[:, 2].cpu().numpy(), label=r'$\pi_0$', alpha=0.05)
                ax.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1]
                                   [:, 1].cpu().numpy(),traj[-1][:, 2].cpu().numpy(), label='Generated', alpha=0.2)

                traj_particles = torch.stack(traj)
                for i in range(num_connections):
                    ax.plot(traj_particles[:, i, 0].cpu(
                    ).numpy(), traj_particles[:, i, 1].cpu().numpy(), traj_particles[:, i, 2].cpu().numpy())

                if equal:
                    plt.gca().set_aspect('equal')
                ax.axes.set_xlim3d(left=-7, right=11) 
                ax.axes.set_ylim3d(bottom=2, top=20) 
                ax.axes.set_zlim3d(bottom=-7, top=11)
                plt.gca().set_axis_off()
                #plt.savefig(f"images/toy9_{1 + column}")
                plt.savefig(f"images/{self.name}_{1 + column}")
                plt.show()
            # plt.set_title(f"{column}-Rectified Flow")
            # plt.legend(fontsize=10, loc="lower left")
        
        flow = self.flows[-1]
        flow.v_model.eval()
        traj = flow.reverse_sample_ode(z1=z1, num_steps=num_steps)

        if dimension == 2:
            plt.scatter(z1[:, 0].cpu().numpy(),
                           z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
            plt.scatter(z0[:, 0].cpu().numpy(),
                           z0[:, 1].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)
            plt.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1]
                           [:, 1].cpu().numpy(), label='Generated', alpha=0.15)

            traj_particles = torch.stack(traj)
            for i in range(num_connections):
                plt.plot(traj_particles[:, i, 0].cpu().numpy(),
                            traj_particles[:, i, 1].cpu().numpy())


        if dimension == 3:
            fig = plt.figure(figsize=(5, 5), dpi=120)
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev, azim, roll)
            ax.scatter(z1[:, 0].cpu().numpy(),
                           z1[:, 1].cpu().numpy(), z1[:, 2].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
            ax.scatter(z0[:, 0].cpu().numpy(),
                           z0[:, 1].cpu().numpy(), z0[:, 2].cpu().numpy(), label=r'$\pi_0$', alpha=0.0)
            ax.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1]
                           [:, 1].cpu().numpy(), traj[-1][:, 2].cpu().numpy(), label='Generated', alpha=0.15)

            traj_particles = torch.stack(traj)
            for i in range(num_connections):
                ax.plot(traj_particles[:, i, 0].cpu().numpy(),
                            traj_particles[:, i, 1].cpu().numpy(),
                            traj_particles[:, i, 2].cpu().numpy())

        # plt.title("Reverse Sampling")
        plt.xlim(x_lim_low, x_lim_up)
        plt.ylim(y_lim_low, y_lim_up)
        plt.xticks([])
        plt.yticks([])
        if dimension == 3:
            ax.axes.set_xlim3d(left=-7, right=11) 
            ax.axes.set_ylim3d(bottom=2, top=20) 
            ax.axes.set_zlim3d(bottom=-7, top=11)
            plt.gca().set_axis_off()
        if equal:
            plt.gca().set_aspect('equal')
        plt.gca().set_axis_off()
        #plt.legend(fontsize=10, loc="lower left")
        #plt.savefig(f"images/toy9_{4}")
        plt.savefig(f"images/{self.name}_4")
        plt.show()
  

    def show_zones(self, num_samples, num_steps=100, elev=30, azim=-45, roll=0):
        dimension = self.input_dim
        z1 = self.get_samples_target(num_samples)
        if dimension == 2:
            fig, ax = plt.subplots(
                    1, 1, sharex=True, sharey=True)
            ax.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
        if dimension == 3:
            fig, ax = plt.subplots(
                    1, 1, sharex=True, sharey=True, subplot_kw={'projection': '3d'})
            ax.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(),  z1[:, 2].cpu().numpy(),label=r'$\pi_1$', alpha=0.00)
        for i in range(self.num_piles_init):
            p = np.zeros([self.num_piles_init])
            p[i] = 1
            pile_sampler = Categorical(torch.tensor(p))

            z0 = MixtureSameFamily(pile_sampler, self.distribution_init).sample([int(num_samples / self.num_piles_init)]).to(self.device)
            traj = self.flows[-1].sample_ode(z0=z0, num_steps=num_steps)
            if self.input_dim == 2:
                ax.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), label='Generated', alpha=0.3)
            if self.input_dim == 3:
                ax.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), traj[-1][:, 2].cpu().numpy(), label='Generated', alpha=0.4)
                ax.view_init(elev,azim,roll)

        plt.axis('scaled')
        #ax.legend(fontsize=10, loc="lower left")
        ax.set_title('Distribution')
        plt.show()

    def show_and_save_zones(self, num_samples, num_steps=100, elev=30, azim=-45, roll=0, x_lim_low=0, x_lim_up=5, y_lim_low=0, y_lim_up=5):
        plt.rcParams['figure.dpi'] = 120
        #plt.rcParams["figure.figsize"] = (9,5)
        dimension = self.input_dim
        z1 = self.get_samples_target(num_samples)
        if dimension == 2:
            fig, ax = plt.subplots(
                    1, 1, sharex=True, sharey=True)
            ax.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
        if dimension == 3:
            fig, ax = plt.subplots(
                    1, 1, sharex=True, sharey=True, subplot_kw={'projection': '3d'})
            ax.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(),  z1[:, 2].cpu().numpy(),label=r'$\pi_1$', alpha=0.00)
        fig.set_size_inches(5,5)
        for i in range(self.num_piles_init):
            p = np.zeros([self.num_piles_init])
            p[i] = 1
            pile_sampler = Categorical(torch.tensor(p))

            z0 = MixtureSameFamily(pile_sampler, self.distribution_init).sample([int(num_samples / self.num_piles_init)]).to(self.device)
            traj = self.flows[-1].sample_ode(z0=z0, num_steps=num_steps)
            if self.input_dim == 2:
                ax.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), label='Generated', alpha=0.3)
            if self.input_dim == 3:
                ax.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), traj[-1][:, 2].cpu().numpy(), label='Generated', alpha=0.4)
                ax.view_init(elev,azim,roll)

        plt.axis('scaled')
        plt.xlim(x_lim_low, x_lim_up)
        plt.ylim(y_lim_low, y_lim_up)
        #ax.legend(fontsize=10, loc="lower left")
        plt.gca().set_aspect('equal')
        plt.gca().set_axis_off()
        plt.rcParams['figure.dpi'] = 120
        plt.savefig(f"images/{self.name}_distr")
        #ax.set_title('Distribution')
        plt.show()

    def train_distill(self, num_samples = 10000, layers=1, hidden_num=100, batchsize=2048, iterations=10000, num_steps=100, p_drop=0.0):
        distill_model = Toy_MLP_distill(self.input_dim, layers, hidden_num, p_drop=p_drop).to(self.device)

        z0 = self.get_samples_init(num_samples).to(self.device)

        flow = self.flows[-1]
        flow.v_model.eval()
        z1 = flow.sample_ode(z0=z0, num_steps=num_steps)[-1]

        pairs = torch.stack([z0, z1], axis=1)

        optimizer = torch.optim.Adam(
            distill_model.parameters(), lr=5e-3)
        for i in range(iterations + 1):
            optimizer.zero_grad()
            indeces = torch.randperm(len(pairs))[:batchsize]
            batch = pairs[indeces]
            z0 = batch[:, 0].detach().clone().to(self.device)
            z1 = batch[:, 1].detach().clone().to(self.device)

            pred = distill_model(z0)
            loss = (z1 - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
            loss = loss.mean()
            loss.backward()

            optimizer.step()

        self.distill = distill_model

        return distill_model

    def show_distill(self, num_samples):
        distill = self.distill
        distill.eval()

        z0 = self.get_samples_init(num_samples).to(self.device)
        z1 = self. get_samples_target(num_samples).to(self.device)
        generated = distill(z0)

        plt.scatter(z0[:, 0].cpu().numpy(), z0[:, 1].cpu().numpy(), alpha=0.15)
        plt.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), alpha=0.15)
        plt.scatter(generated[:, 0].detach().cpu().numpy(), generated[:, 1].detach().cpu().numpy(), alpha=0.15)
        plt.show()

    def show_and_save_distill(self, num_samples, num_steps=100, elev=30, azim=-45, roll=0, x_lim_low=0, x_lim_up=5, y_lim_low=0, y_lim_up=5):
        plt.rcParams['figure.dpi'] = 120
        #plt.rcParams["figure.figsize"] = (9,5)
        dimension = self.input_dim

        distill = self.distill
        distill.eval()

        z0 = self.get_samples_init(num_samples).to(self.device)
        z1 = self. get_samples_target(num_samples).to(self.device)
        generated = distill(z0)

        if dimension == 2:
            fig, ax = plt.subplots(
                    1, 1, sharex=True, sharey=True)
            ax.scatter(z0[:, 0].cpu().numpy(), z0[:, 1].cpu().numpy(), alpha=0.15)
            ax.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), alpha=0.15)
            ax.scatter(generated[:, 0].detach().cpu().numpy(), generated[:, 1].detach().cpu().numpy(), alpha=0.15)
        if dimension == 3:
            fig, ax = plt.subplots(
                    1, 1, sharex=True, sharey=True, subplot_kw={'projection': '3d'})
            ax.scatter(z0[:, 0].cpu().numpy(), z0[:, 1].cpu().numpy(), z0[:, 2].cpu().numpy())
            ax.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), z1[:, 2].cpu().numpy())
            ax.scatter(generated[:, 0].detach().cpu().numpy(), generated[:, 1].detach().cpu().numpy(), generated[:, 2].detach().cpu().numpy())
        fig.set_size_inches(5,5)
        
        plt.axis('scaled')
        plt.xlim(x_lim_low, x_lim_up)
        plt.ylim(y_lim_low, y_lim_up)
        #ax.legend(fontsize=10, loc="lower left")
        plt.gca().set_aspect('equal')
        plt.gca().set_axis_off()
        plt.rcParams['figure.dpi'] = 120
        plt.savefig(f"images/{self.name}_distill")
        #ax.set_title('Distribution')
        plt.show()

    def show_field(self, num_samples, num_steps=100, x_lim_low=0, x_lim_up=5, y_lim_low=0, y_lim_up=5):
        plt.rcParams['figure.dpi'] = 300
        #plt.rcParams["figure.figsize"] = (9,5)
        dimension = self.input_dim
        
        x, y = np.meshgrid(np.linspace(1,7,20),np.linspace(1,7,20))
        z0 = torch.tensor(np.concatenate([x.reshape(-1,1),y.reshape(-1,1)], axis=1)).float().to(self.device)
        t = np.linspace(0, 1, 20)
        t = torch.tensor(t).float().to(self.device)
        t = t.repeat(20, 1).flatten()[:, None]
        print(t.shape, z0.shape)
        traj = self.flows[-2].v_model(z0, t).detach().cpu().numpy()

        plt.quiver(x, y, traj[:,0], traj[:,1], color='r', alpha=0.5)

        x, y = np.linspace(2.5, 5.2, 9),np.array([5.33, 5, 4.66,4.33, 4.0, 4.0, 4.33, 4.66, 5])
        print(x.shape)
        z0 = torch.tensor(np.concatenate([x.reshape(-1,1),y.reshape(-1,1)], axis=1)).float().to(self.device)
        t = np.linspace(0.1, 0.9, 9)
        t = torch.tensor(t).float().to(self.device)
        t = t.repeat(1, 1).flatten()[:, None]
        print(t.shape, z0.shape)
        traj = self.flows[-2].v_model(z0, t).detach().cpu().numpy()

        plt.quiver(x, y, traj[:,0], traj[:,1], color='b', alpha=0.8)

        x, y = np.linspace(2.9, 4.5, 3),np.array([1.8,1.8, 1.8])
        print(x.shape)
        z0 = torch.tensor(np.concatenate([x.reshape(-1,1),y.reshape(-1,1)], axis=1)).float().to(self.device)
        t = np.linspace(0.3, 0.7, 3)
        t = torch.tensor(t).float().to(self.device)
        t = t.repeat(1, 1).flatten()[:, None]
        print(t.shape, z0.shape)
        traj = self.flows[-2].v_model(z0, t).detach().cpu().numpy()

        plt.quiver(x, y, traj[:,0], traj[:,1], color='b', alpha=0.8, scale=50)

        z0 = self.get_samples_init(num_samples).to(self.device)
        z1 = self. get_samples_target(num_samples).to(self.device)
        plt.scatter(z0[:, 0].cpu().numpy(), z0[:, 1].cpu().numpy(), alpha=0.15)
        plt.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), alpha=0.15)

        plt.axis('scaled')
        # plt.xlim(x_lim_low, x_lim_up)
        # plt.ylim(y_lim_low, y_lim_up)
        #ax.legend(fontsize=10, loc="lower left")
        plt.gca().set_aspect('equal')
        plt.gca().set_axis_off()
        plt.rcParams['figure.dpi'] = 120
        plt.savefig(f"images/{self.name}_distr")
        #ax.set_title('Distribution')
        plt.show()

#### configurable Toy_problems with different data distributions, inheriting from Toy_problem class ####

class Toy_problem1(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 2
        self.num_piles_target = 2
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/self.num_piles_init for i in range(self.num_piles_init)]))
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_init = MultivariateNormal(torch.tensor(
            [[2, 2], [2, 6]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2)]))
        self.distribution_target = MultivariateNormal(torch.tensor(
            [[6, 2], [6, 6]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2)]))

        self.info = {"distribution_init": "gaussian", "piles_init": 2,
                     "distribution_target": "gaussian", "piles_target": 2, "formation": "square"}
        # add gaussian params to info / and parameter?


class Toy_problem2(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 3
        self.num_piles_target = 3
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/self.num_piles_init for i in range(self.num_piles_init)]))
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_init = MultivariateNormal(torch.tensor([[10 * np.sqrt(3) / 2., 5], [-10 * np.sqrt(
            3) / 2., 5], [0.0, -10 * np.sqrt(3) / 2]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2), torch.eye(2)]))
        self.distribution_target = MultivariateNormal(torch.tensor([[10 * np.sqrt(3) / 2., -5], [-10 * np.sqrt(
            3) / 2., -5], [0.0, 10 * np.sqrt(3) / 2]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2), torch.eye(2)]))


class Toy_problem3(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 6
        self.num_piles_target = 6
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/self.num_piles_init for i in range(self.num_piles_init)]))
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_init = MultivariateNormal(torch.tensor([[5, 0], [5 * np.cos(2 * np.pi / self.num_piles_init), 5 * np.sin(2 * np.pi / self.num_piles_init)], [5 * np.cos(4 * np.pi / self.num_piles_init), 5 * np.sin(4 * np.pi / self.num_piles_init)], [5 * np.cos(6 * np.pi / self.num_piles_init), 5 * np.sin(
            6 * np.pi / self.num_piles_init)], [5 * np.cos(8 * np.pi / self.num_piles_init), 5 * np.sin(8 * np.pi / self.num_piles_init)], [5 * np.cos(10 * np.pi / self.num_piles_init), 5 * np.sin(10 * np.pi / self.num_piles_init)]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2), torch.eye(2), torch.eye(2), torch.eye(2), torch.eye(2)]))
        self.distribution_target = MultivariateNormal(torch.tensor([[10, 0], [10 * np.cos(2 * np.pi / self.num_piles_target), 10 * np.sin(2 * np.pi / self.num_piles_target)], [10 * np.cos(4 * np.pi / self.num_piles_target), 10 * np.sin(4 * np.pi / self.num_piles_target)], [10 * np.cos(6 * np.pi / self.num_piles_target), 10 * np.sin(
            6 * np.pi / self.num_piles_target)], [10 * np.cos(8 * np.pi / self.num_piles_target), 10 * np.sin(8 * np.pi / self.num_piles_target)], [10 * np.cos(10 * np.pi / self.num_piles_target), 10 * np.sin(10 * np.pi / self.num_piles_target)]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2), torch.eye(2), torch.eye(2), torch.eye(2), torch.eye(2)]))


class Toy_problem4(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 6
        self.num_piles_target = 1
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/self.num_piles_init for i in range(self.num_piles_init)]))
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_init = MultivariateNormal(torch.tensor([[5, 0], [5 * np.cos(2 * np.pi / self.num_piles_init), 5 * np.sin(2 * np.pi / self.num_piles_init)], [5 * np.cos(4 * np.pi / self.num_piles_init), 5 * np.sin(4 * np.pi / self.num_piles_init)], [5 * np.cos(6 * np.pi / self.num_piles_init), 5 * np.sin(
            6 * np.pi / self.num_piles_init)], [5 * np.cos(8 * np.pi / self.num_piles_init), 5 * np.sin(8 * np.pi / self.num_piles_init)], [5 * np.cos(10 * np.pi / self.num_piles_init), 5 * np.sin(10 * np.pi / self.num_piles_init)]]).float(), 0.1*torch.stack([torch.eye(2) for i in range(self.num_piles_init)]))
        self.distribution_target = MultivariateNormal(
            torch.tensor([[0, 0]]).float(), 0.1*torch.stack([torch.eye(2)]))


class Toy_problem5(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 6
        self.num_piles_target = 1
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/self.num_piles_init for i in range(self.num_piles_init)]))
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_init = MultivariateNormal(torch.tensor([[2, 4], [2, 6], [2, 8], [2, 10], [
                                         2, 12], [2, 14]]).float(), 0.1*torch.stack([torch.eye(2) for i in range(self.num_piles_init)]))
        self.distribution_target = MultivariateNormal(torch.tensor(
            [[6, 9]]).float(), 0.1*torch.stack([torch.eye(2)]))


class Toy_problem6(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 1
        self.num_piles_target = 1

    def get_sampler_init(self, num_samples):
        assert False

    def get_sampler_target(self, num_samples):
        assert False

    def get_samples_init(self, num_samples):
        samples, label = make_moons(num_samples * 2, shuffle=False, noise=0.05)
        samples_init = torch.tensor(samples[:num_samples]).float()
        indices = torch.randperm(num_samples)
        samples_init = samples_init[indices].to(self.device)

        return samples_init

    def get_samples_target(self, num_samples):
        samples, label = make_moons(num_samples * 2, shuffle=False, noise=0.05)
        samples_target = torch.tensor(samples[num_samples:]).float()
        indices = torch.randperm(num_samples)
        samples_target = samples_target[indices].to(self.device)

        return samples_target

    def show_zones(self, num_samples, num_steps=100):
        assert False


class Toy_problem7(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 2
        self.num_piles_target = 2
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/self.num_piles_init for i in range(self.num_piles_init)]))
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_init = MultivariateNormal(torch.tensor(
            [[2, 2], [2, 6]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2)]))
        self.distribution_target = MultivariateNormal(torch.tensor(
            [[2, 2], [2, 6]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2)]))

class Toy_problem8(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 1
        self.num_piles_target = 6
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/self.num_piles_init for i in range(self.num_piles_init)]))
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_target = MultivariateNormal(torch.tensor([[5, 0], [5 * np.cos(2 * np.pi / self.num_piles_target), 5 * np.sin(2 * np.pi / self.num_piles_target)], [5 * np.cos(4 * np.pi / self.num_piles_target), 5 * np.sin(4 * np.pi / self.num_piles_target)], [5 * np.cos(6 * np.pi / self.num_piles_target), 5 * np.sin(
            6 * np.pi / self.num_piles_target)], [5 * np.cos(8 * np.pi / self.num_piles_target), 5 * np.sin(8 * np.pi / self.num_piles_target)], [5 * np.cos(10 * np.pi / self.num_piles_target), 5 * np.sin(10 * np.pi / self.num_piles_target)]]).float(), 0.1*torch.stack([torch.eye(2) for i in range(self.num_piles_target)]))
        self.distribution_init = MultivariateNormal(
            torch.tensor([[0, 0]]).float(), 0.1*torch.stack([torch.eye(2)]))

class Toy_problem9(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 2
        self.num_piles_target = 1
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_target = MultivariateNormal(
            torch.tensor([[0.5, -2]]).float(), 0.1*torch.stack([torch.eye(2)]))

    def get_sampler_init(self, num_samples):
        assert False

    def get_samples_init(self, num_samples):
        samples, label = make_moons(num_samples, shuffle=False, noise=0.05)
        indices = torch.randperm(num_samples)
        samples_init = torch.tensor(samples[indices]).float().to(self.device)

        return samples_init
    
    def get_samples_init_moon1(self, num_samples):
        samples, label = make_moons(num_samples*2, shuffle=False, noise=0.05)
        samples = torch.tensor(samples[:num_samples]).float().to(self.device)

        return samples

    def get_samples_init_moon2(self, num_samples):
        samples, label = make_moons(num_samples*2, shuffle=False, noise=0.05)
        samples = torch.tensor(samples[num_samples:]).float().to(self.device)

        return samples

    def show_zones(self, num_samples, num_steps=100, elev=30, azim=-45, roll=0):
        dimension = self.input_dim
        z1 = self.get_samples_target(num_samples)
        if dimension == 2:
            fig, ax = plt.subplots(
                    1, 1, sharex=True, sharey=True)
            ax.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
        if dimension == 3:
            fig, ax = plt.subplots(
                    1, 1, sharex=True, sharey=True, subplot_kw={'projection': '3d'})
            ax.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(),  z1[:, 2].cpu().numpy(),label=r'$\pi_1$', alpha=0.00)
        for i in range(self.num_piles_init):
            p = np.zeros([self.num_piles_init])
            p[i] = 1
            pile_sampler = Categorical(torch.tensor(p))

            z0 = torch.tensor(make_moons(num_samples*2, shuffle=False, noise=0.05)[0][i*num_samples:(i+1)*num_samples]).float().to(self.device)
            traj = self.flows[-1].sample_ode(z0=z0, num_steps=num_steps)
            if self.input_dim == 2:
                ax.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), label='Generated', alpha=0.3)
            if self.input_dim == 3:
                ax.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), traj[-1][:, 2].cpu().numpy(), label='Generated', alpha=0.4)
                ax.view_init(elev,azim,roll)

        plt.axis('scaled')
        #ax.legend(fontsize=10, loc="lower left")
        ax.set_title('Distribution')
        plt.show()

    def show_field(self, num_samples, num_steps=100, x_lim_low=0, x_lim_up=5, y_lim_low=0, y_lim_up=5):
        plt.rcParams['figure.dpi'] = 120
        #plt.rcParams["figure.figsize"] = (9,5)
        dimension = self.input_dim
        
        x, y = np.meshgrid(np.linspace(2,6,10),np.linspace(2,6,10))
        z0 = torch.tensor(np.concatenate([x.reshape(-1,1),y.reshape(-1,1)], axis=1)).float().to(self.device)
        t = np.linspace(0, 1, 10)
        traj = self.flows[-1].v_model(z0, t)

        plt.quiver(x, y, traj[:,0].cpu().numpy(), traj[:,1].cpu().numpy(), color='r', alpha=0.5)

        plt.axis('scaled')
        # plt.xlim(x_lim_low, x_lim_up)
        # plt.ylim(y_lim_low, y_lim_up)
        #ax.legend(fontsize=10, loc="lower left")
        plt.gca().set_aspect('equal')
        plt.gca().set_axis_off()
        plt.rcParams['figure.dpi'] = 120
        plt.savefig(f"images/{self.name}_distr")
        #ax.set_title('Distribution')
        plt.show()

class Toy_problem10(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 1
        self.num_piles_target = 2
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/self.num_piles_init for i in range(self.num_piles_init)]))
        self.distribution_init = MultivariateNormal(
            torch.tensor([[0.5, -2]]).float(), 0.1*torch.stack([torch.eye(2)]))

    def get_sampler_target(self, num_samples):
        assert False

    def get_samples_target(self, num_samples):
        samples, label = make_moons(num_samples, shuffle=False, noise=0.05)
        indices = torch.randperm(num_samples)
        samples_target = torch.tensor(samples[indices]).float().to(self.device)

        return samples_target

    def show_zones(self, num_samples, num_steps=100):
        assert False

class Toy_problem11(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 1
        self.num_piles_target = 1

    def get_sampler_init(self, num_samples):
        assert False

    def get_sampler_target(self, num_samples):
        assert False

    def get_samples_init(self, num_samples):
        samples, label = make_circles(num_samples * 2, shuffle=False, noise=0.05)
        samples_init = torch.tensor(samples[:num_samples]).float()
        indices = torch.randperm(num_samples)
        samples_init = samples_init[indices].to(self.device)

        return samples_init

    def get_samples_target(self, num_samples):
        samples, label = make_circles(num_samples * 2, shuffle=False, noise=0.05)
        samples_target = torch.tensor(samples[num_samples:]).float()
        indices = torch.randperm(num_samples)
        samples_target = samples_target[indices].to(self.device)

        return samples_target

    def show_zones(self, num_samples, num_steps=100):
        assert False

class Toy_problem12(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 1
        self.num_piles_target = 2
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/self.num_piles_init for i in range(self.num_piles_init)]))
        self.distribution_init = MultivariateNormal(
            torch.tensor([[0, 0]]).float(), 0.05*torch.stack([torch.eye(2)]))

    def get_sampler_target(self, num_samples):
        assert False

    def get_samples_target(self, num_samples):
        samples, label = make_circles(num_samples, shuffle=False, noise=0.05)
        indices = torch.randperm(num_samples)
        samples_target = torch.tensor(samples[indices]).float().to(self.device)

        return samples_target

    def show_zones(self, num_samples, num_steps=100):
        assert False

class Toy_problem13(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 2
        self.num_piles_target = 1
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_target = MultivariateNormal(
            torch.tensor([[0, 0]]).float(), 0.05*torch.stack([torch.eye(2)]))

    def get_sampler_init(self, num_samples):
        assert False

    def get_samples_init(self, num_samples):
        samples, label = make_circles(num_samples, shuffle=False, noise=0.037)
        indices = torch.randperm(num_samples)
        samples_init = torch.tensor(samples[indices]).float().to(self.device)

        return samples_init
    
    def get_samples_init_moon1(self, num_samples):
        samples, label = make_circles(num_samples*2, shuffle=False, noise=0.037)
        samples = torch.tensor(samples[:num_samples]).float().to(self.device)

        return samples

    def get_samples_init_moon2(self, num_samples):
        samples, label = make_circles(num_samples*2, shuffle=False, noise=0.037)
        samples = torch.tensor(samples[num_samples:]).float().to(self.device)

        return samples

    def show_zones(self, num_samples, num_steps=100):
        assert False

    def show_and_save_zones(self, num_samples, num_steps=100, x_lim_low=0, x_lim_up=5, y_lim_low=0, y_lim_up=5):
        plt.rcParams['figure.dpi'] = 120
        #plt.rcParams["figure.figsize"] = (9,5)
        dimension = self.input_dim
        z1 = self.get_samples_target(num_samples)
        if dimension == 2:
            fig, ax = plt.subplots(
                    1, 1, sharex=True, sharey=True)
            ax.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
        
        fig.set_size_inches(5,5)
        

        z0 = self.get_samples_init_moon1(num_samples=int(num_samples/2))
        traj = self.flows[-2].sample_ode(z0=z0, num_steps=num_steps)
        ax.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), label='Generated', alpha=0.3)

        z0 = self.get_samples_init_moon2(num_samples=int(num_samples/2))
        traj = self.flows[-2].sample_ode(z0=z0, num_steps=num_steps)
        ax.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), label='Generated', alpha=0.3)

        plt.axis('scaled')
        plt.xlim(x_lim_low, x_lim_up)
        plt.ylim(y_lim_low, y_lim_up)
        #ax.legend(fontsize=10, loc="lower left")
        plt.gca().set_aspect('equal')
        plt.gca().set_axis_off()
        plt.rcParams['figure.dpi'] = 120
        plt.savefig(f"images/{self.name}_distr")
        #ax.set_title('Distribution')
        plt.show()

class Toy_problem14(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 2
        self.num_piles_target = 2
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/self.num_piles_init for i in range(self.num_piles_init)]))
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_target = MultivariateNormal(torch.tensor([[2,2],[14,2]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2)]))
        self.distribution_init = MultivariateNormal(torch.tensor([[6,2],[10,2]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2)]))

class Toy_problem15(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 2
        self.num_piles_target = 2
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/self.num_piles_init for i in range(self.num_piles_init)]))
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_target = MultivariateNormal(torch.tensor([[2,2],[10,2]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2)]))
        self.distribution_init = MultivariateNormal(torch.tensor([[6,2],[14,2]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2)]))

class Toy_problem16(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 3
        self.num_piles_init = 1
        self.num_piles_target = 1
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/self.num_piles_init for i in range(self.num_piles_init)]))
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_init = MultivariateNormal(torch.tensor([[0,0,0]]).float(), 0.1*torch.stack([torch.eye(3)]))
        self.distribution_target = MultivariateNormal(torch.tensor([[2,2,2]]).float(), 0.1*torch.stack([torch.eye(3)]))

class Toy_problem17(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 3
        self.num_piles_init = 2
        self.num_piles_target = 2
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/self.num_piles_init for i in range(self.num_piles_init)]))
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_init = MultivariateNormal(torch.tensor([[0,0,0], [-1,1,1]]).float(), 0.05*torch.stack([torch.eye(3) for i in range(2)]))
        self.distribution_target = MultivariateNormal(torch.tensor([[0,2,0], [1,1,1]]).float(), 0.05*torch.stack([torch.eye(3) for i in range(2)]))

class Toy_problem18(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 3
        self.num_piles_init = 1
        self.num_piles_target = 1
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/self.num_piles_init for i in range(self.num_piles_init)]))
        self.distribution_init = MultivariateNormal(torch.tensor([[0,10,0]]).float(), 15.0*torch.stack([torch.eye(3)]))

    def get_sampler_target(self, num_samples):
        assert False

    def get_samples_target(self, num_samples):
        samples_target = torch.tensor(make_swiss_roll(num_samples, noise=0.1)[0]).float().to(self.device)
        return samples_target

    def show_zones(self, num_samples, num_steps=100):
        assert False

class Toy_problem19(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 2
        self.num_piles_target = 1
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/self.num_piles_init for i in range(self.num_piles_init)]))
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_init = MultivariateNormal(torch.tensor([[2,2],[2,6]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2)]))
        self.distribution_target = MultivariateNormal(torch.tensor([[6,4]]).float(), 0.1*torch.stack([torch.eye(2)]))

class Toy_problem20(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 4
        self.num_piles_target = 1
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/self.num_piles_init for i in range(self.num_piles_init)]))
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_init = MultivariateNormal(torch.tensor([[2,2],[2,6], [0,4], [4,4]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2), torch.eye(2), torch.eye(2)]))
        self.distribution_target = MultivariateNormal(torch.tensor([[8,4]]).float(), 0.1*torch.stack([torch.eye(2)]))

class Toy_problem21(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 4
        self.num_piles_target = 1
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/self.num_piles_init for i in range(self.num_piles_init)]))
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_init = MultivariateNormal(torch.tensor([[2,2],[1.5,3.5],[2,3], [3.5,3.5]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2), torch.eye(2), torch.eye(2)]))
        self.distribution_target = MultivariateNormal(torch.tensor([[4,1.5]]).float(), 0.1*torch.stack([torch.eye(2)]))

class Toy_problem22(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 3
        self.num_piles_init = 4
        self.num_piles_target = 1
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/self.num_piles_init for i in range(self.num_piles_init)]))
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_init = MultivariateNormal(torch.tensor([[0,0,0],[4,0,0],[2,0.866 * 4,0],[2,0.866 * 2,0.866 * 4]]).float(), 0.1*torch.stack([torch.eye(3) for i in range(self.num_piles_init)]))
        self.distribution_target = MultivariateNormal(torch.tensor([[2,0.866 * 2,0.866*2]]).float(), 0.1*torch.stack([torch.eye(3)]))

class Toy_problem23(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 3
        self.num_piles_init = 2
        self.num_piles_target = 1
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/self.num_piles_init for i in range(self.num_piles_init)]))
        self.distribution_init = MultivariateNormal(torch.tensor([[-15,10,0],[15,10,0],]).float(), 0.5*torch.stack([torch.eye(3) for i in range(self.num_piles_init)]))

    def get_sampler_target(self, num_samples):
        assert False

    def get_samples_target(self, num_samples):
        samples_target = torch.tensor(make_swiss_roll(num_samples, noise=0.1)[0]).float().to(self.device)
        return samples_target

    def show_zones(self, num_samples, num_steps=100):
        assert False

class Toy_problem24(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 6
        self.num_piles_target = 1
        self.pile_sampler_init = Categorical(torch.tensor(
            [1/10, 1/10, 1/2, 1/10, 1/10, 1/10]))
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_init = MultivariateNormal(torch.tensor([[2, 4], [2, 6], [2, 8], [2, 10], [
                                         2, 12], [2, 14]]).float(), 0.1*torch.stack([torch.eye(2) for i in range(self.num_piles_init)]))
        self.distribution_target = MultivariateNormal(torch.tensor(
            [[6, 9]]).float(), 0.1*torch.stack([torch.eye(2)]))

class Toy_problem25(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 6
        self.num_piles_target = 1
        self.pile_sampler_init = Categorical(torch.tensor([1/2, 1/10, 1/10, 1/10, 1/10, 1/10]))
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_init = MultivariateNormal(torch.tensor([[5, 0], [5 * np.cos(2 * np.pi / self.num_piles_init), 5 * np.sin(2 * np.pi / self.num_piles_init)], [5 * np.cos(4 * np.pi / self.num_piles_init), 5 * np.sin(4 * np.pi / self.num_piles_init)], [5 * np.cos(6 * np.pi / self.num_piles_init), 5 * np.sin(
            6 * np.pi / self.num_piles_init)], [5 * np.cos(8 * np.pi / self.num_piles_init), 5 * np.sin(8 * np.pi / self.num_piles_init)], [5 * np.cos(10 * np.pi / self.num_piles_init), 5 * np.sin(10 * np.pi / self.num_piles_init)]]).float(), 0.1*torch.stack([torch.eye(2) for i in range(self.num_piles_init)]))
        self.distribution_target = MultivariateNormal(
            torch.tensor([[0, 0]]).float(), 0.1*torch.stack([torch.eye(2)]))

class Toy_problem26(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        self.num_piles_init = 6
        self.num_piles_target = 1
        self.pile_sampler_init = Categorical(torch.tensor([1/4, 1/8, 1/8, 1/6, 1/6, 1/6]))
        self.pile_sampler_target = Categorical(torch.tensor(
            [1/self.num_piles_target for i in range(self.num_piles_target)]))
        self.distribution_init = MultivariateNormal(torch.tensor([[5, 0], [5 * np.cos(2 * np.pi / self.num_piles_init), 5 * np.sin(2 * np.pi / self.num_piles_init)], [5 * np.cos(4 * np.pi / self.num_piles_init), 5 * np.sin(4 * np.pi / self.num_piles_init)], [5 * np.cos(6 * np.pi / self.num_piles_init), 5 * np.sin(
            6 * np.pi / self.num_piles_init)], [5 * np.cos(8 * np.pi / self.num_piles_init), 5 * np.sin(8 * np.pi / self.num_piles_init)], [5 * np.cos(10 * np.pi / self.num_piles_init), 5 * np.sin(10 * np.pi / self.num_piles_init)]]).float(), 0.1*torch.stack([torch.eye(2) for i in range(self.num_piles_init)]))
        self.distribution_target = MultivariateNormal(
            torch.tensor([[0, 0]]).float(), 0.1*torch.stack([torch.eye(2)]))