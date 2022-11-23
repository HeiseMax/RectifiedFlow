import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily

from RectifiedFlow import Toy_RectifiedFlow, train_toy_rectified_flow
from models import Toy_MLP

from sklearn.datasets import make_moons

### parent class for toy problem
class Toy_problem():
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.name = type(self).__name__

        self.input_dim = 1
        self.sampler_init = MultivariateNormal(torch.tensor([[[0,0]]]).float(), 0.1*torch.stack([torch.eye(2)]))
        self.sampler_target = MultivariateNormal(torch.tensor([[[2,0]]]).float(), 0.1*torch.stack([torch.eye(2)]))

        self.info = "No info provided!"
        # add gaussian params to info / and parameter?

        self.flows = []

    def save(self):
        for i, flow in enumerate(self.flows):
            torch.save(flow, f"models/ToyExamples/{self.name}_flow_{i}.pth")

    def load(self):
        flows = []
        for i in range(10):
            fname = f"models/ToyExamples/{self.name}_flow_{i}.pth"
            if os.path.isfile(fname):
                flows.append(torch.load(fname))
            else:
                break
        self.flows = flows

    def get_info(self):
        return self.info

    def show_distribution(self, num_samples = 1000, show_pairs = False, num_pairs=100):
        pairs = self.get_pairs(num_samples)
        samples_init = pairs[:, 0]
        samples_target = pairs[:, 1]
        plt.scatter(samples_init[:,0].cpu().numpy(), samples_init[:,1].cpu().numpy())
        plt.scatter(samples_target[:,0].cpu().numpy(), samples_target[:,1].cpu().numpy())

        if show_pairs:
            for pair in range(num_pairs):
                plt.plot([pairs[pair][0][0].cpu().numpy(), pairs[pair][1][0].cpu().numpy()], [pairs[pair][0][1].cpu().numpy(), pairs[pair][1][1].cpu().numpy()])
        plt.show()

    def get_samplers(self):
        return self.sampler_init, self.sampler_target

    def get_sampler_init(self):
        return self.sampler_init

    def get_sampler_target(self):
        return self.sampler_target

    def get_samples_init(self, num_samples):
        return self.sampler_init.sample([num_samples])[torch.randperm(num_samples)].to(self.device)
    
    def get_samples_target(self, num_samples):
        return self.sampler_target.sample([num_samples])[torch.randperm(num_samples)].to(self.device)

    def get_samples(self, num_samples):
        return self.get_samples_init(num_samples), self.get_samples_target(num_samples)

    def get_pairs(self, num_samples):
        samples = self.get_samples(num_samples)
        return torch.stack([samples[0], samples[1]], axis=1)

    def rectified_flow(self, pairs, layers=1, hidden_num=100, batchsize=2048, iterations=10000):
        v_model = Toy_MLP(self.input_dim, layers, hidden_num)
        rectified_flow = Toy_RectifiedFlow(v_model, self.device)

        optimizer = torch.optim.Adam(rectified_flow.v_model.parameters(), lr=5e-3)

        rectified_flow = train_toy_rectified_flow(rectified_flow, optimizer, pairs, batchsize, iterations)
        return rectified_flow

    def train_flows(self, num_samples, num_reflows, layers=1, hidden_num=100, batchsize=2048, iterations=10000):
        pairs = self.get_pairs(num_samples)
        samples_init = self.get_samples_init(num_samples)
        for reflow in range(num_reflows + 1):
            self.flows.append(self.rectified_flow(pairs, layers, hidden_num, batchsize, iterations))
            del pairs
            flow = self.flows[-1].sample_ode(samples_init, num_samples)[-1]
            if reflow < num_reflows:
                pairs = torch.stack([samples_init, flow], axis=1)
            del flow
            torch.cuda.empty_cache()

    def show_flows(self, num_samples, num_connections, num_steps=100):        
        rows = 1
        columns = len(self.flows) + 1
        size = (20, 4)
        fig, ax = plt.subplots(rows, columns + 1, figsize=(size), sharex=True, sharey=True)
        dimension = self.input_dim

        z0, z1 = self.get_samples(num_samples)

        if dimension == 2:
            ax[0].scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
            ax[0].scatter(z0[:, 0].cpu().numpy(), z0[:, 1].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)

            for pair in range(num_connections):
                ax[0].plot([z0[pair][0].cpu().numpy(), z1[pair][0].cpu().numpy()], [z0[pair][1].cpu().numpy(), z1[pair][1].cpu().numpy()])

        ax[0].set_title("Initial Matching")
        ax[0].legend(fontsize=10, loc="lower left")

        for column in range(1, columns):
            flow = self.flows[column -1]
            flow.v_model.eval()
            traj = flow.sample_ode(z0=z0, num_steps=num_steps)

            if dimension == 2:
                ax[column].scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
                ax[column].scatter(z0[:, 0].cpu().numpy(), z0[:, 1].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)
                ax[column].scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), label='Generated', alpha=0.15)

                traj_particles = torch.stack(traj)
                for i in range(num_connections):
                    ax[column].plot(traj_particles[:, i, 0].cpu().numpy(), traj_particles[:, i, 1].cpu().numpy())

            ax[column].set_title(f"{column}-Rectified Flow")
            ax[column].legend(fontsize=10, loc="lower left")

        if dimension == 2:
            flow = self.flows[-1]
            flow.v_model.eval()
            traj = flow.reverse_sample_ode(z1=z1, num_steps=num_steps)

            ax[-1].scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
            ax[-1].scatter(z0[:, 0].cpu().numpy(), z0[:, 1].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)
            ax[-1].scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), label='Generated', alpha=0.15)
            
            traj_particles = torch.stack(traj)
            for i in range(num_connections):
                ax[-1].plot(traj_particles[:, i, 0].cpu().numpy(), traj_particles[:, i, 1].cpu().numpy())


        ax[-1].set_title("Reverse Sampling")
        ax[-1].legend(fontsize=10, loc="lower left")
        plt.show()


#### configurable Toy_problems with different data distributions, inheriting from Toy_problem class ####

class Toy_problem1(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        pile_sampler = Categorical(torch.tensor([1/2 , 1/2]))
        normal_init = MultivariateNormal(torch.tensor([[2,2],[2,6]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2)]))
        normal_target = MultivariateNormal(torch.tensor([[6,2],[6,6]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2)]))
        self.sampler_init = MixtureSameFamily(pile_sampler, normal_init)
        self.sampler_target = MixtureSameFamily(pile_sampler, normal_target)

        self.info = {"distribution_init": "gaussian", "piles_init": 2, "distribution_target": "gaussian", "piles_target": 2, "formation": "square"}
        # add gaussian params to info / and parameter?

class Toy_problem2(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        pile_sampler = Categorical(torch.tensor([1/3, 1/3, 1/3]))
        normal_init = MultivariateNormal(torch.tensor([[10* np.sqrt(3) /2., 5],[-10 * np.sqrt(3) /2., 5], [0.0, -10 * np.sqrt(3) /2]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2), torch.eye(2)]))
        normal_target = MultivariateNormal(torch.tensor([[10* np.sqrt(3) /2., -5],[-10 * np.sqrt(3) /2., -5], [0.0, 10 * np.sqrt(3) /2]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2), torch.eye(2)]))
        self.sampler_init = MixtureSameFamily(pile_sampler, normal_init)
        self.sampler_target = MixtureSameFamily(pile_sampler, normal_target)

        self.info = {"distribution_init": "gaussian", "piles_init": 2, "distribution_target": "gaussian", "piles_target": 2, "formation": "square"}
        # add gaussian params to info / and parameter?

class Toy_problem3(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        num_piles = 6
        pile_sampler = Categorical(torch.tensor([1/num_piles for i in range(num_piles)]))
        normal_init = MultivariateNormal(torch.tensor([[5, 0], [5 * np.cos(2 * np.pi / num_piles), 5 * np.sin(2 *np.pi /num_piles)],[5 * np.cos(4 * np.pi / num_piles), 5 * np.sin(4 * np.pi /num_piles)], [5 * np.cos(6 * np.pi / num_piles), 5 * np.sin(6 * np.pi /num_piles)],[5 * np.cos(8 * np.pi / num_piles), 5 * np.sin(8 * np.pi /num_piles)],[5 * np.cos(10 * np.pi / num_piles), 5 * np.sin(10 * np.pi /num_piles)]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2), torch.eye(2), torch.eye(2), torch.eye(2), torch.eye(2)]))
        normal_target = MultivariateNormal(torch.tensor([[10, 0], [10 * np.cos(2 * np.pi / num_piles), 10 * np.sin(2 *np.pi /num_piles)],[10 * np.cos(4 * np.pi / num_piles), 10 * np.sin(4 * np.pi /num_piles)], [10 * np.cos(6 * np.pi / num_piles), 10 * np.sin(6 * np.pi /num_piles)],[10 * np.cos(8 * np.pi / num_piles), 10 * np.sin(8 * np.pi /num_piles)],[10 * np.cos(10 * np.pi / num_piles), 10 * np.sin(10 * np.pi /num_piles)]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2), torch.eye(2), torch.eye(2), torch.eye(2), torch.eye(2)]))
        self.sampler_init = MixtureSameFamily(pile_sampler, normal_init)
        self.sampler_target = MixtureSameFamily(pile_sampler, normal_target)

        self.info = {"distribution_init": "gaussian", "piles_init": 2, "distribution_target": "gaussian", "piles_target": 2, "formation": "square"}
        # add gaussian params to info / and parameter?

class Toy_problem4(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        num_piles = 6
        pile_sampler = Categorical(torch.tensor([1/num_piles for i in range(num_piles)]))
        pile_sampler2 = Categorical(torch.tensor([1]))
        normal_init = MultivariateNormal(torch.tensor([[5, 0], [5 * np.cos(2 * np.pi / num_piles), 5 * np.sin(2 *np.pi /num_piles)],[5 * np.cos(4 * np.pi / num_piles), 5 * np.sin(4 * np.pi /num_piles)], [5 * np.cos(6 * np.pi / num_piles), 5 * np.sin(6 * np.pi /num_piles)],[5 * np.cos(8 * np.pi / num_piles), 5 * np.sin(8 * np.pi /num_piles)],[5 * np.cos(10 * np.pi / num_piles), 5 * np.sin(10 * np.pi /num_piles)]]).float(), 0.1*torch.stack([torch.eye(2) for i in range(num_piles)]))
        normal_target = MultivariateNormal(torch.tensor([[0,0]]).float(), 0.1*torch.stack([torch.eye(2)]))
        self.sampler_init = MixtureSameFamily(pile_sampler, normal_init)
        self.sampler_target = MixtureSameFamily(pile_sampler2, normal_target)

        self.info = {"distribution_init": "gaussian", "piles_init": 2, "distribution_target": "gaussian", "piles_target": 2, "formation": "square"}
        # add gaussian params to info / and parameter?

class Toy_problem5(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        num_piles = 6
        pile_sampler = Categorical(torch.tensor([1/num_piles for i in range(num_piles)]))
        pile_sampler2 = Categorical(torch.tensor([1]))
        normal_init = MultivariateNormal(torch.tensor([[2,4],[2,6],[2,8],[2,10],[2,12],[2,14]]).float(), 0.1*torch.stack([torch.eye(2) for i in range(num_piles)]))
        normal_target = MultivariateNormal(torch.tensor([[6,9]]).float(), 0.1*torch.stack([torch.eye(2)]))
        self.sampler_init = MixtureSameFamily(pile_sampler, normal_init)
        self.sampler_target = MixtureSameFamily(pile_sampler2, normal_target)

        self.info = {"distribution_init": "gaussian", "piles_init": 2, "distribution_target": "gaussian", "piles_target": 2, "formation": "square"}
        # add gaussian params to info / and parameter?

class Toy_problem6(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2

        self.info = {"distribution_init": "gaussian", "piles_init": 2, "distribution_target": "gaussian", "piles_target": 2, "formation": "square"}
        # add gaussian params to info / and parameter?
    
    def get_samples_init(self, num_samples):
        assert False

    def get_samples_target(self, num_samples):
        assert False

    def get_samples_init(self, num_samples):
        samples, label = make_moons(num_samples * 2, shuffle=False)
        samples_init = torch.tensor(samples[:num_samples]).float()
        indices = torch.randperm(num_samples)
        samples_init = samples_init[indices].to(self.device)

        return samples_init

    def get_samples_target(self, num_samples):
        samples, label = make_moons(num_samples * 2, shuffle=False)
        samples_target = torch.tensor(samples[num_samples:]).float()
        indices = torch.randperm(num_samples)
        samples_target = samples_target[indices].to(self.device)

        return samples_target

class Toy_problem_2(Toy_problem):
    def __init__(self, device):
        super().__init__(device)

        self.input_dim = 2
        pile_sampler = Categorical(torch.tensor([1/2 , 1/2]))
        normal_init = MultivariateNormal(torch.tensor([[2,2],[2,6]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2)]))
        normal_target = MultivariateNormal(torch.tensor([[2,2],[2,6]]).float(), 0.1*torch.stack([torch.eye(2), torch.eye(2)]))
        self.sampler_init = MixtureSameFamily(pile_sampler, normal_init)
        self.sampler_target = MixtureSameFamily(pile_sampler, normal_target)

        self.info = {"distribution_init": "gaussian", "piles_init": self.input_dim, "distribution_target": "gaussian", "piles_target": 2, "formation": "square"}
        # add gaussian params to info / and parameter?
