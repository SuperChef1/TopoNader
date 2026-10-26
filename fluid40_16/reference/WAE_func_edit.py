import cv2
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from Functions import same_seeds, FeatureDataset, weights_init, imq_kernel, get_latentsAE, plot_gmm
import scipy.io as scio
from Gpre_model import Gpre_

# from pso_ import pso
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = '.\content'
sns.set(color_codes=True)


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class WAE(nn.Module):
    def __init__(self, w, h, batch_size):
        super(WAE, self).__init__()
        self.input_size = w * h
        self.w = w
        self.h = h
        self.batch_size = batch_size
        node = 512

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 30),
            nn.Sigmoid()
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(30, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, self.input_size),
            nn.Sigmoid()
        )
        self.Fpre = nn.Sequential(
            nn.Linear(30, node),
            nn.BatchNorm1d(node),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(node, int(node / 2)),
            nn.Dropout(0.4),
            nn.BatchNorm1d(int(node / 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(node / 2), int(node / 4)),
            nn.BatchNorm1d(int(node / 4)),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(node / 4), 1),
        )

    def forward(self, x):
        x = x.view(-1, self.input_size)
        z = self.encoder(x)
        f = self.Fpre(z)
        x_recon = self.decoder(z)
        x_recon = x_recon.view(-1, self.w, self.h)
        return x_recon, z, f

    def train_WAE(self, optimizer, dataloader, params):
        model_name = params['model_name']
        num_epoch = params['num_epoch']
        sigma = params['sigma']  # assuming the latent space follows Gaussian
        MMD_lambda = params['MMD_lambda']  # WAE distance (maximum mean discrepancy)

        folder_dir = os.path.join(root, model_name)  # a folder to save models
        if not os.path.isdir(folder_dir):
            os.makedirs(folder_dir, exist_ok=True)

        loss_ = []
        for epoch in range(num_epoch):
            start_time = time.time()
            total_loss = []  # save for plot, recon loss+MMD
            total_recon = []  # binary cross entropy
            total_MMD = []  # maximum mean discrepancy
            total_pre = []  # Prediction loss

            for i, (x, y) in enumerate(dataloader):
                x = x.to(device)
                y = y.to(device)
                self.train()  # Set model to training mode
                recon_x, z_tilde, f_tilde = self(x)  # latent space is Z_tilde
                z = sigma * torch.randn_like(z_tilde)  # Sample from a Gaussian with the same shape as z_tilde

                recon_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
                MMD_loss = imq_kernel(z_tilde, z, h_dim=2).to(device) / x.size(0)
                pre_loss = F.mse_loss(f_tilde.squeeze(), y, reduction='mean')
                loss = recon_loss + MMD_loss * MMD_lambda + pre_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss.append(loss.item())
                total_recon.append(recon_loss.item())
                total_MMD.append(MMD_loss.item())
                total_pre.append(pre_loss.item())

            avg_loss = sum(total_loss) / len(total_loss)
            avg_recon = sum(total_recon) / len(total_recon)
            avg_MMD = sum(total_MMD) / len(total_MMD)
            avg_pre = sum(total_pre) / len(total_pre)
            loss_.append(avg_loss)

            print(
                '[{:03}/{:03}] loss: {:.6f} Recon_loss: {:.6f}, MMD_loss:{:.6f}, pre_loss:{:.6f}, time: {:.3f} sec'.format(
                    epoch + 1, num_epoch, avg_loss, avg_recon, avg_MMD, avg_pre, time.time() - start_time))

            if (epoch + 1) % 5 == 0:
                save_model_dir = str(model_name + "_{}.pth".format(epoch + 1))
                torch.save(self.state_dict(), os.path.join(folder_dir, save_model_dir))

        return loss_

    @torch.no_grad()
    def get_latentsAE(self, dataset):  # from dataset to latents
        self.eval()  # Set model to evaluation mode
        latents = []
        la_pre = []
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            recon_x, z, free = self(x)
            latents.append(z.cpu().numpy())
            la_pre.append(free.cpu().numpy())
        return np.concatenate(latents, axis=0), np.concatenate(la_pre, axis=0)


def func(trainfile, latentfile, topk):
    # Set random seeds for reproducibility
    same_seeds(1)
    # torch.manual_seed(63)
    # torch.cuda.manual_seed(63)
    # np.random.seed(63)
    # torch.backends.cudnn.deterministic = True
    # Define parameters
    params = {
        'num_epoch': 1000,
        'num_epoch_pre': 1000,
        'batch_size': 1024,
        'lr1': 5e-4,
        'lr2': 0.01,
        'weight_decay': 1e-4,
        'sigma': 8.0,
        'MMD_lambda': 1e-4,
        'model_name': 'WAE',
        'function_model1': 'regressor1',
        'function_model2': 'regressor2',
    }

    # Load data
    filename = trainfile.split('.mat')[0]
    all_data = scio.loadmat(filename + '.mat')
    raw_x = all_data['inputs']
    # raw_x = raw_x[:600]
    raw_y = all_data['outputs'].reshape(-1)
    # raw_y = raw_y[:600]
    raw_y = (raw_y - np.mean(raw_y)) / np.std(raw_y)
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(raw_x).float(), torch.from_numpy(raw_y).float())
    drop_last = raw_y.shape[0] % params['batch_size'] == 1
    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last)

    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WAE(raw_x.shape[1], raw_x.shape[2], params['batch_size']).to(device)
    optimizer1 = optim.Adam(model.parameters(), lr=params['lr1'], weight_decay=params['weight_decay'])

    # Train model
    loss_ = model.train_WAE(optimizer1, dataloader, params)

    # Save model
    model_dir = os.path.join(root,
                             '{}/{}_{}.pth'.format(params['model_name'], params['model_name'], params['num_epoch']))
    # torch.save(model.state_dict(), model_dir)

    # Load trained model for inference
    model = WAE(raw_x.shape[1], raw_x.shape[2], params['batch_size']).to(device)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    # Get latent representations
    latents, lat_f = model.get_latentsAE(dataset)

    pop_max = np.max(latents)
    pop_min = np.min(latents)
    scio.savemat(latentfile, {'latent': latents.astype(np.float64), 'outputs': all_data['outputs'], 'pop_max': pop_max,
                              'pop_min': pop_min})
    gm = GaussianMixture(n_components=30, random_state=0, init_params='kmeans').fit(latents)
    # plot a n_components v.s. Average negative log likelihood
    print('Average negative log likelihood:', -1 * gm.score(latents))
    gm_population_1_np = gm.sample(1000)[0]
    gm_population_1 = torch.from_numpy(gm_population_1_np).float().to(device)
    gm_prediction_1 = model.Fpre(gm_population_1).to(device)
    gm_prediction_1_np = gm_prediction_1.cpu().detach().numpy() * -1
    index_all = np.argsort(gm_prediction_1_np.reshape(gm_prediction_1_np.shape[0]))
    index = index_all[:int(topk + 5)]
    population_1 = gm_population_1_np[index]
    prediction_1 = gm_prediction_1_np[index] * -1
    z = torch.from_numpy(population_1).float().to(device)
    add_x_gm = model.decoder(z)
    add_x_gm_np = torch.round(add_x_gm).cpu().detach().numpy()

    # GD
    model.requires_grad_(True)
    grad_list = []

    max_index = np.argmax(raw_y)  # find the index of the max value of fittness
    latent_intial = latents[max_index]
    curr_embedding = torch.tensor(latent_intial, requires_grad=True).unsqueeze(0).to(device)
    curr_fit = model.Fpre(curr_embedding).to(device)
    embed_dim = 640
    out_embedding_array = np.zeros((int(1000), embed_dim))
    out_fit_array = np.zeros((int(1000)))
    for step in range(0, int(1000)):
        model.train()
        grad = torch.autograd.grad(curr_fit, curr_embedding)[0]  # get gradient
        grad_list.append(grad.detach())
        # curr_embedding = curr_embedding.detach()
        # update step
        update_step = grad * 0.1
        curr_embedding = curr_embedding + update_step
        # cycle bool
        model = model.eval()
        curr_fit = model.Fpre(curr_embedding)
        out_embedding = model.decoder(curr_embedding)

        # save step i info
        out_embedding_array[step] = out_embedding.cpu().detach().numpy()
        out_fit_array[step] = curr_fit.cpu().detach().numpy()
    out_embedding_array = np.round(out_embedding_array)
    add_x = np.concatenate((add_x_gm_np, out_embedding_array))
    add_x = np.reshape(add_x, (-1, 40, 16))
    prediction = np.concatenate((prediction_1, out_fit_array.reshape(-1, 1)))
    # # 去重
    # _, indices_repeat = np.unique(add_x, return_index=True)
    # add_x = add_x[np.sort(indices_repeat)]
    # prediction = prediction[np.sort(indices_repeat)]

    scio.savemat(latentfile, {'base': add_x, 'prediction': prediction})


if __name__ == '__main__':
    trainfile = './data_g/10ntrain_train_step1.mat'
    # optfile = './data_g/10ntrain_opt_step1.mat'
    latentfile = './data_latent_g/10ntrain_train_step1.mat'
    # n_train = None
    seed_opt = 123
    topk = 10
    func(trainfile, latentfile, topk)
