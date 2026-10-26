import cv2
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from Functions import same_seeds, FeatureDataset, weights_init, imq_kernel, get_latentsAE, plot_gmm
import scipy.io as scio

# from pso_ import pso
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = '.\content'
sns.set(color_codes=True)


class Gpre_():
    def __init__(self, filename, node=512):
        self.filename = filename.split('.mat')[0]
        self.data = scio.loadmat(self.filename + '.mat')
        self.img_size1, self.img_size2 = self.data['inputs'].shape[1:]
        self.img_ele = self.img_size1 * self.img_size2
        # building the DNN
        self.net = nn.Sequential(
            nn.Linear(self.img_ele, node),
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

        # a function to preprocess input and output data

    def normalize(self, x, y):
        return x, (y - self.mean_of_y) / self.std_of_y

        # train the DNN
    def train(self, n_epochs=1000, sample_interval=200, lr=0.01, batch_size=1024, n_train=None, decay=1):
        folder_dir = os.path.join(root, 'regressor1')
        x = self.data['inputs']
        y = self.data['outputs']
        y = y.reshape(-1, 1)
        x = x[:n_train, :, :]
        y = y[:n_train, :]
        self.mean_of_y = np.mean(y)
        self.std_of_y = np.std(y)
        x, y = self.normalize(x, y)

        cuda = True if torch.cuda.is_available() else False

        self.net.train()
        # Loss functions
        loss_fun = nn.MSELoss()
        if cuda:
            self.net.cuda()
            loss_fun.cuda()
        else:
            self.net.cpu()

        # Configure data loader
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        tensor_x = FloatTensor(x)
        tensor_y = FloatTensor(y)
        my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        drop_last = tensor_y.shape[
                        0] % batch_size == 1  # batch_norm cannot work with one sample, discard the last batch if last_batch.shape[0]=1
        dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, shuffle=True,
                                                 drop_last=drop_last)

        # Optimizers
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        lr_lambda = lambda epoch: decay ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        # ----------
        #  Training
        # ----------
        time_start = time.time()

        for epoch in range(n_epochs):
            for i, (imgs, labels) in enumerate(dataloader):
                #
                imgs = imgs.view(imgs.size(0), self.img_ele)
                prediction = self.net(imgs)
                optimizer.zero_grad()
                loss = loss_fun(prediction, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

            if sample_interval > 0 and epoch % sample_interval == 0:
                print('epoch', epoch, 'train_loss', loss.data.cpu().numpy())
                time_end = time.time()
                print('time cost', time_end - time_start, 's')

            if (epoch + 1) % 5 == 0:
                save_model_dir = str('regressor1' + "_{}.pth".format(epoch + 1))
                torch.save(self.net.state_dict(), os.path.join(folder_dir, save_model_dir))

        print('epoch', epoch, 'train_loss', loss.data.cpu().numpy())
        time_end = time.time()
        print('Time cost for training', time_end - time_start, 's')

        return 1

def func(trainfile, latentfile, topk):
    # %% Data loading, params - here you can play around with all different combinations of paramters to reduce the total loss. You can visualize your training history just to see how good your chosen set of hyperparameters performs
    # same_seeds(1)  # seed equals to 3
    torch.manual_seed(3)
    torch.cuda.manual_seed(3)
    np.random.seed(3)
    torch.backends.cudnn.deterministic = True
    params = {
        'num_epoch': 500,
        'num_epoch_pre': 500,
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

    Gpre = Gpre_(trainfile, node=512)
    model_Gpre = Gpre.net.to(device)
    decay = 1
    loss_ = Gpre.train(n_train=600, lr=0.01, batch_size=1024, n_epochs=1000, sample_interval=1)
    # model_Gpre_dir = os.path.join(root, '{}/{}_{}.pth'.format(params['function_model1'], params['function_model1'],
    #                                                           params['num_epoch_pre']))  # load your model
    # model_Gpre.load_state_dict(torch.load(model_Gpre_dir))


if __name__ == '__main__':
    trainfile = './data_g/10ntrain_train_step1.mat'
    # optfile = './data_g/10ntrain_opt_step1.mat'
    latentfile = './data_latent_g/10ntrain_train_step1.mat'
    # n_train = None
    seed_opt = 123
    topk = 10
    func(trainfile, latentfile, topk)
