import torch
import time
from Gpre_model import Gpre_
import os
import scipy.io as scio
import numpy as np
import random

# 固定随机数种子
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#
#
# set_seed(123)  # 设置固定的随机种子

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = './content'


# class PSOOptimizer:
#     def __init__(self, objf, lb, ub, dim, PopSize=100, iters=200, topk=1000, initial=None, device='cpu'):
#         self.objf = objf
#         self.lb = torch.tensor(lb, dtype=torch.float32, device=device)
#         self.ub = torch.tensor(ub, dtype=torch.float32, device=device)
#         self.dim = dim
#         self.PopSize = PopSize
#         self.iters = iters
#         self.initial = torch.tensor(initial, dtype=torch.float32, device=device) if initial is not None else None
#         self.device = device
#         self.topk = topk
#         # 按照 particleswarm 函数的默认参数设置
#         self.VelocityLimit = (self.ub - self.lb) * 1  # Vmax 作为搜索空间范围的20%
#         self.InertiaWeight = lambda iter: 0.9 - (0.5 * iter / self.iters)  # w 从0.9线性减少到0.4
#         self.SelfAdjustmentWeight = 1.49618  # c1
#         self.SocialAdjustmentWeight = 1.49618  # c2
#
#     @torch.no_grad()
#     def optimize(self):
#         vel = torch.zeros((self.PopSize, self.dim), dtype=torch.float32, device=self.device)
#         pBestScore = torch.full((self.PopSize,), float("inf"), dtype=torch.float32, device=self.device)
#         pBest = torch.zeros((self.PopSize, self.dim), dtype=torch.float32, device=self.device)
#         gBest = torch.zeros(self.dim, dtype=torch.float32, device=self.device)
#         gBestScore = float("inf")
#
#         if self.initial is None:
#             pos = torch.rand((self.PopSize, self.dim), device=self.device) * (self.ub - self.lb) + self.lb
#         else:
#             pos = self.initial
#
#         convergence_curve = []
#         timerStart = time.time()
#         gBestScore_list = []
#         gBest_list = []
#         for l in range(0, self.iters):
#             # 确保位置在边界内
#             pos = torch.clamp(pos, self.lb, self.ub)
#
#             # 计算所有粒子的适应度值
#             fitnesses = - self.objf(pos)
#             fitnesses = torch.squeeze(fitnesses, dim=1)
#
#             # 更新个人最佳
#             update_mask = fitnesses < pBestScore
#             pBestScore[update_mask] = fitnesses[update_mask]
#             pBest[update_mask] = pos[update_mask].clone()
#
#             # 更新全局最佳
#             best_idx = torch.argmin(pBestScore)
#             if pBestScore[best_idx] < gBestScore:
#                 gBestScore = pBestScore[best_idx]
#                 gBest = pBest[best_idx].clone()
#
#             # 动态调整惯性权重
#             w = self.InertiaWeight(l)
#
#             # 更新速度和位置
#             r1 = torch.rand((self.PopSize, 1), device=self.device)
#             r2 = torch.rand((self.PopSize, 1), device=self.device)
#             vel = torch.clamp(
#                 w * vel + self.SelfAdjustmentWeight * r1 * (pBest - pos) + self.SocialAdjustmentWeight * r2 * (
#                         gBest - pos),
#                 -self.VelocityLimit,
#                 self.VelocityLimit
#             )
#             pos += vel
#
#             convergence_curve.append(gBestScore.item())
#
#             print(l)
#             gBestScore_list.append(gBestScore)
#             gBest_list.append(gBest)
#
#         timerEnd = time.time()
#         print(f"Time: {timerEnd - timerStart}")
#         gBestScore_tensor = torch.stack(gBestScore_list, dim=0)
#         gBest_tensor = torch.stack(gBest_list, dim=0)
#         best_s_list = torch.unique(torch.cat((gBestScore_tensor.unsqueeze(1), gBest_tensor), dim=1), dim=0)
#
#         # pBest = torch.round(pBest)
#         # best_s_list = torch.unique(torch.cat((pBestScore.unsqueeze(1), pBest), dim=1), dim=0)[0:self.topk, :]
#         # best_s_list = torch.unique(torch.cat((pBestScore.unsqueeze(1), pBest), dim=1), dim=0)
#         return best_s_list

class PSOOptimizer:
    def __init__(self, objf, lb, ub, dim, PopSize=100, iters=200, topk=10, initial=None, device='cpu'):
        self.objf = objf
        self.lb = torch.tensor(lb, dtype=torch.float32, device=device)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=device)
        self.dim = dim
        self.PopSize = PopSize
        self.iters = iters
        self.initial = torch.tensor(initial, dtype=torch.float32, device=device) if initial is not None else None
        self.device = device
        self.topk = topk  # 修改为每次迭代中保存的pBest数量
        # 按照 particleswarm 函数的默认参数设置
        self.VelocityLimit = (self.ub - self.lb) * 0.2  # Vmax 作为搜索空间范围的20%
        self.InertiaWeight = lambda iter: 0.9 - (0.5 * iter / self.iters)  # w 从0.9线性减少到0.4
        self.SelfAdjustmentWeight = 2  # c1
        self.SocialAdjustmentWeight = 2  # c2

    @torch.no_grad()
    def optimize(self):
        vel = torch.zeros((self.PopSize, self.dim), dtype=torch.float32, device=self.device)
        pBestScore = torch.full((self.PopSize,), float("inf"), dtype=torch.float32, device=self.device)
        pBest = torch.zeros((self.PopSize, self.dim), dtype=torch.float32, device=self.device)
        gBest = torch.zeros(self.dim, dtype=torch.float32, device=self.device)
        gBestScore = float("inf")

        if self.initial is None:
            pos = torch.rand((self.PopSize, self.dim), device=self.device) * (self.ub - self.lb) + self.lb
        else:
            pos = self.initial

        convergence_curve = []
        timerStart = time.time()
        best_pBest_list = []  # 用于存储每次迭代中最好的topk个pBest

        for l in range(0, self.iters):
            # 确保位置在边界内
            pos = torch.clamp(pos, self.lb, self.ub)

            # 计算所有粒子的适应度值
            fitnesses = - self.objf(pos)
            fitnesses = torch.squeeze(fitnesses, dim=1)

            # 更新个人最佳
            update_mask = fitnesses < pBestScore
            pBestScore[update_mask] = fitnesses[update_mask]
            pBest[update_mask] = pos[update_mask].clone()

            # 更新全局最佳
            best_idx = torch.argmin(pBestScore)
            if pBestScore[best_idx] < gBestScore:
                gBestScore = pBestScore[best_idx]
                gBest = pBest[best_idx].clone()

            # 动态调整惯性权重
            w = self.InertiaWeight(l)

            # 更新速度和位置
            r1 = torch.rand((self.PopSize, 1), device=self.device)
            r2 = torch.rand((self.PopSize, 1), device=self.device)
            vel = torch.clamp(
                w * vel + self.SelfAdjustmentWeight * r1 * (pBest - pos) + self.SocialAdjustmentWeight * r2 * (
                        gBest - pos),
                -self.VelocityLimit,
                self.VelocityLimit
            )
            pos += vel

            # 选择当前迭代中最好的topk个pBest
            # topk_indices = torch.argsort(pBestScore)[:self.topk]
            # topk_pBest = pBest[topk_indices]
            # topk_pBestScores = pBestScore[topk_indices]

            # 将最好的topk个pBest添加到列表中
            best_pBest_list.append(torch.cat((pBestScore.unsqueeze(1), pBest), dim=1))

            convergence_curve.append(gBestScore.item())
            # print(l)

        timerEnd = time.time()
        print(f"Time: {timerEnd - timerStart}")

        # 将所有迭代中得到的最好pBest拼接起来
        all_best_pBest = torch.cat(best_pBest_list, dim=0)

        # 去重并按适应度排序
        unique_best_pBest = torch.unique(all_best_pBest, dim=0)
        sorted_best_pBest = unique_best_pBest[torch.argsort(unique_best_pBest[:, 0])]

        # return sorted_best_pBest[:self.topk * self.iters]  # 返回前topk*iters个最好的pBest
        return sorted_best_pBest[:self.topk * self.iters]


def func(trainfile, optfile, topk=110):
    torch.manual_seed(3)
    torch.cuda.manual_seed(3)
    np.random.seed(3)
    torch.backends.cudnn.deterministic = True
    dim = 640
    PopSize = 700
    iters = 2000
    lb = [0] * dim
    ub = [1] * dim
    Gpre = Gpre_(trainfile, 512)
    model = Gpre.net.to(device)
    loss_ = Gpre.train(n_train=None, lr=0.01, batch_size=1024, n_epochs=1000, sample_interval=0)
    model.eval()
    model_dir = os.path.join(root, '{}/{}_{}.pth'.format('regressor1', 'regressor1', 1000))
    model.load_state_dict(torch.load(model_dir, map_location=device))

    # initial pop
    filename = trainfile.split('.mat')[0]
    all_data = scio.loadmat(filename + '.mat')
    raw_x = all_data['inputs']
    # initial = None
    initial = np.reshape(raw_x[:600], (-1, dim))
    initial = np.concatenate((initial, (np.random.rand(100, dim) > 0.5).astype(np.float32)))
    # initial = np.concatenate((initial, np.zeros((1, 160)), (np.random.rand(99899, 160) < 0.9).astype(np.float32)))

    # Initialize the optimizer with parameters similar to MATLAB's particleswarm
    optimizer = PSOOptimizer(objf=model, lb=lb, ub=ub, dim=dim, PopSize=PopSize, iters=iters, initial=initial,
                             device=device)

    # Run the optimization
    best_s_list = optimizer.optimize()
    fit = best_s_list[:, 0]
    x = best_s_list[:, 1:]
    x = x.view(-1, 40, 16)

    scio.savemat(optfile,
                 {'x': x.detach().cpu().numpy().astype('float64'),
                  'fit': fit.detach().cpu().numpy().astype('float64')})


func(r'train.mat', r'opt_g\10nopt_step1.mat')
