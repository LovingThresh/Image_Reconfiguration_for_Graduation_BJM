# -*- coding: utf-8 -*-
# @Time    : 2022/5/15 16:38
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : main.py
# @Software: PyCharm
import numpy as np

import torchsummary
import torchmetrics
import torch.optim as optim

from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from train import *
from model import *
from comet_ml import Experiment

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

np.random.seed(24)
torch.manual_seed(24)
torch.cuda.manual_seed_all(24)

train_comet = False

hyper_params = {
    "ex_number"     : 'SRResnet_3080Ti',
    "down_scale"    : 0,
    "input_size"    : (3, 128, 128),
    "batch_size"    : 16,
    "learning_rate" : 1e-5,
    "epochs"        : 200,
    "threshold"     : 22,
    "src_path"      : 'E:/BJM/Super_Resolution'
}

experiment  =   object
lr          =   hyper_params['learning_rate']
Epochs      =   hyper_params['epochs']
src_path    =   hyper_params['src_path']
down_scale  =   hyper_params['down_scale']
batch_size  =   hyper_params['batch_size']
input_size  =   hyper_params["input_size"]
threshold   =   hyper_params['threshold']

# ===============================================================================
# =                                    Comet                                    =
# ===============================================================================

if train_comet:
    experiment = Experiment(
        api_key      = "sDV9A5CkoqWZuJDeI9JbJMRvp",
        project_name = "Image_Reconfiguration_BJM",
        workspace    = "LovingThresh",
    )

# ===============================================================================
# =                                     Data                                    =
# ===============================================================================

train_loader, val_loader, test_loader = get_data(down_scale=down_scale, batch_size=batch_size)

# ===============================================================================
# =                                     Model                                   =
# ===============================================================================

model = ResNet(101)
model.init_weights()
# model = SRResNet()

torchsummary.summary(model, input_size=input_size, batch_size=batch_size, device='cpu')

# ===============================================================================
# =                                    Setting                                  =
# ===============================================================================

loss_function_mse = nn.MSELoss()
loss_function_L1 = nn.L1Loss()
loss_function = {'loss_function_mse': loss_function_mse,
                 'loss_function_L1': loss_function_L1}

eval_function_psnr = torchmetrics.functional.image.psnr.peak_signal_noise_ratio
eval_function_ssim = torchmetrics.functional.image.ssim.structural_similarity_index_measure
eval_function = {'eval_function_psnr': eval_function_psnr,
                 'eval_function_ssim': eval_function_ssim}

optimizer_ft     = optim.AdamW(model.parameters(), lr=lr)
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, int(Epochs / 10))

# ===============================================================================
# =                                  Copy & Upload                              =
# ===============================================================================

output_dir   =  copy_and_upload(experiment, hyper_params, train_comet, src_path)
timestamp    =  datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
train_writer =  SummaryWriter('{}/trainer_{}'.format(os.path.join(output_dir, 'summary'), timestamp))
val_writer  =  SummaryWriter('{}/valer_{}'.format(os.path.join(output_dir, 'summary'), timestamp))

# ===============================================================================
# =                                    Training                                 =
# ===============================================================================

train(model, optimizer_ft, loss_function, eval_function,
      train_loader, val_loader, Epochs, exp_lr_scheduler,
      device, threshold, output_dir, train_writer, val_writer, experiment, train_comet)
