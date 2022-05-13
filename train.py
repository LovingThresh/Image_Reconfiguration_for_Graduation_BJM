# -*- coding: utf-8 -*-
# @Time    : 2022/4/4 11:37
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : train.py
# @Software: PyCharm

import data_loader
from comet_ml import Experiment
from model import ResNet, SRResNet

import torch
import torchsummary
import torchmetrics
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import os
import shutil
import datetime
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

experiment = object

np.random.seed(24)
torch.manual_seed(24)
torch.cuda.manual_seed_all(24)

hyper_params = {
    'ex_number': 'SRResnet_3090',
    "input_size": (3, 128, 128),
    "learning_rate": 1e-4,
    "epochs": 200,
    "batch_size": 8,
    "src_path": 'E:/BJM/Super_Resolution'
}

src_path = hyper_params['src_path']
train_comet = False

# -------------------------
#           Comet
# -------------------------

if train_comet:
    experiment = Experiment(
        api_key="sDV9A5CkoqWZuJDeI9JbJMRvp",
        project_name="Image_Reconfiguration_BJM",
        workspace="LovingThresh",
    )
    experiment.log_parameters(hyper_params)

# -------------------------
#            MODEL
# -------------------------

# model = ResNet(34)
model = SRResNet()

torchsummary.summary(model, input_size=hyper_params["input_size"], batch_size=hyper_params["batch_size"], device='cpu')

# -------------------------
#            DATA
# -------------------------
batch_size = hyper_params['batch_size']

train_data_txt = './bjm_data/train.txt'
val_data_txt = './bjm_data/valid.txt'
test_data_txt = './bjm_data/test.txt'

low_rs_train_dir = 'bjm_data/bicubic/train'
raw_train_dir = 'bjm_data/raw_image/train'

low_rs_val_dir = 'bjm_data/bicubic/valid'
raw_val_dir = 'bjm_data/raw_image/valid'

low_rs_test_dir = 'bjm_data/bicubic/test'
raw_test_dir = 'bjm_data/raw_image/test'

train_dataset = data_loader.Super_Resolution_Dataset(low_resolution_image_path=low_rs_train_dir,
                                                     raw_image_path=raw_train_dir,
                                                     data_txt=train_data_txt,
                                                     down_scale=2)

val_dataset = data_loader.Super_Resolution_Dataset(low_resolution_image_path=low_rs_val_dir,
                                                   raw_image_path=raw_val_dir,
                                                   data_txt=val_data_txt,
                                                   down_scale=2)

test_dataset = data_loader.Super_Resolution_Dataset(low_resolution_image_path=low_rs_test_dir,
                                                    raw_image_path=raw_test_dir,
                                                    data_txt=test_data_txt,
                                                    down_scale=2)

# when using weightedRandomSampler, it is already balanced random, so DO NOT shuffle again
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
# ==============================================================================
# =                           copy & upload                                    =
# ==============================================================================

a = str(datetime.datetime.now())
b = list(a)
b[10] = '-'
b[13] = '-'
b[16] = '-'
output_dir = ''.join(b)
output_dir = os.path.join(src_path, output_dir)
os.mkdir(output_dir)
hyper_params['output_dir'] = output_dir

# 本地复制源代码，便于复现(模型文件、数据文件、训练文件、测试文件)
# 冷代码
shutil.copytree('utils', '{}/{}'.format(output_dir, 'utils'))

# 个人热代码
shutil.copy('model.py', output_dir)
shutil.copy('train.py', output_dir)
shutil.copy('data_loader.py', output_dir)

# 云端上次源代码
if train_comet:
    experiment.log_asset_folder('utils', log_file_name=True)

    experiment.log_code('model.py')
    experiment.log_code('train.py')
    experiment.log_code('data_loader.py')

hyper_params['output_dir'] = output_dir
hyper_params['ex_date'] = a[:10]

if train_comet:
    hyper_params['ex_key'] = experiment.get_key()
    experiment.log_parameters(hyper_params)
    experiment.set_name('{}-{}'.format(hyper_params['ex_date'], hyper_params['ex_number']))
# -------------------------
#           Setting
# -------------------------

Epochs = hyper_params['epochs']
lr = hyper_params['learning_rate']

eval_function = torchmetrics.functional.image.psnr.peak_signal_noise_ratio
loss_function = nn.L1Loss()

# Observe that all parameters are being optimized
optimizer_ft = optim.AdamW(model.parameters(), lr=lr)
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, int(Epochs / 10))
# exp_lr_scheduler = object

# -------------------------
#           Train
# -------------------------


def train(training_model, optimizer, loss_fn, eval_fn, train_load, val_load, epochs, scheduler, Device, comet=False):
    training_model = training_model.to(device)

    def train_batch(train_model):
        training_loss = 0.0
        training_evaluation = 0.0
        train_model.train()
        # Iterate over data.
        it = 0
        for batch in train_load:
            inputs, target = batch
            inputs = inputs.to(Device)
            target = target.to(Device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            output = train_model(inputs)
            evaluation = eval_fn(output * 255, target * 255)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            training_loss += loss.item() * inputs.size(0)
            training_evaluation += evaluation.item() * inputs.size(0)
            it = it + 1
            print("Iter:{}/{}, Training Loss:{:.2f}".format(it, len(train_load), training_loss))

        training_loss /= len(train_load)
        training_evaluation /= len(train_load)
        scheduler.step()

        return training_loss, training_evaluation

    def val_batch(valid_model):
        valid_loss = 0.0
        valid_evaluation = 0.0
        valid_model.eval()
        for batch in val_load:
            inputs, target = batch

            inputs = inputs.to(Device)
            output = valid_model(inputs)

            target = target.to(Device)
            evaluation = eval_fn(output * 255, target * 255)
            target = target.float()
            loss = loss_fn(output, target)
            valid_loss += loss.detach().cpu().numpy()
            valid_evaluation += evaluation.detach().cpu().numpy()

        valid_loss /= len(val_load)
        valid_evaluation /= len(val_load)

        return valid_loss, valid_evaluation, valid_model

    def train_process(B_comet):
        train_eval_list = np.array([], dtype=float)
        val_eval_list = np.array([], dtype=float)
        for epoch in range(epochs):
            print(f'Epoch {epoch}/{epochs - 1}')
            print('-' * 10)
            train_loss, train_evaluation = train_batch(training_model)
            val_loss, val_evaluation, save_model = val_batch(training_model)
            if B_comet:
                experiment.log_metrics({"epoch_train_psnr": train_evaluation,
                                        "epoch_train_loss": train_loss,
                                        "epoch_val_psnr": val_evaluation,
                                        "epoch_val_loss": val_loss
                                        }, step=epoch)

            print('Epoch: {}, Training Loss:{:.5f}, Validation Loss: {:.5f}, '
                  'Mean Training evaluation:{:.5f}, Mean Validation evaluation:{:.5f} '
                  .format(epoch, train_loss, val_loss, train_evaluation, val_evaluation))
            train_eval_list = np.append(train_eval_list, [train_loss, train_evaluation])
            val_eval_list = np.append(val_eval_list, [val_loss, val_evaluation])
            if val_evaluation > 23:
                torch.save(save_model.state_dict(),
                           os.path.join(output_dir, 'save_model', 'Epoch_{}_eval_{}'.format(epoch, val_evaluation)))
        np.save(os.path.join(output_dir, 'train.npy'), train_eval_list)
        np.save(os.path.join(output_dir, 'train.npy'), val_eval_list)

        print(np.max(val_eval_list))

    if not comet:
        train_process(comet)
    else:
        with experiment.train():
            train_process(comet)


train(model, optimizer_ft, loss_function, eval_function,
      train_loader, val_loader, Epochs, exp_lr_scheduler, device,
      train_comet)
