# -*- coding: utf-8 -*-
# @Time    : 2022/4/4 11:37
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : train.py
# @Software: PyCharm
import os
import json
import torch
import shutil
import datetime
import data_loader
from torch.utils.data import DataLoader


# ==============================================================================
# =                                 data                                       =
# ==============================================================================


def get_data(down_scale=0, batch_size=1, re_size=(512, 512)):
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
                                                         re_size=re_size,
                                                         data_txt=train_data_txt,
                                                         down_scale=down_scale)

    val_dataset = data_loader.Super_Resolution_Dataset(low_resolution_image_path=low_rs_val_dir,
                                                       raw_image_path=raw_val_dir,
                                                       re_size=re_size,
                                                       data_txt=val_data_txt,
                                                       down_scale=down_scale)

    test_dataset = data_loader.Super_Resolution_Dataset(low_resolution_image_path=low_rs_test_dir,
                                                        raw_image_path=raw_test_dir,
                                                        re_size=re_size,
                                                        data_txt=test_data_txt,
                                                        down_scale=down_scale)

    # when using weightedRandomSampler, it is already balanced random, so DO NOT shuffle again

    Train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    Val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    Test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return Train_loader, Val_loader, Test_loader


# ==============================================================================
# =                           copy & upload                                    =
# ==============================================================================


# 本地复制源代码，便于复现(模型文件、数据文件、训练文件、测试文件)
# 冷代码
def copy_and_upload(experiment_, hyper_params_, comet, src_path):
    a = str(datetime.datetime.now())
    b = list(a)
    b[10] = '-'
    b[13] = '-'
    b[16] = '-'
    output_dir = ''.join(b)
    output_dir = os.path.join(src_path, output_dir)
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir, 'summary'))
    os.mkdir(os.path.join(output_dir, 'save_model'))
    os.mkdir(os.path.join(output_dir, 'checkpoint'))
    hyper_params_['output_dir'] = output_dir
    shutil.copytree('utils', '{}/{}'.format(output_dir, 'utils'))

    # 个人热代码
    shutil.copy('model.py', output_dir)
    shutil.copy('train.py', output_dir)
    shutil.copy('data_loader.py', output_dir)

    # 云端上次源代码
    if comet:
        experiment_.log_asset_folder('utils', log_file_name=True)

        experiment_.log_code('model.py')
        experiment_.log_code('train.py')
        experiment_.log_code('data_loader.py')
        hyper_params_['ex_key'] = experiment_.get_key()
        experiment_.log_parameters(hyper_params_)
        experiment_.set_name('{}-{}'.format(hyper_params_['ex_date'], hyper_params_['ex_number']))

    hyper_params_['output_dir'] = output_dir
    hyper_params_['ex_date'] = a[:10]
    with open('{}/hyper_params.json'.format(output_dir), 'w') as fp:
        json.dump(hyper_params_, fp)
    return output_dir


def operate_dict_mean(ob_dict: dict, iter_num):
    new_ob_dict = {}

    def mean_dict(value):
        return value / iter_num

    for k, v in ob_dict.items():
        new_ob_dict[k] = mean_dict(v)

    return new_ob_dict


def train_epoch(train_model, train_load, Device, loss_fn, eval_fn, optimizer, scheduler, epoch, Epochs):
    it = 0
    training_loss = {}
    training_evaluation = {}

    training_loss_sum = {}
    training_eval_sum = {}

    training_loss_mean = {}
    training_eval_mean = {}
    for batch in train_load:
        it = it + 1

        inputs, target = batch
        inputs = inputs.to(Device)
        target = target.to(Device)
        optimizer.zero_grad()

        output = train_model(inputs)

        sum_loss = 0
        if isinstance(loss_fn, dict):
            if it == 1:
                for k, _ in loss_fn.items():
                    training_loss_sum[k] = 0
            for k, v in loss_fn.items():
                loss = v(output, target)
                sum_loss += loss
                training_loss[k] = loss.item()
                training_loss_sum[k] += loss.item()
            loss = sum_loss
        else:
            if it == 1:
                training_loss_sum['training_loss_mean'] = 0
            loss = loss_fn(output, target)
            training_loss['training_loss'] = loss.item()
            training_loss_sum['training_loss_mean'] += loss.item()
        loss.backward()
        optimizer.step()

        if isinstance(eval_fn, dict):
            if it == 1:
                for k, _ in eval_fn.items():
                    training_eval_sum[k] = 0
            for k, v in eval_fn.items():
                evaluation = v(output, target)
                training_evaluation[k] = evaluation.item()
                training_eval_sum[k] += evaluation.item()
        else:
            if it == 1:
                training_eval_sum['training_loss_mean'] = 0
            evaluation = eval_fn(output * 255, target * 255)
            training_evaluation['training_evaluation'] = evaluation.item()
            training_eval_sum['training_evaluation_mean'] += evaluation.item()

        training_loss_mean = operate_dict_mean(training_loss_sum, it)
        training_eval_mean = operate_dict_mean(training_eval_sum, it)

        print("Epoch:{}/{}, Iter:{}/{},".format(epoch, Epochs, it, len(train_load)))
        print(training_loss)
        print(training_evaluation)
        print("Epoch:{}/{}, Iter:{}/{}_Mean,".format(epoch, Epochs, it, len(train_load)))
        print(training_loss_mean)
        print(training_eval_mean)
        print("-" * 80)

    scheduler.step()

    training_dict = {
        'loss_mean': training_loss_mean,
        'evaluation_mean': training_eval_mean,
    }
    return training_loss_mean, training_eval_mean, training_dict


def val_epoch(valid_model, val_load, Device, loss_fn, eval_fn, epoch, Epochs):
    it = 0
    valid_loss = {}
    valid_evaluation = {}

    valid_loss_sum = {}
    valid_eval_sum = {}

    valid_loss_mean = {}
    valid_eval_mean = {}

    for batch in val_load:
        it = it + 1
        inputs, target = batch

        inputs = inputs.to(Device)
        output = valid_model(inputs)
        target = target.to(Device)

        if isinstance(loss_fn, dict):
            if it == 1:
                for k, _ in loss_fn.items():
                    valid_loss_sum[k] = 0
            for k, v in loss_fn.items():
                loss = v(output, target)
                valid_loss[k] = loss.item()
                valid_loss_sum[k] += loss.item()
        else:
            if it == 1:
                valid_loss_sum['validation_loss_mean'] = 0
            loss = loss_fn(output, target)
            valid_loss['validation_loss'] = loss.item()
            valid_loss_sum['validation_loss_mean'] += loss.item()

        if isinstance(eval_fn, dict):
            if it == 1:
                for k, _ in eval_fn.items():
                    valid_eval_sum[k] = 0
            for k, v in eval_fn.items():
                evaluation = v(output, target)
                valid_evaluation[k] = evaluation.item()
                valid_eval_sum[k] += evaluation.item()
        else:
            if it == 1:
                valid_eval_sum['training_loss_mean'] = 0
            evaluation = eval_fn(output * 255, target * 255)
            valid_evaluation['validation_evaluation'] = evaluation.item()
            valid_eval_sum['validation_evaluation_mean'] += evaluation.item()

        valid_loss_mean = operate_dict_mean(valid_loss_sum, it)
        valid_eval_mean = operate_dict_mean(valid_eval_sum, it)

        print("Epoch:{}/{}, Iter:{}/{},".format(epoch, Epochs, it, len(val_load)))
        print(valid_loss)
        print(valid_evaluation)
        print("Epoch:{}/{}, Iter:{}/{}_Mean,".format(epoch, Epochs, it, len(val_load)))
        print(valid_loss_mean)
        print(valid_eval_mean)
        print("-" * 80)
    valid_dict = {
        'loss_mean': valid_loss_mean,
        'evaluation_mean': valid_eval_mean
    }

    return valid_loss_mean, valid_eval_mean, valid_dict


def write_dict(dict_to_write, writer, step):
    for k, v in dict_to_write.items():
        for i, j in v.items():
            writer.add_scalar('{}/{}'.format(k, i), j, step)


def write_summary(train_writer_summary, valid_writer_summary, train_dict, valid_dict, step):
    write_dict(train_dict, train_writer_summary, step)
    write_dict(valid_dict, valid_writer_summary, step)


def train(training_model, optimizer, loss_fn, eval_fn,
          train_load, val_load, epochs, scheduler, Device,
          threshold, output_dir, train_writer_summary, valid_writer_summary,
          experiment, comet=False, init_epoch=1):
    training_model = training_model.to(Device)

    def train_process(B_comet, experiment_comet, init_epoch_num=init_epoch):

        for epoch in range(init_epoch_num, epochs + init_epoch_num):

            print(f'Epoch {epoch}/{epochs - 1}')
            print('-' * 10)

            training_model.train(True)
            train_loss, train_evaluation, train_dict = train_epoch(training_model, train_load, Device, loss_fn, eval_fn,
                                                                   optimizer, scheduler, epoch, epochs)
            training_model.train(False)
            val_loss, val_evaluation, valid_dict = val_epoch(training_model, val_load, Device, loss_fn, eval_fn,
                                                             epoch, epochs)
            write_summary(train_writer_summary, valid_writer_summary, train_dict, valid_dict, step=epoch)

            if B_comet:
                experiment_comet.log_metrics(train_dict, step=epoch)
                experiment_comet.log_metrics(valid_dict, step=epoch)

            print('Epoch: {}, \n Mean Training Loss:{}, \n Mean Validation Loss: {}, \n'
                  'Mean Training evaluation:{}, \nMean Validation evaluation:{} '
                  .format(epoch, train_loss, val_loss, train_evaluation, val_evaluation))

            if val_evaluation['eval_function_psnr'] > threshold:
                torch.save(training_model.state_dict(),
                           os.path.join(output_dir, 'save_model', 'Epoch_{}_eval_{}.pt'.format(epoch, val_evaluation['eval_function_psnr'])))
            if (epoch % 100) == 0:
                save_checkpoint_path = os.path.join(output_dir, 'checkpoint')
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": training_model.state_dict(),
                    "loss_fn": loss_fn,
                    "eval_fn": eval_fn,
                    "lr_schedule_state_dict": scheduler.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, os.path.join(save_checkpoint_path,  str(epoch),  '.pth'))
    if not comet:
        train_process(comet, experiment)
    else:
        with experiment.train():
            train_process(comet, experiment)
