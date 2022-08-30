# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
import argparse
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
import data_loader as loaders
import data_collate as collates
import json

from model.fastspeech2 import FastSpeech2, FastSpeech2Xvector

import torch


def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


def latest_checkpoint_path(dir_path, regex="grad_*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


# def load_checkpoint(logdir, model, num=None):
#     if num is None:
#         model_path = latest_checkpoint_path(logdir, regex="grad_*.pt")
#     else:
#         model_path = os.path.join(logdir, f"grad_{num}.pt")
#     print(f'Loading checkpoint {model_path}...')
#     model_dict = torch.load(model_path, map_location=lambda loc, storage: loc)
#     model.load_state_dict(model_dict, strict=False)
#     return model
def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = 1
    if 'iteration' in checkpoint_dict.keys():
        iteration = checkpoint_dict['iteration']
    if 'learning_rate' in checkpoint_dict.keys():
        learning_rate = checkpoint_dict['learning_rate']
    else:
        learning_rate = None
    if optimizer is not None and 'optimizer' in checkpoint_dict.keys():
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    logger.info("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_tensor(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_plot(tensor, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return


def get_correct_class(hps, train=True):
    assert hps.pe, "fastspeech2 must have pitch energy!"
    if train:
        if not hps.xvector:
            loader = loaders.SpkIDLoaderWithPE
            collate = collates.SpkIDCollateWithPE
            model = FastSpeech2
            dataset = loader(utts=hps.data.train_utts,
                             hparams=hps.data,
                             feats_scp=hps.data.train_feats_scp,
                             phn2id=hps.data.phn2id,
                             utt2phn_duration=hps.data.train_utt2phn_duration,
                             utt2spk=hps.data.train_utt2spk,
                             utt2phns=hps.data.train_utt2phns,
                             var_scp=hps.data.train_var_scp)
        else:  # no pitch energy
            loader = loaders.XvectorLoaderWithPE
            collate = collates.XvectorCollateWithPE
            model = FastSpeech2Xvector
            dataset = loader(utts=hps.data.train_utts,
                             hparams=hps.data,
                             feats_scp=hps.data.train_feats_scp,
                             utt2phns=hps.data.train_utt2phns,
                             phn2id=hps.data.phn2id,
                             utt2phn_duration=hps.data.train_utt2phn_duration,
                             spk_xvector_scp=hps.data.train_spk_xvector_scp,
                             utt2spk_name=hps.data.train_utt2spk,
                             var_scp=hps.data.train_var_scp)
    else:
        if not hps.xvector:
            loader = loaders.SpkIDLoaderWithPE
            collate = collates.SpkIDCollateWithPE
            model = FastSpeech2
            dataset = loader(utts=hps.data.val_utts,
                             hparams=hps.data,
                             feats_scp=hps.data.val_feats_scp,
                             phn2id=hps.data.phn2id,
                             utt2phn_duration=hps.data.val_utt2phn_duration,
                             utt2spk=hps.data.val_utt2spk,
                             utt2phns=hps.data.val_utt2phns,
                             var_scp=hps.data.val_var_scp)
        else:  # no pitch energy
            loader = loaders.XvectorLoaderWithPE
            collate = collates.XvectorCollateWithPE
            model = FastSpeech2Xvector
            dataset = loader(utts=hps.data.val_utts,
                             hparams=hps.data,
                             feats_scp=hps.data.val_feats_scp,
                             utt2phns=hps.data.val_utt2phns,
                             phn2id=hps.data.phn2id,
                             utt2phn_duration=hps.data.val_utt2phn_duration,
                             spk_xvector_scp=hps.data.val_spk_xvector_scp,
                             utt2spk_name=hps.data.val_utt2spk,
                             var_scp=hps.data.val_var_scp)
    return dataset, collate(), model


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                        help='JSON file for configuration')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model name')
    parser.add_argument('-s', '--seed', type=int, default=1234)
    parser.add_argument('--not-pretrained', action='store_true', help='if set to true, then train from scratch')

    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(config_path, "r") as f:
            data = f.read()
        with open(config_save_path, "w") as f:
            f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    hparams.train.seed = args.seed
    hparams.not_pretrained = args.not_pretrained
    return hparams


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info("Saving model and optimizer state at iteration {} to {}".format(
        iteration, checkpoint_path))
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({'model': state_dict,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path)


def get_hparams_decode():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                        help='JSON file for configuration')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model name')
    parser.add_argument('-s', '--seed', type=int, default=1234)
    parser.add_argument('--dataset', choices=['train', 'val'], default='val', type=str, help='which dataset to use')
    parser.add_argument('--use-control-spk', action='store_true', help='whether to use GT spk or other spk')
    parser.add_argument('--control-spk-id', default=None, type=int, help='if use control spk, then which spk')
    parser.add_argument('--control-spk-name', default=None, type=str, help='if use control spk, then which spk')
    parser.add_argument("--max-utt-num", default=100, type=int, help='maximum utts number to decode')
    parser.add_argument("--specify-utt-name", default=None, type=str, help='if specified, only decodes for that utt')
    parser.add_argument('-t', "--timesteps", type=int, default=10, help='how many timesteps to perform reverse diffusion')

    parser.add_argument("--stoc", action='store_true', default=False, help="Whether to add stochastic term into decoding")


    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")  # NOTE: which config to load
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    hparams.train.seed = args.seed

    if args.use_control_spk:
        if hparams.xvector:
            assert args.control_spk_name is not None
        else:
            assert args.control_spk_id is not None

    return hparams, args
