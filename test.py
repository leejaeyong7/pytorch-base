# -*- coding: utf-8 -*-
"""Boilerplate Testing code.

Sets up commonly used arguments, model / dataloader
"""
from os import path
import argparse
import logging

import torch
import torch.nn.functional as NF
import torchvision.transforms.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import utils
from torch import optim

from model_saver import CheckPoint
# TODO: change this
from model.foo_model import FooModel
from dataset.bar_dataset import BarDataset

logging.basicConfig(level='INFO',
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

def main(args):
    run_name = args.run_name
    gpus = args.gpu
    epochs = args.epochs
    dataset_dir = args.dataset_dir
    checkpoint_dir = args.checkpoint_path

    ##################################
    # -- setup dataloader / variables
    if(gpus != None):
        device = torch.device('cuda:{}'.format(gpus))
    else:
        device = torch.device('cpu')

    # Setup default values
    # TODO: setup model
    model = FooModel().to(device)
    model.eval()

    # load from previous checkpoint if exists
    latest_checkpoint_name = '{}-latest.ckpt'.format(run_name)
    latest_checkpoint_path = path.join(checkpoint_dir, latest_checkpoint_name)
    checkpoint = CheckPoint.load(latest_checkpoint_path, device)
    model.load_state_dict(checkpoint['model'])
    total_step = 0

    #################################
    # -- setup datasets
    # TODO: setup dataset
    dataset = BarDataset()
    dataloader = DataLoader(dataset)

    #####################
    # -- Actual Testing
    for i, data in enumerate(dataloader):
        # TODO: get accuracy somehow using model
        acc = model.get_accuracy(data)
        message = '[Testing ] Step: {:06d}, Accuracy: {:.04f})'
        logging.info(message.format(total_step, acc.item()))
        total_step += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', type=str, required=True, help='theme of this run')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Path of Dataset')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID used for this run. Default=CPU')
    parser.add_argument('--checkpoint-path', type=str, default='./CheckPoint/', help='Path of checkpoint')
    args = parser.parse_args()
    main(args)

