# -*- coding: utf-8 -*-
"""Boilerplate Training code.

Sets up commonly used arguments, model / dataloader / optimizer / scheduler
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
    save_step = args.save_step
    epochs = args.epochs
    dataset_dir = args.dataset_dir
    checkpoint_dir = args.checkpoint_path
    grad_clip = args.gradient_clip
    resume = args.resume

    # optimizer related args
    learning_rate = args.learning_rate
    scheduler_step = args.scheduler_step
    scheduler_gamma = args.scheduler_gamma
    scheduler_end = args.scheduler_end

    writer = SummaryWriter(comment='/runs/{}'.format(run_name))

    latest_checkpoint_name = '{}-latest.ckpt'.format(run_name)
    latest_checkpoint_path = path.join(checkpoint_dir, latest_checkpoint_name)

    ##################################
    # -- setup dataloader / variables
    if(gpus != None):
        device = torch.device('cuda:{}'.format(gpus))
    else:
        device = torch.device('cpu')

    # Setup default values
    # TODO: setup model
    model = FooModel().to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                            step_size=scheduler_step,
                                            gamma=scheduler_gamma)
    total_step = 0
    epoch = 0

    # load from previous checkpoint if exists
    if((not path.exists(latest_checkpoint_path)) or (not resume)):
        checkpoint = CheckPoint.load(latest_checkpoint_path, device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']
        total_step = checkpoint['total_step']

    #################################
    # -- setup datasets
    # TODO: setup dataset
    dataset = BarDataset()
    dataloader = DataLoader(dataset)

    #####################
    # -- Actual training
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            if(total_step < scheduler_end):
                scheduler.step()

            # TODO: get loss somehow using model
            loss = 

            message = '[Training] Step: {:06d}, Loss: {:.04f})'
            logging.info(message.format(total_step, loss.item()))

            # reset optimizer (and clear out gradients to be applied)
            optimizer.zero_grad()

            # compute gradient
            loss.backward()

            # clip gradient if grad_clip is given
            if(grad_clip):
                utils.clip_grad_norm_(model.parameters(), grad_clip)

            # update optimizer (and actually apply gradients)
            optimizer.step()
            total_step += 1

            # write to tensorboard
            writer.add_scalar('data/loss', loss, total_step)

            # -- save the run every some time
            if((total_step) % save_step == 0):
                checkpoint_name = '{}-{}.ckpt'.format(run_name, total_step)
                checkpoint_path = path.join(checkpoint_dir, checkpoint_name)
                CheckPoint.save(checkpoint_path, model, optimizer, scheduler, total_step, epoch)
                CheckPoint.save(latest_checkpoint_path, model, optimizer, scheduler, total_step, epoch)

                # write historgram (optional, NOT recommended)
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), total_step)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # checkpoint related arguments
    parser.add_argument('--run-name', type=str, required=True, help='theme of this run')
    parser.add_argument('--checkpoint-path', type=str, default='./CheckPoint/', help='Path of checkpoint')
    parser.add_argument('--resume', dest='resume', action='store_true', help='Resume from previous model')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not Resume from previous model')
    parser.add_argument('--save-step', type=int, default=5000, help='Recurring number of steps for saving model')

    # environment related arguments
    parser.add_argument('--dataset-dir', type=str, required=True, help='Path of Dataset')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID used for this run. Default=CPU')

    # learning step related arguments
    parser.add_argument('--epochs', type=int, default=50000, help='Number of Epochs to run')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--gradient-clip', type=float, default=None, help='Clips gradient. None do not clip')
    parser.add_argument('--scheduler-step', type=int, default=2000, help='Scheduler step index value')
    parser.add_argument('--scheduler-end', type=int, default=10000, help='Scheduler final step value')
    parser.add_argument('--scheduler-gamma', type=float, default=0.2, help='Scheduler step update ratio')
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    main(args)
