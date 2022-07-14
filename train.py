import argparse
import collections
import torch
import numpy as np
# import data_loader.data_loaders as module_data
import dataset.datasets as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
# from trainer import Trainer
from pytorch_accelerated import Trainer
from utils import prepare_device


def main(config):
    # create a logger
    logger = config.get_logger('train')

    # setup datasets
    train_dataset, eval_dataset = config.init_obj('dataset', module_data)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        loss_func=criterion,
        optimizer=optimizer
        )

    trainer.train(
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        num_epochs = config['trainer']['num_epochs'],
        train_dataloader_kwargs = config['trainer']['train_dataloader_args'],
        eval_dataloader_kwargs = config['trainer']['eval_dataloader_args']
        )


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
