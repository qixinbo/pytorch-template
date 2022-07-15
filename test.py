import argparse
import torch
from tqdm import tqdm
import dataset.datasets as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

from pytorch_accelerated.trainer import Trainer, DEFAULT_CALLBACKS

def main(config):
    logger = config.get_logger('test')

    test_dataset = config.init_obj('dataset', module_data)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    trainer = Trainer(
        model,
        loss_func=loss,
        optimizer=None,
        callbacks=[
            *metrics,
            *DEFAULT_CALLBACKS
        ]
        )

    logger.info('Loading checkpoint: {} ...'.format(config.resume))

    trainer.load_checkpoint(config.resume, load_optimizer=False)

    trainer.evaluate(
        dataset=test_dataset,
        per_device_batch_size=64,
    )


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    config = ConfigParser.from_args(args)
    main(config)
