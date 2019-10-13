import argparse
import json
import sys
from pathlib import Path

import torch
from attrdict import AttrDict

sys.path.append('../model')
from dataset import Spacenet5Dataset, TrainTransform, TestTransform
from model import Unet
from trainer import Trainer
from utils import set_seed, train_val_split, get_spacenet5_data


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/local_config.json')

    return parser


def main(args):
    with open(args.config_path, 'r') as file:
        config = AttrDict(json.load(file))
        config.data_dirs = [Path(d) for d in config.data_dirs]
        config.cache_dir = Path(config.cache_dir)

    set_seed(config.seed)

    train_data, val_data = [], []
    for current_dir in config.data_dirs:
        current_data = get_spacenet5_data(current_dir, image_type='PS-MS')
        current_train_data, current_val_data = train_val_split(current_data, val_size=config.val_size)
        train_data.extend(current_train_data)
        val_data.extend(current_val_data)

    train_transform = TrainTransform(config.crop_size)
    train_dataset = Spacenet5Dataset(train_data,
                                     channels=config.channels,
                                     transform=train_transform,
                                     cache_dir=config.cache_dir)

    val_transform = TestTransform(config.image_size)
    val_dataset = Spacenet5Dataset(val_data,
                                   channels=config.channels,
                                   transform=val_transform,
                                   cache_dir=config.cache_dir)

    model = Unet(in_channels= (8 if config.channels is None else len(config.channels)),
                 out_channels=config.n_out_channels,
                 model=config.base_model,
                 pretrained=config.pretrained)

    trainer = Trainer(model=model,
                      optimizer_params={'lr': config.lr,
                                        'weight_decay': config.weight_decay},
                      loss_params={'focal_weight': config.focal_weight,
                                   'dice_weight': config.dice_weight},
                      amp_params={'opt_level': config.opt_level,
                                  'loss_scale': config.loss_scale},
                      device=config.device,
                      n_jobs=config.n_jobs)

    trainer.train(train_data=train_dataset,
                  n_epochs=config.n_epochs,
                  batch_size=config.batch_size,
                  test_data=val_dataset,
                  test_batch_size=config.test_batch_size,
                  best_checkpoint_path=config.best_checkpoint_path)


if __name__ == '__main__':
    arg_parser = get_parser()
    args = arg_parser.parse_known_args()[0]
    main(args)