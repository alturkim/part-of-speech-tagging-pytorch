import json
import logging
import logging.config
import os
import torch


class Config:
    """Utility for loading hyperparameters from json file
    """

    def __init__(self, json_path):
        self.json_path = json_path
        with open(json_path, 'r') as f:
            config = json.load(f)
            self.dict.update(config)

    def save(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.dict, f, indent=4)

    def update(self, json_path):
        with open(json_path, 'r') as f:
            config = json.load(f)
            self.dict.update(config)

    @property
    def dict(self):
        return self.__dict__


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, size=1):
        self.count += size
        self.sum += val * size
        self.avg = self.sum / self.count


def set_logger(log_config_path):
    logging.config.fileConfig(log_config_path)


def save_checkpoint(state, is_best, checkpoint_dir):
    """Saves model and training parameters

    Args:
        state: (dict) model's state_dict
        is_best: (bool) True if model is the best seen so far.
        checkpoint_dir: (str) Directory name to save checkpoint files in.
    """
    if not os.path.exists(checkpoint_dir):
        print("Creating a directory {}".format(checkpoint_dir))
        os.mkdir(checkpoint_dir)
    torch.save(state, os.path.join(checkpoint_dir, 'checkpoint.pth.tar'))
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'BEST_checkpoint.pth.tar'))


def load_checkpoint(checkpoint_path):
    """Loads model and training parameters

    Args:
        checkpoint_path: (str) path to checkpoint file to be used
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.nn.optim) optimizer to be resumed from checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise IOError('Checkpoint file {} does not exist'.format(checkpoint_path))
    logging.info('Restoring parameters from {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    return checkpoint
    # model.load_state_dict(checkpoint['state_dict'])
    # if optimizer:
    #     optimizer.load_state_dict(checkpoint['optim_dict'])


def adjust_learning_rate(optimizer, epoch, lr, lr_decay):
    logging.info('adjusting learning rate...')
    lr = lr / (1 + (epoch + 1) * lr_decay)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logging.info('The new learning rate is {}'.format(lr))
