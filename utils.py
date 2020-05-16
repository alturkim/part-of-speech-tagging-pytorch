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

    def update(self, val):
        self.count += 1
        self.sum += val
        self.avg = self.sum / self.count


def set_logger(log_config_path):
    logging.config.fileConfig(log_config_path)


def save_checkpoint(state, is_best):
    """Saves model and training parameters

    Args:
        state: (dict) model's state_dict
        is_best: (bool) True if model is the best seen so far.
    """
    checkpoint_dir = './checkpoints'
    if not os.path.exists(checkpoint_dir):
        print("Creating a directory {}".format(checkpoint_dir))
        os.mkdir(checkpoint_dir)
    torch.save(state, os.path.join(checkpoint_dir, 'checkpoint.pth.tar'))
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'BEST_checkpoint.pth.tar'))


def load_checkpoint(checkpoint_file, model):
    """Loads model and training parameters

    Args:
        checkpoint: (str) file name containing the checkpoint
        model: (torch.nn.Module) model for which the parameters are loaded
    """
    checkpoint_dir = './checkpoints'
    path = os.path.join(checkpoint_dir, checkpoint_file)
    if not os.path.exists(path):
        raise IOError('Checkpoint file {} does not exist'.format(path))
    logging.info('Restoring parameters from {}'.format(path))
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])

