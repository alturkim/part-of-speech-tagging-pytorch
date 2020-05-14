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


def set_logger(log_config_path):
    logging.config.fileConfig(log_config_path)

