"""
parser.py

This module contains the implementation of the parser class.
"""

import os
import json
import logging
import logging.config
from pathlib import Path
from collections import OrderedDict
from datetime import datetime


class ConfigParser:
    def __init__(self, config, fold_id, train_id=None):
        """
        Class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param train_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        self._config = config
        save_dir = Path(self.config['trainer']['save_dir'])
        exper_name = self.config['name']

        # Use timestamp as default run-id
        if train_id is None:
            train_id = datetime.now().strftime('%H_%M_%m_%d_%Y')
        if fold_id is not None:
            fold_temp = "_fold" + str(fold_id)
            train_id += fold_temp

        self._save_dir = save_dir / exper_name / train_id
        self._log_dir = save_dir / exper_name / train_id

        # Make directory for saving checkpoints and log.
        exist_ok = train_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        self.write_json(self.config, self.save_dir / 'config.json')

        # Configure logging module
        self.setup_logging(self.log_dir)

        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @staticmethod
    def read_json(fname):
        fname = Path(fname)
        with fname.open('rt') as handle:
            return json.load(handle, object_hook=OrderedDict)

    @staticmethod
    def write_json(content, fname):
        fname = Path(fname)
        with fname.open('wt') as handle:
            json.dump(content, handle, indent=4, sort_keys=False)

    @staticmethod
    def setup_logging(save_dir, log_config = 'logger_config.json', default_level = logging.INFO):
        log_config = Path(log_config)
        if log_config.is_file():
            config = ConfigParser.read_json(log_config)
            for _, handler in config['handlers'].items():
                if 'filename' in handler:
                    handler['filename'] = str(save_dir / handler['filename'])

            logging.config.dictConfig(config)
        else:
            print("Warning: logging configuration file is not found in {}.".format(log_config))
            logging.basicConfig(level = default_level)

    @classmethod
    def from_args(cls, args, fold_id, options=''):
        """
        Initialize this class from some cli arguments.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
        assert args.config is not None, msg_no_cfg
        cfg_fname = Path(args.config)

        config = cls.read_json(cfg_fname)
        if args.config:
            # Update new config for fine-tuning
            config.update(cls.read_json(args.config))
        return cls(config, fold_id)

    def __getitem__(self, name):
        """
        Access items like ordinary dict.
        """
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                       self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # Setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir
