"""
Logging class.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import logging
import sys


class Logger:
    def __init__(self, filename=None):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s : %(message)s')

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        if filename is not None:
            fhandler = logging.FileHandler(filename)
            fhandler.setFormatter(formatter)
            self.logger.addHandler(fhandler)

    def close(self):
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
