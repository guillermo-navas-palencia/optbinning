"""
Base optimal binning sketch algorithm class.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

from ...exceptions import NotSolvedError
from ...logging import Logger


class BaseSketch:
    def __getstate__(self):
        d = self.__dict__.copy()
        del d["_logger"]
        del d["_class_logger"]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._class_logger = Logger(__name__)
        self._logger = self._class_logger.logger

    def _check_is_solved(self):
        if not self._is_solved:
            raise NotSolvedError("This {} instance is not solved yet. Call "
                                 "'solve' with appropriate arguments."
                                 .format(self.__class__.__name__))
