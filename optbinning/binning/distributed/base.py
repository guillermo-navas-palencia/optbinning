"""
Base optimal binning sketch algorithm class.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

from ...exceptions import NotSolvedError


class BaseSketch:
    def _check_is_solved(self):
        if not self._is_solved:
            raise NotSolvedError("This {} instance is not solved yet. Call "
                                 "'solve' with appropriate arguments."
                                 .format(self.__class__.__name__))
