"""
Custom error and warning exceptions.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021


class NotDataAddedError(ValueError, AttributeError):
    """"""


class NotGeneratedError(ValueError, AttributeError):
    """"""


class NotSolvedError(ValueError, AttributeError):
    """"""


class CounterfactualsFoundWarning(UserWarning):
    """"""
