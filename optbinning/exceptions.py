"""
Custom error and warning exceptions.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021


class NotDataAddedError(ValueError, AttributeError):
    """Exception class to raise if binning sketch is solved before adding
    data.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


class NotSolvedError(ValueError, AttributeError):
    """Exception class to raise if binning sketch methods are called before
    solving.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


class NotGeneratedError(ValueError, AttributeError):
    """Exception class to raise is counterfactual information is requested
    before generating explanations.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


class CounterfactualsFoundWarning(UserWarning):
    """Warning used to notify no feasible counterfactual explanations were
    found.
    """
