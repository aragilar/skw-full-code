"""
Utility functions for data strutures
"""

from fractions import Fraction

import attr

from h5preserve import Registry

ds_registry = Registry("skw_full_code")


def get_fields(cls):
    """
    Get the list of field names from an attr.s class
    """
    return tuple(field.name for field in attr.fields(cls))
