#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" organizes support functions used by the MWE """

__author__ = "Matthew J Magoon"
__credits__ = [
    "David M Leone",
    "Matthew J Magoon",
    "Neha Arunkumar",
    "Laurie A Soine",
    "Elizabeth C Bayley",
    "Patrick M Boyle",
    "Jonathan Buber",
]

try:
    import matplotlib
    import numpy
    import pandas
    import scipy
except Exception as exc:
    raise ImportError(
        "Please install the required packages in requirements.txt"
    ) from exc

from . import data_reader
from . import calculations
from . import output
from .utils import META_TYPE, RESULTS_TYPE, DATA_TYPE, RESULTS_INFO_TYPE, time_to_index
