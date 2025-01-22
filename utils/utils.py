#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" organizes functions used by the MWE """

from typing import Any, Dict, List, Literal, TypeAlias

import numpy as np
import pandas as pd


META_TYPE: TypeAlias = Dict[
    Literal["ID", "Instance", "Study Type", "Start Index", "Start Time", "End Index", "End Time", "File Path"],
    str | int | float | None,
]


DATA_TYPE: TypeAlias = Dict[
    Literal['Time', 'Work', 'HR', 'VO2', 'O2-Pulse'],
    List[int|float],
]


RESULTS_TYPE: TypeAlias = Dict[
    Literal[
        "ID", 
        "Instance", 
        "Study Type", 
        "Start Index", 
        "End Index", 
        'Transition Time (min)',
        'O2-Pulse Response Ratio, time=min',
        'O2-Pulse AUC (mL*min/beat)',
        'Flattening Fraction',
        'O2-Pulse Response Ratio, time=nondimensional',
        'O2-Pulse AUC, nondimensional time (mL/beat)',
    ],
    str | int | float,
]


RESULTS_INFO_TYPE: TypeAlias = Dict[
    Literal[
        'time_min', 'time_nd', 
        'slopes_min', 'slopes_nd', 
        'intercepts_min', 'intercepts_nd', 
        'r_values_min', 'r_values_nd', 
    ],
    np.ndarray
]


def time_to_index(time: int|float|Any, times: pd.Series|np.ndarray) -> int:
    """ Find the index of a given time in a series of times """
    if isinstance(times, pd.DataFrame):
        t = times.to_numpy(dtype=np.float64)
    elif isinstance(times, np.ndarray):
        t = times.copy().flatten()
    else:
        t = np.array(times, dtype=np.float64).flatten()

    if not isinstance(time, (int,float)):
        time = float(time)

    dt = np.abs(t - time)

    return int(np.argmin(dt))

