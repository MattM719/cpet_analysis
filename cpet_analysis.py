#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Mimimum working example to fits a penalized bilinear regression curve to CPET data.

Expects to find an initial increase in O2-Pulse, then expects the O2-Pulse to stop changing.
"""

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
__version__ = "1.1.0"

import os

import numpy as np

from utils import meta_data_iterator, get_study_data
from utils import optimize_plateau # this is the key method!
from utils import nondimensionalize, flattening_fraction, o2p_response_ratio, o2p_auc
from utils import save_spreadsheet

# FIXME: Provide the paths on your computer. You can generate pseudo data first.
META_DATA_PATH: str = ...
MERGED_DATA_PATH: str = ...
OUTPUT_DIR: str = ...


def main():
    """main"""
    results = []
    for study in meta_data_iterator(META_DATA_PATH):
        # get data
        sheet_name = (
            f'{study.get("ID")}_{study.get("Instance")}_{study.get("Study Type")}'
        )
        df = get_study_data(MERGED_DATA_PATH, sheet_name)
        time = df["Time"].to_numpy(dtype=np.float64)
        nd_time = nondimensionalize(
            time, study.get("Start Index"), study.get("End Index")
        )
        o2_pulse = df["O2-Pulse"].to_numpy(dtype=np.float64)

        # find optimized plateau - this is the key method described in the paper.
        # can use minutes or dimensional time
        for t, label in zip([time, nd_time], ["min", "nd"]):
            idx_transition, slopes, intercepts, r_values = optimize_plateau(
                t,
                o2_pulse,
                start_idx=int(study.get("Start Index")),
                stop_idx=int(study.get("End Index")),
                plat_method="invslope",
            )  # note, r_values are correlation coeffs (r), not coeffs. of determination (R^2)

            study = flattening_fraction(study, t, idx_transition, (label == "nd"))
            study = o2p_response_ratio(study, slopes, (label == "nd"))
            study = o2p_auc(
                study,
                t,
                o2_pulse,
                study.get("Start Index"),
                study.get("End Index"),
                (label == "nd"),
            )

        results.append(study)

    save_spreadsheet(os.path.join(OUTPUT_DIR, "results.csv"), results)


if __name__ == "__main__":
    main()
