#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Mimimum working example to fit a penalized bilinear regression curve to CPET data.

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
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from utils import meta_data_iterator, get_study_data_excel, get_study_data_csv
from utils import optimize_plateau # this is the key method!
from utils import nondimensionalize, flattening_fraction, o2p_response_ratio, o2p_auc
from utils import save_spreadsheet

# FIXME: Provide the paths on your computer. You can generate pseudo data first.
"""META_DATA_PATH: str = "pseudo_data/meta_data.csv"
MERGED_DATA_PATH: Optional[str] = "pseudo_data/merged_data.xlsx"
OUTPUT_DIR: str = "pseudo_data/"
"""
META_DATA_PATH: str = "/Users/mmagoon/Documents/data/cpet/tof/New O2 Pulse Testing/New O2 Pulse Testing.csv"
MERGED_DATA_PATH: Optional[str] = None
OUTPUT_DIR: str = "/Users/mmagoon/Documents/data/cpet/tof/results/"

META_FORMAT: Literal["UW","Cincinnati"] = "Cincinnati"
METHOD: Literal["MERGED", "CSV"] = "CSV"


def main():
    """main"""
    results = []

    data_dir = None
    csv_paths = None
    if METHOD == "CSV":
        data_dir = Path(str(META_DATA_PATH)).parent
        csv_paths: set[str] = {p.name for p in data_dir.glob("*.csv")}
        csv_paths.discard(META_DATA_PATH)

    for study in meta_data_iterator(META_DATA_PATH, meta_format=META_FORMAT):
        # get data
        if METHOD == "MERGED":
            sheet_name = (
                f'{study.get("ID")}_{study.get("Instance")}_{study.get("Study Type")}'
            )
            df = get_study_data_excel(MERGED_DATA_PATH, sheet_name)
        elif METHOD == "CSV":
            _info = {key: study[key] for key in ["ID", "Instance", "Study Type"]}
            df = get_study_data_csv(data_dir=data_dir, names=csv_paths, info=_info)
            print(df, "\n")
            continue
        else:
            raise ValueError(f"Unrecognized METHOD: '{METHOD}'")
        
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
