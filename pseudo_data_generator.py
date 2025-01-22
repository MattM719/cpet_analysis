#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simple code to generate pseudo-data. 

This is intended to support the minimum working example. Synthetic data are generated with noise 
and other perturbations and help users get the code working. This synthetic data also demonstrates 
proper data formatting to be interpretted by the MWE, cpet_analysis.py.
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

import os
import csv
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from utils import time_to_index
from utils.output import plot_data_time

# set the number of synthetic data files you would like to generate
N_FILES: int = 3
OUTPUT_DIR: Path = Path("pseudo_data")
PLOT: bool = True


# ======================================================================== #
# code to generate synthetic data for testing this minimum working example #
# ======================================================================== #


def create_pseudo_data(n: int, duration: float) -> dict:
    """creates synthetic CPET

    Args:
        n: number of samples
        duration: total data collection time, in minutes

    Returns: dictionary of
        Time (min), heart rate (bpm), VO2 (mL/min), and O2-Pulse (mL/beat)
    """
    time = np.linspace(0, duration, n)  # minutes

    start_exercise_idx = time_to_index(0.1 * duration, time)
    transition_idx = time_to_index(0.4 * duration, time)
    stop_exercise_idx = time_to_index(0.8 * duration, time)

    start_exercise_time = time[start_exercise_idx]  
    transition_time = time[transition_idx]
    stop_exercise_time = time[stop_exercise_idx]

    work = np.zeros_like(time)
    exercising = (time >= start_exercise_time) * (time <= stop_exercise_time)
    ramp = np.linspace(0, 20 * (stop_exercise_time-start_exercise_time), np.sum(exercising))
    ramp += float(ramp[1])
    work[exercising] = ramp[:]

    # create pseudo data
    hr = np.concatenate(
        (
            np.full_like(time[time < start_exercise_time], 80),
            np.linspace(
                80,
                150,
                sum((time >= start_exercise_time) * (time < transition_time)),
                endpoint=False,
            ),
            np.linspace(
                150,
                180,
                sum((time >= transition_time) * (time <= stop_exercise_time)),
                endpoint=False,
            ),
            np.linspace(180, 80, sum(time > stop_exercise_time)),
        )
    )
    o2p = np.concatenate(
        (
            np.full_like(time[time < start_exercise_time], 4),
            np.linspace(
                4,
                9,
                sum((time >= start_exercise_time) * (time < transition_time)),
                endpoint=False,
            ),
            np.linspace(
                9,
                10,
                sum((time >= transition_time) * (time <= stop_exercise_time)),
                endpoint=False,
            ),
            np.linspace(10, 4, sum(time > stop_exercise_time)),
        )
    )

    # add synthetic noise
    hr += np.random.normal(0, 2, size=n)
    o2p += np.random.normal(0, 0.2, size=n)

    # calculate VO2 measurements that would create the synthetic O2-Pulse data
    vo2 = o2p * hr

    pseudo_data = {
        "Start Time": start_exercise_time,
        "True Transition Time": transition_time,
        "End Time": stop_exercise_time,
        "Time": time,
        "Work": work,
        "HR": hr,
        "VO2": vo2,
        "O2-Pulse": o2p,
    }

    return pseudo_data


def save_cincinnati_csv(path: Path, data: dict[Literal["Time", "Work", "HR", "VO2", "O2-Pulse"],np.ndarray]) -> None:
    """ Saves pseudo data to a CSV with the Cincinnati layout """
    headers = ['Time_sec', 'Work_Watts', 'HR', 'VO2_mLmin', 'VO2HR']
    units = ['sec', 'Watts', 'BPM', 'mL/min', 'mL/beat']
    data_keys = ["Time", "Work", "HR", "VO2", "O2-Pulse"]

    arr = np.vstack(
        tuple(
            [data[k].flatten() for k in data_keys]
        ),
        dtype=np.float64,
    ).T
    arr[:,0] *= 60 # convert from minutes to seconds

    with open(path, mode="w", encoding="utf-8") as file:
        writer = csv.writer(file, dialect="excel", lineterminator="\n")
        writer.writerow(headers)
        writer.writerow(units)
        writer.writerows(arr.tolist())
    
    path.chmod(0o660)

    return None


def main():
    """main"""
    # initialize meta data file
    meta_data_file = os.path.join(OUTPUT_DIR, "meta_data.csv")
    meta_data_headers = [
        "ID",
        "Instance",
        "Study Type",
        "Start Time",
        "Start Index",
        "End Time",
        "End Index",
        "True Transition Time",
    ]
    with open(meta_data_file, "w", encoding="utf-8") as f:
        writer = csv.writer(f, dialect="excel", lineterminator="\n")
        writer.writerow(meta_data_headers)

    # create some variability between datta files
    samples_per_minute = np.random.uniform(18, 30, N_FILES)
    durations = np.random.uniform(15, 20, N_FILES)
    n_samples = (durations * samples_per_minute).astype(np.int64)

    # create synthetic data files
    synthetic_data = []
    for file_num, (n, duration) in enumerate(zip(n_samples, durations)):
        data = create_pseudo_data(n, duration)
        data["ID"] = file_num + 1
        data["Instance"] = 1  # used if a participant has had multiple studies
        data["Study Type"] = "CPET"

        # we performed a preprocessing step where we converted the exercise start/end
        # times to indices in the CPET data
        # a simplified methodology is modeled here
        data["Start Index"] = time_to_index(data["Start Time"], data["Time"])
        data["End Index"] = time_to_index(data["End Time"], data["Time"])

        # append meta data to correct file
        with open(meta_data_file, "a", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=meta_data_headers, dialect="excel", lineterminator="\n"
            )
            writer.writerow({key: data[key] for key in meta_data_headers})

        # store generated data
        synthetic_data.append(data)

        if PLOT:
            plot_data_time(data, extra_info=None, output_dir=Path(OUTPUT_DIR), is_synthetic=True, no_lines=True, no_auc=True)

    # create an excel file that merges data form every participant
    column_names = ["Time", "Work", "HR", "VO2", "O2-Pulse"]
    with pd.ExcelWriter(
        os.path.join(OUTPUT_DIR, "merged_data.xlsx"), "xlsxwriter"
    ) as xl:
        for data in synthetic_data:
            sheet_name = f"{data['ID']}_{data['Instance']}_{data['Study Type']}"
            df = pd.DataFrame(
                {key: data[key] for key in column_names}, columns=column_names
            )
            df.to_excel(xl, sheet_name=sheet_name, index=False)

    csv_dir = OUTPUT_DIR / "cincinnati"
    csv_dir.mkdir(0o751)
    for data in synthetic_data:
        file_name = f"{data['ID']}_{data['Instance']}_{data['Study Type']}.csv"
        save_cincinnati_csv(
            path=csv_dir / file_name,
            data=data,
        )

    empty_results = OUTPUT_DIR / 'results'
    empty_results.mkdir(0o751)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(0o751)
    main()
