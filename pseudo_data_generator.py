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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# set the number of synthetic data files you would like to generate
N_FILES: int = 3
OUTPUT_DIR: str = ...  #FIXME supply a directory or use: os.path.dirname(__file__)
PLOT: bool = False


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
    time = np.linspace(0, duration, n)

    start_exercise_time = 0.1 * duration  # minutes
    transition_time = 0.4 * duration
    stop_exercise_time = 0.8 * duration

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
        "HR": hr,
        "VO2": vo2,
        "O2-Pulse": o2p,
    }

    return pseudo_data


def find_nearest(target: float, arr: np.ndarray) -> int:
    """finds index of nearest value in arr"""
    err = np.abs(arr - target)
    return np.argmin(err)


def plot_data(data: dict) -> None:
    """plots HR, VO2, HR, and O2-Pulse data for reference"""
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax: tuple[Axes, ...]
    ax[0].scatter(data["Time"], data["HR"], s=0.8, c="k", alpha=0.6)
    ax[0].set_ylabel("Heart Rate\n(beats/min)")
    ax[1].scatter(data["Time"], data["VO2"], s=0.8, c="k", alpha=0.6)
    ax[1].set_ylabel("VO2\n(mL/min)")
    ax[2].scatter(data["Time"], data["O2-Pulse"], s=0.8, c="k", alpha=0.6)
    ax[2].set_ylabel("O2-Pulse\n(mL/beat)")
    ax[2].set_xlabel("Time (min)")

    for i in range(3):
        ax[i].axvline(
            data["Start Time"],
            label="Start Time",
            color="g",
            linestyle=":",
            linewidth=0.8,
            alpha=0.8,
        )
        ax[i].axvline(
            data["True Transition Time"],
            label="True Transition Time",
            color="b",
            linestyle="-.",
            linewidth=0.8,
            alpha=0.8,
        )
        ax[i].axvline(
            data["End Time"],
            label="End Time",
            color="r",
            linestyle="-",
            linewidth=0.8,
            alpha=0.8,
        )
    ax[0].legend()

    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, f"synthetic_data_{data['ID']}.png"), format="png"
    )
    plt.close()


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
        data["ID"] = file_num
        data["Instance"] = 1  # used if a participant has had multiple studies
        data["Study Type"] = (
            "CPET"  # indicates if data were collected by a non-standard method
        )

        # we performed a preprocessing step where we converted the exercise start/end times
        # to indices in the CPET data
        # a simplified methodology is modeled here
        data["Start Index"] = find_nearest(data["Start Time"], data["Time"])
        data["End Index"] = find_nearest(data["End Time"], data["Time"])

        # append meta data to correct file
        with open(meta_data_file, "a", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=meta_data_headers, dialect="excel", lineterminator="\n"
            )
            writer.writerow({key: data[key] for key in meta_data_headers})

        # store generated data
        synthetic_data.append(data)

        if PLOT:
            plot_data(data)

    # create an excel file that merges data form every participant
    column_names = ["Time", "HR", "VO2", "O2-Pulse"]
    with pd.ExcelWriter(
        os.path.join(OUTPUT_DIR, "merged_data.xlsx"), "xlsxwriter"
    ) as xl:
        for data in synthetic_data:
            sheet_name = f"{data['ID']}_{data['Instance']}_{data['Study Type']}"
            df = pd.DataFrame(
                {key: data[key] for key in column_names}, columns=column_names
            )
            df.to_excel(xl, sheet_name=sheet_name, index=False)


if __name__ == "__main__":
    main()
