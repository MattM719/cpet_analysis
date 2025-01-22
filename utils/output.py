#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Outputs data

"""

import csv
from pathlib import Path, PosixPath, WindowsPath
from collections.abc import Mapping
from typing import Literal, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.integrate import trapezoid

from .utils import RESULTS_TYPE, RESULTS_INFO_TYPE


COLUMN_NAMES = [
    "ID",
    "Instance",
    "Study Type",
    "Transition Time (min)",
    "Flattening Fraction",
    "O2-Pulse Response Ratio, time=min",
    "O2-Pulse Response Ratio, time=nondimensional",
    "O2-Pulse AUC (mL*min/beat)",
    "O2-Pulse AUC, nondimensional time (mL/beat)",
]


def save_spreadsheet(path: str, results: list[RESULTS_TYPE]) -> None:
    """Saves a spreadsheet of calculation results

    Parameters:
    ----------
    path: path to file  

    results: list of 'study' dicts with all the parameters of interest
    """
    with open(path, "w", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=COLUMN_NAMES, dialect="excel", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows([{col: study.get(col, None) for col in COLUMN_NAMES} for study in results])


def update_spreadsheet(path: str|Path, result: RESULTS_TYPE) -> None:
    """Saves a spreadsheet of calculation results

    Args:
        results: list of 'study' dicts with all the parameters of interest
    """
    if isinstance(path, str):
        path = Path(path)

    if path.is_dir():
        path = path / "results.csv"
    
    mode = "a" if path.exists() else "w"

    with open(path, mode=mode, encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=COLUMN_NAMES, dialect="excel", lineterminator="\n"
        )
        if mode == "w":
            writer.writeheader()
        writer.writerow({col: result.get(col, None) for col in COLUMN_NAMES})


def plot_data_time(
        data: Mapping[
            Literal[
                "ID", "Instance", "Study Type", "Time", "HR", "VO2", "O2-Pulse",
                "Start Time", "True Transition Time", "Transition Time (min)",
                "End Time"
            ], Any
        ],
        extra_info: Optional[RESULTS_INFO_TYPE],
        output_dir: Path,
        is_synthetic: bool = False,
        no_lines: bool = True,
        no_transition_annotations: bool = False,
        no_auc: bool = False,
        fmt: Literal["png","svg","pdf"] = "png",
        dpi: int = 600,
) -> None:
    """ plots HR, VO2, HR, and O2-Pulse data for reference """
    assert isinstance(output_dir, (Path,PosixPath,WindowsPath))
    tag = "_SYNTHETIC" if is_synthetic else ""

    fig, ax = plt.subplots(3, 1, sharex=True, dpi=dpi)
    ax: tuple[Axes, ...]
    ax[0].scatter(data["Time"], data["HR"], s=0.8, c="k", alpha=0.6)
    ax[0].set_ylabel("Heart Rate\n(beats/min)")
    ax[1].scatter(data["Time"], data["VO2"], s=0.8, c="k", alpha=0.6)
    ax[1].set_ylabel("VO2\n(mL/min)")
    ax[2].scatter(data["Time"], data["O2-Pulse"], s=0.8, c="k", alpha=0.6)
    ax[2].set_ylabel(f"O$_2$ Pulse\n(mL/beat)")
    ax[2].set_xlabel("Time (min)")

    start_time = data.get(
        "Start Time", 
        data["Time"][int(data["Start Index"])],
    )
    end_time = data.get(
        "End Time", 
        data["Time"][int(data["End Index"])],
    )

    if is_synthetic:
        trans_time = data["True Transition Time"]
    else:
        trans_time = data['Transition Time (min)']

    if not no_transition_annotations:
        trans_label = "True Transition Time" if is_synthetic else "Transition Time"
        for i in range(3):
            ann0 = ax[i].axvline(
                start_time,
                label=("Start Time" if i == 2 else None),
                color="g",
                linestyle=":",
                linewidth=0.8,
                alpha=0.8,
            )
            ann1 = ax[i].axvline(
                trans_time,
                label=(trans_label if i == 2 else None),
                color="b",
                linestyle="-.",
                linewidth=0.8,
                alpha=0.8,
            )
            ann2 = ax[i].axvline(
                end_time,
                label=("End Time" if i == 2 else None),
                color="r",
                linestyle="-",
                linewidth=0.8,
                alpha=0.8,
            )
        ann_handles = [ann0, ann1, ann2]
        ann_labels = ["Start Time", trans_label, "End Time"]

    # add lines
    if not no_lines and extra_info is not None:
        slopes = extra_info["slopes_min"]
        intercepts = extra_info["intercepts_min"]
        r_values = extra_info["r_values_min"]
        start_end_times = np.array([start_time, end_time], dtype=np.float64)
        ac = slopes * start_end_times + intercepts # y vals of start and end points
        b = slopes[0] * trans_time + intercepts[0] # y val of transition time
        r_sq = r_values**2.0
        line_labels = [
            f"O$_2$ Pulse$ = {slopes[0]:.2f}t + {intercepts[0]:.2f}, R^2 = {r_sq[0]:.4f}$",
            f"O$_2$ Pulse$ = {slopes[1]:.2f}t + {intercepts[1]:.2f}, R^2 = {r_sq[1]:.4f}$",
        ]
        l0 = ax[2].plot(
            [start_time, trans_time],
            [ac[0], b],
            color="k",
            linewidth=1.5,
            linestyle=':',
            label=line_labels[0],
        )
        l1 = ax[2].plot(
            [trans_time, end_time],
            [b, ac[1]],
            color="k",
            linewidth=1.5,
            linestyle='--',
            label=line_labels[1],
        )
        line_handles = [l0, l1]

    # add AUC
    if not no_auc:
        time = np.array(data["Time"], dtype=np.float64)
        o2p = np.array(data["O2-Pulse"], dtype=np.float64)
        during = (time >= data["Start Time"]) * (time <= data["End Time"])
        auc = data['O2-Pulse AUC (mL*min/beat)']
        units_text = r"mL $\bullet$ min / beat"
        auc_label = f"Exercise AUC = {round(auc)} {units_text}"
        auc_handle = ax[2].fill_between(
            x=time[during], 
            y1=np.zeros_like(time[during]), 
            y2=o2p[during],
            interpolate=True,
            facecolor="k",
            edgecolor='none',
            alpha=0.4,
            label=auc_label,
        )

    # creates a legend if needed
    if no_transition_annotations and no_auc and no_lines: # nothing
        pass
    elif no_auc and no_lines: # only transition annotations
        ax[2].legend(
            handles = ann_handles, labels = ann_labels,
            loc='upper center', bbox_to_anchor=(0.5, -0.5),
            fancybox=False, shadow=False, 
            ncol=1
        )
    elif no_transition_annotations and no_auc: # only lines
        ax[2].legend(
            handles = line_handles, labels = line_labels,
            loc='upper center', bbox_to_anchor=(0.5, -0.5),
            fancybox=False, shadow=False, 
            ncol=1
        )
    elif no_transition_annotations and no_lines: # only auc
        ax[2].legend(
            handles = [auc_handle], labels = [auc_label],
            loc='upper center', bbox_to_anchor=(0.5, -0.5),
            fancybox=False, shadow=False, 
            ncol=1
        )
    elif no_transition_annotations: # lines and auc
        ax[2].legend(
            handles = [*line_handles, auc_handle], labels = [*line_labels, auc_label],
            loc='upper center', bbox_to_anchor=(0.5, -0.5),
            fancybox=False, shadow=False, 
            ncol=1
        )
    elif no_lines: # transition_annotations and auc
        ax[2].legend(
            handles = [*ann_handles, auc_handle], labels = [*ann_labels, auc_label],
            loc='upper center', bbox_to_anchor=(0.5, -0.5),
            fancybox=False, shadow=False, 
            ncol=2
        )
    elif no_auc: # transition_annotations and lines
        ax[2].legend(
            handles = [*ann_handles, *line_handles], labels = [*ann_labels, *line_labels],
            loc='upper center', bbox_to_anchor=(0.5, -0.5),
            fancybox=False, shadow=False, 
            ncol=2
        )
    else: # all 3 sets
        ax[2].legend(
            loc='upper center', bbox_to_anchor=(0.5, -0.5),
            fancybox=False, shadow=False, ncol=2
        )

    fig.tight_layout()

    ident = data["ID"]
    instance = data.get("Instance", "-")
    study_type = data.get("Study Type", "-")
    name = f"{ident}_{instance}_{study_type}_time{tag}.{fmt}"
    path = output_dir / name

    fig.savefig(output_dir / name, format=fmt, dpi=dpi)
    plt.close()
    path.chmod(0o660)

    return None


def plot_data_nondimensional(
        data: Mapping[
            Literal[
                "ID", "Instance", "Study Type", "Time", "HR", "VO2", "O2-Pulse",
                "Start Time", "True Transition Time", "Transition Time (min)",
                "End Time", 'Flattening Fraction',
            ], Any
        ],
        extra_info: Optional[RESULTS_INFO_TYPE],
        output_dir: Path,
        no_lines: bool = True,
        no_transition_annotations: bool = False,
        no_auc: bool = False,
        fmt: Literal["png","svg","pdf"] = "png",
        dpi: int = 600,
) -> None:
    """ plots HR, VO2, HR, and O2-Pulse data for reference """
    assert isinstance(output_dir, (Path,PosixPath,WindowsPath))

    fig, ax = plt.subplots(3, 1, sharex=True, dpi=dpi)
    ax: tuple[Axes, ...]
    ax[0].scatter(extra_info["time_nd"], data["HR"], s=0.8, c="k", alpha=0.6)
    ax[0].set_ylabel("Heart Rate\n(beats/min)")
    ax[1].scatter(extra_info["time_nd"], data["VO2"], s=0.8, c="k", alpha=0.6)
    ax[1].set_ylabel("VO2\n(mL/min)")
    ax[2].scatter(extra_info["time_nd"], data["O2-Pulse"], s=0.8, c="k", alpha=0.6)
    ax[2].set_ylabel(f"O$_2$ Pulse\n(mL/beat)")
    ax[2].set_xlabel("Exercise Fraction")

    start_time = 0
    end_time = 1
    trans_time = data['Flattening Fraction']

    if not no_transition_annotations:
        trans_label = "Transition Time"
        for i in range(3):
            ann0 = ax[i].axvline(
                start_time,
                label=("Start Time" if i == 2 else None),
                color="g",
                linestyle=":",
                linewidth=0.8,
                alpha=0.8,
            )
            ann1 = ax[i].axvline(
                trans_time,
                label=(trans_label if i == 2 else None),
                color="b",
                linestyle="-.",
                linewidth=0.8,
                alpha=0.8,
            )
            ann2 = ax[i].axvline(
                end_time,
                label=("End Time" if i == 2 else None),
                color="r",
                linestyle="-",
                linewidth=0.8,
                alpha=0.8,
            )
        ann_handles = [ann0, ann1, ann2]
        ann_labels = ["Start Time", trans_label, "End Time"]

    # add lines
    if not no_lines and extra_info is not None:
        slopes = extra_info["slopes_nd"]
        intercepts = extra_info["intercepts_nd"]
        r_values = extra_info["r_values_nd"]
        start_end_times = np.array([start_time, end_time], dtype=np.float64)
        ac = slopes * start_end_times + intercepts # y vals of start and end points
        b = slopes[0] * trans_time + intercepts[0] # y val of transition time
        r_sq = r_values**2.0
        tau = r'\tau'
        line_labels = [
            f"O$_2$ Pulse$ = {slopes[0]:.2f}{tau} + {intercepts[0]:.2f}, R^2 = {r_sq[0]:.4f}$",
            f"O$_2$ Pulse$ = {slopes[1]:.2f}{tau} + {intercepts[1]:.2f}, R^2 = {r_sq[1]:.4f}$",
        ]
        l0 = ax[2].plot(
            [start_time, trans_time],
            [ac[0], b],
            color="k",
            linewidth=1.5,
            linestyle=':',
            label=line_labels[0],
        )
        l1 = ax[2].plot(
            [trans_time, end_time],
            [b, ac[1]],
            color="k",
            linewidth=1.5,
            linestyle='--',
            label=line_labels[1],
        )
        line_handles = [l0, l1]

    # add AUC
    if not no_auc:
        time = extra_info["time_nd"]
        o2p = np.array(data["O2-Pulse"], dtype=np.float64)
        during = (time >= start_time) * (time <= end_time)
        auc = data['O2-Pulse AUC, nondimensional time (mL/beat)']
        units_text = "mL / beat"
        auc_label = f"Exercise AUC = {round(auc)} {units_text}"
        auc_handle = ax[2].fill_between(
            x=time[during], 
            y1=np.zeros_like(time[during]), 
            y2=o2p[during],
            interpolate=True,
            facecolor="k",
            edgecolor='none',
            alpha=0.4,
            label=auc_label,
        )

    # creates a legend if needed
    if no_transition_annotations and no_auc and no_lines: # nothing
        pass
    elif no_auc and no_lines: # only transition annotations
        ax[2].legend(
            handles = ann_handles, labels = ann_labels,
            loc='upper center', bbox_to_anchor=(0.5, -0.5),
            fancybox=False, shadow=False, 
            ncol=1
        )
    elif no_transition_annotations and no_auc: # only lines
        ax[2].legend(
            handles = line_handles, labels = line_labels,
            loc='upper center', bbox_to_anchor=(0.5, -0.5),
            fancybox=False, shadow=False, 
            ncol=1
        )
    elif no_transition_annotations and no_lines: # only auc
        ax[2].legend(
            handles = [auc_handle], labels = [auc_label],
            loc='upper center', bbox_to_anchor=(0.5, -0.5),
            fancybox=False, shadow=False, 
            ncol=1
        )
    elif no_transition_annotations: # lines and auc
        ax[2].legend(
            handles = [*line_handles, auc_handle], labels = [*line_labels, auc_label],
            loc='upper center', bbox_to_anchor=(0.5, -0.5),
            fancybox=False, shadow=False, 
            ncol=1
        )
    elif no_lines: # transition_annotations and auc
        ax[2].legend(
            handles = [*ann_handles, auc_handle], labels = [*ann_labels, auc_label],
            loc='upper center', bbox_to_anchor=(0.5, -0.5),
            fancybox=False, shadow=False, 
            ncol=2
        )
    elif no_auc: # transition_annotations and lines
        ax[2].legend(
            handles = [*ann_handles, *line_handles], labels = [*ann_labels, *line_labels],
            loc='upper center', bbox_to_anchor=(0.5, -0.5),
            fancybox=False, shadow=False, 
            ncol=2
        )
    else: # all 3 sets
        ax[2].legend(
            loc='upper center', bbox_to_anchor=(0.5, -0.5),
            fancybox=False, shadow=False, ncol=2
        )

    fig.tight_layout()

    ident = data["ID"]
    instance = data.get("Instance", "-")
    study_type = data.get("Study Type", "-")
    name = f"{ident}_{instance}_{study_type}_nondim.{fmt}"
    path = output_dir / name

    fig.savefig(output_dir / name, format=fmt, dpi=dpi)
    plt.close()
    path.chmod(0o660)

    return None

