#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" organizes functions used by the MWE """

import csv
from pathlib import Path
from pprint import pformat
from typing import Generator, Literal, Any

import pandas as pd


KEYS = ["ID", "Instance", "Study Type", "Start Index", "End Index"]


def _cincinnati_reader(row: dict[str, Any]) -> dict[str, Any]:
    """ Reads the cincinnati data format and converts it to the UW format """
    fmt = {
        "ID": 'PatientID', 
        "Start Index": 'StartExercise', 
        "End Index": 'StartRecovery', 
    }
    data = {k1: row[k2] for k1, k2 in fmt.items()}
    data["Instance"] = row.get("Instance", None)
    data["Study Type"] = row.get("Study Type", "CPET")

    return data


def meta_data_iterator(meta_data_path: str, meta_format: Literal['UW','Cincinnati']) -> Generator[dict, None, None]:
    """uses meta data file to lazily read data"""
    with open(meta_data_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, dialect="excel", lineterminator="\n")
        for row in reader:
            if meta_format == "UW":
                data = {key: row[key] for key in KEYS}
            elif meta_format == "Cincinnati":
                data = _cincinnati_reader(row)
            else:
                raise ValueError(f"Unrecognized meta_format '{meta_format}'")
            yield data


def get_study_data_excel(study_data_path: str, sheet_name: str) -> pd.DataFrame:
    """Converts data from merged excel file into a DataFrame"""

    # the reported O2-Pulse was truncated to the nearest whole number,
    # whereas other measurements were reported with decimal precision.
    # We would recalculate the O2-Pulse whenever possible
    # O2-Pulse [mL/beat] = VO2 [mL/min] / HR [beat/min]

    # other cleaning/pre-processing steps can go here, as needed

    return pd.read_excel(study_data_path, sheet_name=sheet_name)

def get_study_data_csv(data_dir: Path, names: set[str], info: dict[Literal["ID","Instance","Study Type"], Any]) -> pd.DataFrame:
    """ Converts data from a CSV into a DataFrame """
    candidates = {
        f'{info["ID"]}_{info["Instance"]}_{info["Study Type"]}.csv',
        f'{info["ID"]}_{info["Instance"]}.csv',
        f'{info["ID"]}.csv',
    }

    matched = names.intersection(candidates)
    if len(matched) == 0:
        raise FileNotFoundError(f"Could not find a file matching:\n" + pformat(info))
    elif len(matched) > 1:
        raise FileNotFoundError(
            f"Multiple potential matches were found for:\n" \
            + pformat(info) + "\n  -> " + pformat(matched)
        )
    
    file = data_dir / list(matched)[0]
    df = pd.read_csv(file)

    return df
