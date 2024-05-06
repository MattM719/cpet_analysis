#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" organizes functions used by the MWE """

import csv
from typing import Generator

import pandas as pd


def meta_data_iterator(meta_data_path: str) -> Generator[dict, None, None]:
    """uses meta data file to lazily read data"""
    with open(meta_data_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, dialect="excel", lineterminator="\n")
        for row in reader:
            data = {
                key: row[key]
                for key in ["ID", "Instance", "Study Type", "Start Index", "End Index"]
            }
            yield data


def get_study_data(study_data_path: str, sheet_name: str) -> pd.DataFrame:
    """Converts data from merged excel file into a DataFrame"""

    # the reported O2-Pulse was truncated to the nearest whole number,
    # whereas other measurements were reported with decimal precision.
    # We would recalculate the O2-Pulse whenever possible
    # O2-Pulse [mL/beat] = VO2 [mL/min] / HR [beat/min]

    # other cleaning/pre-processing steps can go here, as needed

    return pd.read_excel(study_data_path, sheet_name=sheet_name)
