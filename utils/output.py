#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Outputs data

"""

import csv


def save_spreadsheet(path: str, results: list) -> None:
    """Saves a spreadsheet of calculation results

    Args:
        results: list of 'study' dicts with all the parameters of interest
    """
    column_names = [
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

    with open(path, "w", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=column_names, dialect="excel", lineterminator="\n"
        )
        writer.writeheader()
        for study in results:
            row = {col: study.get(col, None) for col in column_names}
            writer.writerow(row)
