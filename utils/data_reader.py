#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" organizes functions used by the MWE """

import csv
import os
import warnings
from collections.abc import Callable, Generator
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import META_TYPE, DATA_TYPE, time_to_index


KEYS = ["ID", "Instance", "Study Type", "Start Index", "End Index"]


COLUMN_CONVERSION_CINCINNATI = {
    "Time_sec": "Time",
    "Work_Watts": "Work",
    "HR": "HR",
    "VO2_mLmin": "VO2",
    "VO2HR": "O2-Pulse",
}


def _cincinnati_meta_reader(row: dict[str, Any]) -> META_TYPE:
    """ Reads the cincinnati data format and converts it to the UW format """
    fmt = {
        "ID": 'PatientID', 
        "Start Time": 'StartExercise', 
        "End Time": 'StartRecovery', 
    }
    data = {k1: row.get(k2, row.get(k1,None)) for k1, k2 in fmt.items()}
    data["Instance"] = row.get("Instance", None)
    data["Study Type"] = row.get("Study Type", None)

    return data


def validate_meta(meta: META_TYPE, set_default: bool = True) -> META_TYPE:
    """ Ensures the correct type is used for each entry """
    new_meta: META_TYPE = {'ID': int(meta['ID'])}

    # "ID", "Instance", "Study Type"
    instance = meta.get("Instance", None)
    study_type = meta.get("Study Type", None)

    new_meta['Instance'] = None if instance is None else int(instance)
    if study_type is not None:
        new_meta["Study Type"] = str(study_type).strip().upper()
    else:
        new_meta["Study Type"] = None

    # "Start Index", "Start Time", "End Index", "End Time"
    start_index = meta.get("Start Index", None)
    start_time = meta.get("Start Time", None)
    end_index = meta.get("End Index", None)
    end_time = meta.get("End Time", None)

    new_meta["Start Index"] = None if start_index is None else int(start_index)
    new_meta["End Index"] = None if end_index is None else int(end_index)
    new_meta["Start Time"] = None if start_time is None else float(start_time)
    new_meta["End Time"] = None if end_time is None else float(end_time)

    # "File Path"
    file_path = meta.get("File Path", None)
    new_meta["File Path"] = None if file_path is None else file_path

    return new_meta


def meta_data_iterator(meta_data_path: str, layout: Literal['uw','cincinnati']) -> Generator[META_TYPE, None, None]:
    """uses meta data file to lazily read data"""
    with open(meta_data_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, dialect="excel", lineterminator="\n")
        for row in reader:
            if layout == "uw":
                meta = {key: row[key] for key in KEYS}
            elif layout == "cincinnati":
                meta = _cincinnati_meta_reader(row)
            else:
                raise ValueError(f"Unrecognized meta_format '{meta_format}'")
            
            yield meta


def find_valid_name_wrapper(
        input_path: Path, meta_path: Path
) -> Callable[[Any,Optional[Any],Optional[Any]],tuple[Path,Optional[str]]]:
    """Wrapper function, initializes a callable to find a spreadsheet (CSV or Excel 
    sheet) for a dataset.

    Parameters:
    ----------
    input_path: path to a directory of CSV files or a 'merged' excel file with many 
    sheets.  

    meta_path: path to the meta data file
    """
    is_dir: bool = input_path.is_dir()

    if input_path.is_file() and input_path.suffix in [".xlsx", ".xls"]:
        with pd.ExcelFile(input_path, engine='openpyxl') as f:
            sheet_names: set[str] = set(f.sheet_names)

    elif is_dir:
        tmp = set(input_path.glob("*.csv")).union(
            input_path.glob("*.xls")
        ).union(
            input_path.glob("*.xlsx")
        )
        tmp.discard(meta_path)

        sheet_names: set[str] = {p.name for p in tmp}
    
    else:
        raise FileNotFoundError(f"Could not find a valid input path from: {input_path}")

    def find_valid_name(
            id: Any, instance: Optional[Any], study_type: Optional[Any]
    ) -> tuple[Path, Optional[str]]:
        """Matches supplied data to a valid file/sheet.  

        Parameters:
        ----------
        id: "ID" column of the meta data file.  
        
        instance: "Instance" column of the meta data file. Default: 1.  
        
        study_type: "Study Type" column of the meta data file. Default: 'CPET'.  

        Returns:
        -------
        (path to the relevant data file, a sheet name if needed)
        """
        _instance = 1 if instance is None else instance
        _study_type = "CPET" if study_type is None else study_type

        candidates: set[str] = {
            f'{id}_{_instance}_{_study_type}',
            f'{id}_{_instance}',
            f'{id}',
        }

        # add extensions to the candidates' names, if needed
        if is_dir:
            orig_candidates: list[str] = [str(c) for c in candidates]
            candidates = set()
            for ext in [".csv", ".xls", ".xlsx"]:
                candidates.update({c + ext for c in orig_candidates})
        
        matched = [n for n in sheet_names.intersection(candidates)]

        # confirm only one spreadsheet was found unambiguously
        if len(matched) == 0:
            raise FileNotFoundError(
                "Could not find a spreadsheet matching "
                f"ID='{id}', Instance='{instance}', Study Type='{study_type}'"
            )
        elif len(matched) > 1:
            raise FileNotFoundError(
                'Could not match to a non-ambiguous spreadsheet. '
                f'Found: {matched}'
            )
        
        name = str(matched[0])

        if is_dir:
            return (input_path / name, None)
        return (input_path, name)
    
    return find_valid_name


def _process_missing_sheet(path: Path) -> Optional[str]:
    """Processes an Excel file supplied without a specified sheet"""
    with pd.ExcelFile(path, "openpyxl") as f:
        all_sheet_names = [str(s).strip().replace(" ","_") for s in f.sheet_names]
    if (count := len(all_sheet_names)) != 1:
        raise FileNotFoundError(f"An excel file was supplied with {count} sheets was provided, but no sheet was specified.")
    
    sheet_name = str(all_sheet_names[0])
    n_in_sheet = len(sheet_name.split("_"))
    n_in_path = len(str(path.stem).split("_"))

    if n_in_path <= 3 and n_in_sheet <= 3:
        options = [None, sheet_name]
        errs = np.array([n_in_path, n_in_sheet], dtype=np.float64)
        errs -= 1.5
        return options[np.argmin(np.abs(errs))]
    elif n_in_path <= 3:
        return None
    elif n_in_sheet <= 3:
        return sheet_name
    return None


def read_study_data(
        path: Path,
        layout: Literal["uw","cincinnati"],
        recalculate_o2_pulse: bool, 
        sheet_name: Optional[str] = None,
) -> tuple[pd.DataFrame, Optional[dict[str,str]], Optional[str]]:
    """Converts data from a CSV into a DataFrame
    """
    if path.suffix in [".xlsx", ".xls"] and sheet_name is None:
        sheet_name = _process_missing_sheet(path)
        full_df = pd.read_excel(path)

    elif path.suffix in [".xlsx", ".xls"]:
        full_df = pd.read_excel(path, sheet_name=sheet_name)
    
    elif path.suffix == ".csv":
        sheet_name = None
        full_df = pd.read_csv(path)
    else:
        raise FileNotFoundError(f"Did not recognize extension '{path.suffix}'")

    if layout == "cincinnati":
        df: pd.DataFrame = full_df.loc[:,list(COLUMN_CONVERSION_CINCINNATI.keys())]\
        .rename(columns=COLUMN_CONVERSION_CINCINNATI, copy=True)

    elif layout == "uw":
        df = full_df.loc[:,["Time","Work","HR","VO2","O2-Pulse"]]

    else:
        raise ValueError(f"Unexpected data_format '{data_format}'")
    
    if layout == "cincinnati":
        units = df.iloc[0,:].to_dict()
        df.drop(0, axis=0, inplace=True)
        df.reset_index(inplace=True, drop=True)
    else:
        units = None
    
    df = df.astype(
        {
            "Time": float,
            "Work": float,
            "HR": float,
            "VO2": float,
            "O2-Pulse": float,
        }
    )

    if layout == "cincinnati":
        if isinstance(units, dict):
            time_units = str(units["Time"]).strip().lower()
            if time_units == 'sec':
                df["Time"] = df["Time"].to_numpy(dtype=np.float64) / 60 # convert seconds to minutes
                units["Time"] = 'min'
            elif time_units == 'min':
                pass
            else:
                raise ValueError(f"Unrecognized units '{time_units}'")
        else:
            warnings.warn(
                "Units not specified. Assuming time units are seconds due to cincinnati layout",
                category=UserWarning,
            )
            df["Time"] = df["Time"].to_numpy(dtype=np.float64) / 60 # convert seconds to minutes

    if recalculate_o2_pulse:
        vo2 = df["VO2"].to_numpy(dtype=np.float64).flatten()
        hr = df["HR"].to_numpy(dtype=np.float64).flatten()
        o2p = df["O2-Pulse"].to_numpy(dtype=np.float64).flatten()
        new_o2p = np.divide(
            vo2, 
            hr, 
            out=o2p, 
            where=(np.isfinite(vo2) * np.isfinite(hr) * (hr > 0) * (vo2 > 0)),
        )
        df["O2-Pulse"] = new_o2p

    
    return (df, units, sheet_name)


def infer_meta(path: str|Path, df: pd.DataFrame, is_sheet: bool) -> META_TYPE:
    """ Infers meta data from a dataframe and the path name """
    meta = {
        "ID": None, 
        "Instance": 0, 
        "Study Type": "CPET",
        "Start Index": None, 
        "Start Time": None, 
        "End Index": None, 
        "End Time": None, 
        "File Path": str(path),
    }

    # update identifiers from filename/sheetname
    if is_sheet:
        name = str(path)
    else:
        path = path if isinstance(path, Path) else Path(path)
        name = path.stem

    ident, *comps = name.split("_")
    meta["ID"] = ident
    
    if len(comps) >= 1:
        meta["Instance"] = int(comps[0])
    if len(comps) >= 2:
        study_type = str(comps[1]).strip().upper()
        if 3 < len(study_type) < 6 and "." not in study_type:
            if study_type in ["CPET","NICOM"]:
                meta["Study Type"] = study_type
            else:
                warnings.warn(f"Invalid Study Type '{comps[1]}'", category=UserWarning)
    
    # constrain df to exercising period (where work > 0)
    df = df.reset_index(inplace=False, drop=True)    
    exercise_df = df[df["Work"] > 0]
    rest_df = df[df["Work"] == 0]

    # update meta with exercise start/end data
    exercise_indices = np.array(exercise_df.index.to_list(), dtype=np.int64)
    exercise_times = exercise_df["Time"].to_numpy(dtype=np.float64).flatten()
    start_index = int(exercise_indices[0])
    meta["Start Index"] = start_index
    meta["Start Time"] = float(exercise_times[0])

    rest_indices = np.array(rest_df.index.to_list(), dtype=np.int64)
    recover_indices = rest_indices[rest_indices > start_index]
    if len(recover_indices) > 0:
        recover_index = int(np.min(recover_indices))
        end_index = int(np.max(exercise_indices[exercise_indices < recover_index]))
    else:
        end_index = int(exercise_indices[-1])

    meta["End Index"] = end_index
    meta["End Time"] = float(exercise_df.loc[end_index, ["Time"]].iloc[0])

    return meta


def read_without_meta(
        path: str|Path, 
        layout: Literal["cincinnati", "uw"],
        recalculate_o2_pulse: bool,
        plot: bool,
        sheet_name: Optional[str] = None,
) -> tuple[META_TYPE, pd.DataFrame, dict[str,str]]:
    """Reads all CSVs in the supplied directory and uses the Work to determine start/end times 
    """
    # validate path
    if not isinstance(path, Path):
        path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is not a directory")

    data, units, sheet_name = read_study_data(
        path=path,
        layout=layout,
        sheet_name=sheet_name,
        recalculate_o2_pulse=recalculate_o2_pulse,
    )

    if isinstance(sheet_name, str):
        meta = infer_meta(sheet_name, df=data, is_sheet=True)
    else:
        meta = infer_meta(path, df=data, is_sheet=False)

    return (meta, data, units)


def fix_time(meta: META_TYPE, data: DATA_TYPE|pd.DataFrame, units: Dict[str, str]) -> Tuple[META_TYPE, DATA_TYPE|pd.DataFrame, Dict[str, str]]:
    """Fixes time
    """
    file_name = meta.get("File Path")
    start_t = meta.get("Start Time", None)
    end_t = meta.get("End Time", None)
    start_i = meta.get("Start Index", None)
    end_i = meta.get("End Index", None)

    t = data["Time"].to_numpy(dtype=np.float64).flatten()

    def align_time_index(time: Optional[float], index: Optional[int]) -> Tuple[float, int]:
        """ ensures time and index are appropriate types and are aligned """
        time: Optional[float] = float(time) if time is not None else time
        index: Optional[int] = int(index) if index is not None else index

        # convert NaN to None
        if time != time:
            time = None

        # validate/align start time and index
        if time is None and index is None:
            raise ValueError(f"No (start/end) time or index provided for '{file_name}'")
        elif time is None:
            time = t[index]
        elif index is None:
            index = time_to_index(time, t)

        if not isinstance(time, float) or not isinstance(index, int):
            raise TypeError(f"Failed to reconcile start time/index. Attempted {time=}{type(time)} -> float, {index=}{type(index)} -> int")

        assert index >= 0
        assert abs(time - t[index]) < 1, f"{time=} did not match {t[index]=} for index {index}.\nmeta:\n{pformat(meta)}"

        return (time, index)

    start_t, start_i = align_time_index(start_t, start_i)
    end_t, end_i = align_time_index(end_t, end_i)
    
    assert start_t <= end_t
    assert start_i <= end_i

    meta["Start Time"] = start_t
    meta["Start Index"] = start_i
    meta["End Time"] = end_t
    meta["End Index"] = end_i

    return (meta, data, units)

