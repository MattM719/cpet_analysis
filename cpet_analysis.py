#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Code for fitting a penalized bilinear regression curve to CPET data.

Expects to find an initial increase in O2-Pulse, then expects the O2-Pulse to stop
changing.

Please read the associated publication for more information.
DOI: 10.1016/j.ijcchd.2024.100539

Usage examples:
--------------
# UW formatted data in a merged excel file, supply meta data
python cpet_analysis.py -pr -l uw -m pseudo_data/meta_data.csv pseudo_data/merged_data.xlsx  pseudo_data/results

# UW formatted data in a merged excel file, do not provide meta data
python cpet_analysis.py -pr -l uw pseudo_data/merged_data.xlsx  pseudo_data/results

# Cincinnati formatted CSV files, supply meta data
python cpet_analysis.py -pr -l cincinnati -m pseudo_data/meta_data.csv pseudo_data/cincinnati  pseudo_data/results

# Cincinnati formatted CSV files, do not provide meta data
python cpet_analysis.py -pr -l cincinnati pseudo_data/cincinnati  pseudo_data/results
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
__version__ = "1.2.2"

import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

import pandas as pd
from tqdm import tqdm

from utils import DATA_TYPE, RESULTS_TYPE, RESULTS_INFO_TYPE
from utils.data_reader import (
    find_valid_name_wrapper,
    meta_data_iterator,
    read_study_data,
    read_without_meta,
    fix_time,
)
from utils.calculations import run_analysis
from utils.output import (
    save_spreadsheet,
    plot_data_time,
    plot_data_nondimensional,
    update_spreadsheet,
)


# ==================================================================================== #
# This code offers a command line interface, or arguments may be supplied manually     #
# below.                                                                               #
# ==================================================================================== #

# If True, all other variables in this section will be ignored. Type
# `python cpet_analysis.py --help` into your terminal for info on using the CLI.
# If False, the CLI is deactivated and arguments are set in this section.
USE_COMMAND_LINE_INTERFACE: bool = True

# Path to input data (a directory of CSV files or an Excel file with many sheets).
# The expected sheet (file) naming format is "[ID]_[Instance]_CPET(.csv)", for example
# 4_1_CPET (4_1_CPET.csv), for pationt 4's first CPET study. The study type is assumed
# to be "CPET" if that is not explicitly indicated. The instance also does not need to
# be supplied, but is highly recommended. 4_1.csv or simply 4.csv would be valid file
# names, but at least providing the instance is strongly recomended.
INPUT_PATH: str = ...

OUTPUT_DIRECTORY: str = ...

META_DATA_PATH: Optional[str] = None
LAYOUT: Literal["uw", "cincinnati"] = "cincinnati"
RECALCULATE_O2_PULSE: bool = True

# plot settings
PLOT: bool = True
NO_LINES: bool = False
NO_TRANSITION_ANNOTATIONS: bool = False
NO_AUC: bool = False


# ==================================================================================== #
# Do not modify code beyond this point                                                 #
# ==================================================================================== #


def process_args(*args) -> dict[str, Any]:
    """Processes command line arguments"""
    parser = argparse.ArgumentParser(
        prog="CPET_ANALYSIS",
        description="Analyzes O2-Pulse data from cardiopulmonary exercise testing "
        + "(CPET) data.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        exit_on_error=True,
    )

    # required positional arguments
    required = parser.add_argument_group("Required positional arguments")
    required.add_argument(
        "input",
        type=Path,
        help="Path to the merged study data file or a directory containing the study"
        + " data.",
    )
    required.add_argument(
        "output",
        type=Path,
        help="Path to the directory where results should be saved.",
    )

    # optional arguments
    optional = parser.add_argument_group("Optional arguments")

    optional.add_argument(
        "-m",
        "--meta-data-path",
        type=Path,
        help="If meta data should be referenced , supply the path to the meta data "
        + "file with each sheet representing a different study. If this argument is "
        + 'not supplied, a unique CSV file is expected in the "input" path for each '
        + "study.",
    )
    optional.add_argument(
        "-l",
        "--layout",
        type=str,
        choices=["uw", "cincinnati"],
        default="cincinnati",
        help='The "cincinnati" layout is expected by default, which is auto-generated'
        + " during mass export of CPET data. Specific column headers are used and the"
        + ' second row contains the units for each column. The "uw" format may also'
        + " be used. This expects a simplified and more condensed spreadsheet layout."
        + " See the pseudo_data/ folder for example data formats.",
    )
    optional.add_argument(
        "-r",
        "--recalculate-o2-pulse",
        action="store_true",
        help="Recalculates the O2-Pulse from the VO2 and HR.",
    )

    optional.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Generates plots to visualize the data analysis.",
    )

    optional.add_argument(
        "--no-lines",
        action="store_true",
        help="Remove regression lines from the plot.",
    )
    optional.add_argument(
        "--no-transition-annotations",
        action="store_true",
        help="Remove transition annotations from the plot.",
    )
    optional.add_argument(
        "--no-auc",
        action="store_true",
        help="Remove AUC shading from the plot.",
    )

    # parse arguments
    def drop_empty(x: Any) -> bool:
        return str(x) != ""
    args = [str(s) for s in filter(drop_empty, args)] if len(args) > 0 else None

    requests = parser.parse_args(args=args)
    config = requests.__dict__

    # validation
    config["layout"] = str(config["layout"]).lower()
    if config["layout"] not in ["uw", "cincinnati"]:
        raise ValueError("data_layout must be 'uw' or 'cincinnati'")

    return config


def _update_plot_settings(**updates: Mapping[str, Any]) -> Dict[str, Any]:
    """creates new plot settings"""
    plotting_params: Dict[str, Any] = {
        "no_lines": False,  # bool
        "no_transition_annotations": False,  # bool
        "no_auc": False,  # bool
        "dpi": 600,  # int
        "image_format": "png",  # Literal['png','svg','pdf']
    }
    for kw in updates:
        if kw not in plotting_params:
            raise NotImplementedError(
                f"Unable to process '{kw}': '{plotting_params[kw]}'"
            )
    plotting_params.update(updates)
    return plotting_params


def plot_study(
    data: DATA_TYPE | pd.DataFrame,
    study: RESULTS_TYPE,
    extra_info: RESULTS_INFO_TYPE,
    output_dir: Path,
    image_format: Literal["png", "svg", "pdf"] = "png",
    dpi: int = 600,
    no_lines: bool = False,
    no_transition_annotations: bool = False,
    no_auc: bool = False,
) -> None:
    """Simple function to send study data to plotting function"""
    plot_dir = output_dir / "plots"
    if not plot_dir.is_dir():
        plot_dir.mkdir(0o751)

    if isinstance(data, pd.DataFrame):
        data = data.to_dict(orient="list")

    merged = data | study

    plot_data_time(
        data=merged,
        extra_info=extra_info,
        output_dir=plot_dir,
        is_synthetic=False,
        no_lines=no_lines,
        no_transition_annotations=no_transition_annotations,
        no_auc=no_auc,
        fmt=image_format,
        dpi=dpi,
    )

    plot_data_nondimensional(
        data=merged,
        extra_info=extra_info,
        output_dir=plot_dir,
        no_lines=no_lines,
        no_transition_annotations=no_transition_annotations,
        no_auc=no_auc,
        fmt=image_format,
        dpi=dpi,
    )

    return None


def use_supplied_meta(
    meta_path: Path,
    input_path: Path,
    output_dir: Path,
    layout: Literal["cincinnati", "uw"],
    recalculate_o2_pulse: bool,
    plot: bool,
    return_results: bool,
    plot_settings: Dict[str, Any],
) -> tuple[Optional[list[RESULTS_TYPE]], Optional[list[DATA_TYPE]]]:
    """Parses files if meta data are explicitly supplied. Expects UW (orginal) merged
    data format.
    """
    find_valid_name = find_valid_name_wrapper(input_path=input_path, meta_path=meta_path)

    if return_results:
        all_data = []
        results = []
    else:
        all_data = None
        results = None

    n = len(list(meta_data_iterator(meta_path, layout=layout)))
    iterator = meta_data_iterator(meta_path, layout=layout)

    for meta in tqdm(iterator, desc="Analyzing", total=n):
        # get data
        file, sheet_name = find_valid_name(
            id=meta.get("ID"),
            instance=meta.get("Instance", None),
            study_type=meta.get("Study Type", None),
        )
        data, units, _ = read_study_data(
            file,
            layout=layout,
            sheet_name=sheet_name,
            recalculate_o2_pulse=recalculate_o2_pulse,
        )
        meta, data, units = fix_time(meta, data, units)

        # TODO: verify units are correct

        study, extra_info = run_analysis(meta=meta, data=data)

        update_spreadsheet(path=output_dir / "results.csv", result=study)

        if plot:
            plot_study(
                data=data,
                study=study,
                extra_info=extra_info,
                output_dir=output_dir,
                **plot_settings,
            )

        if return_results:
            all_data.append(data.to_dict(orient="list"))
            results.append(study)

    return (results, all_data)


def use_inferred_meta(
    input_path: Path,
    output_dir: Path,
    layout: Literal["cincinnati", "uw"],
    recalculate_o2_pulse: bool,
    plot: bool,
    return_results: bool,
    plot_settings: Dict[str, Any],
) -> tuple[Optional[list[RESULTS_TYPE]], Optional[list[DATA_TYPE]]]:
    """Reads raw CSVs or excel sheets and infers the exercise start/stop time from work."""
    if return_results:
        all_data = []
        results = []
    else:
        all_data = None
        results = None

    if input_path.is_file() and input_path.suffix in [".xlsx", ".xls"]:
        is_excel = True
        with pd.ExcelFile(input_path, engine="openpyxl") as f:
            sheet_names: list[Path] = f.sheet_names
    elif input_path.is_dir() or input_path.suffix == ".csv":
        is_excel = False
        sheet_names: list[Path | str] = sorted(input_path.glob("*.csv"))
    else:
        raise FileNotFoundError(
            f"Could not read spreadsheet with extension '{input_path.suffix}'"
        )

    for name in tqdm(sheet_names, desc="Analyzing", total=len(sheet_names)):
        if is_excel:
            path, sheet_name = input_path, name
        else:
            path, sheet_name = name, None
        
        try:
            meta, data, units = read_without_meta(
                path,
                sheet_name=sheet_name,
                layout=layout,
                recalculate_o2_pulse=recalculate_o2_pulse,
                plot=plot,
            )
        except IndexError as e:
            if str(e) != "Cannot infer meta without nonzero work":
                raise
            n = path.name if sheet_name is None else sheet_name
            warnings.warn(
                f"Cannot infer meta without nonzero work, skipping '{n}'",
                category=UserWarning,
            )
            continue
        
        #TODO: verify units are correct

        study, extra_info = run_analysis(meta=meta, data=data)

        update_spreadsheet(path=output_dir / "results.csv", result=study)
        if plot:
            plot_study(
                data=data,
                study=study,
                extra_info=extra_info,
                output_dir=output_dir,
                **plot_settings,
            )

        if return_results:
            all_data.append(data.to_dict(orient="list"))
            results.append(study)

    return (results, all_data)


def main(
    input: Path,
    output: Path,
    meta_data_path: Optional[Path],
    layout: Literal["cincinnati", "uw"],
    recalculate_o2_pulse: bool,
    plot: bool,
    return_results: bool = False,
    **format_plot,
) -> None:
    """receives user-specified arguments and"""
    plot_settings = _update_plot_settings(**format_plot)

    # avoid overwriting previous results
    if output.exists():
        now = datetime.now().strftime("%m%d%y_%H%M%S")
        output /= f"results_{now}"
        output.mkdir(0o750)
    else:
        output.mkdir(0o750)

    # decide whether to use inferred or explicit meta data
    if meta_data_path is None:
        results, all_data = use_inferred_meta(
            input_path=input,
            output_dir=output,
            layout=layout,
            recalculate_o2_pulse=recalculate_o2_pulse,
            plot=plot,
            return_results=return_results,
            plot_settings=plot_settings,
        )
    else:
        results, all_data = use_supplied_meta(
            input_path=input,
            output_dir=output,
            meta_path=meta_data_path,
            layout=layout,
            recalculate_o2_pulse=recalculate_o2_pulse,
            plot=plot,
            return_results=return_results,
            plot_settings=plot_settings,
        )

    if return_results and False:
        save_spreadsheet(output / "results.csv", results)


if __name__ == "__main__":
    if USE_COMMAND_LINE_INTERFACE:
        configs = process_args()
    else:
        artificial = [
            str(INPUT_PATH),
            str(OUTPUT_DIRECTORY),
            "-l",
            str(LAYOUT),
        ]
        if META_DATA_PATH is not None:
            artificial.extend(["-m", str(META_DATA_PATH)])
        if RECALCULATE_O2_PULSE:
            artificial.append("-r")
        if PLOT:
            artificial.append("-p")
        if NO_LINES:
            artificial.append("--no-lines")
        if NO_TRANSITION_ANNOTATIONS:
            artificial.append("--no-transition-annotations")
        if NO_AUC:
            artificial.append("--no-auc")

        configs = process_args(*artificial)

    main(**configs)
