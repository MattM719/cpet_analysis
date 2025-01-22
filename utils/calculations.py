#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Performs the actual calculations
"""

from typing import Any, Dict, Literal, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.integrate import trapezoid

from .utils import META_TYPE, RESULTS_TYPE, RESULTS_INFO_TYPE, time_to_index


# ======================================================================== #
# This is the key function! It fits a penalized bilinear regression curve  #
# to the data                                                              #
# ======================================================================== #


def optimize_plateau(
        x_values: np.ndarray,
        y_values: np.ndarray,
        start_idx: int,
        stop_idx: int,
        plat_method: Literal["cos", "invslope", "none"] = "invslope",
        buffer: float = 0.1,
        tolerance: float = 0.8,
        idnum: str = "-1",
        instance: str = "-1",
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Fits a bilinear regression curve to the data.

    This is the main method. A penalty can be applied to try to select a curve with a positive
    initial slope that flattens out at the end. The "invslope" penalty was used for
    calculations in the paper.

    Args:
        x_values: np array
        y_values: np array
        start_idx: int, index that exercise starts at
        stop_idx: int, index that exercise stops at
        plat_method: str, ['cos', 'angle', 'invslope', 'none' (default)]
        buffer: float, fraction of distance from beginning and from end that plateau must occur at
        tolerance: float, higher tolerance values (0-1) allows more overfitting, lower tolerances
            require more similar R values

        ID and instance numbers can optionally be supplied for help with troubleshooting.

    Returns (tuple):
        index of optimized transition point,
        [slope1, slope2],
        [intercept1, intercept2],
        [rValue1, rValue2]

    Raises:
        ValueError: if unrecognized plateau method ('plat_method') is supplied
    """
    # verify types
    x_values = np.array(x_values, dtype=np.float64).flatten()
    y_values = np.array(y_values, dtype=np.float64).flatten()
    start_idx_orig = int(start_idx)
    stop_idx_orig = int(stop_idx)
    buffer = float(buffer)
    tolerance = float(tolerance)
    plat_method = str(plat_method)

    start_idx = start_idx_orig
    stop_idx = stop_idx_orig

    # interpolate any missing values
    if sum(np.isnan(x_values)) > 0:
        mask = np.isfinite(x_values)
        x_values_len = np.arange(len(x_values))
        try:
            x_values = np.interp(x_values_len, x_values_len[mask], x_values[mask])
        except:
            print(
                "\n*******************************************\n" \
                + f"No x values for {idnum}_{instance}" \
                + "\n*******************************************\n"
            )
            raise
    if sum(np.isnan(y_values)) > 0:
        mask = np.isfinite(y_values)
        y_values = np.interp(x_values, x_values[mask], y_values[mask])

    # create set of possible plateau points
    plat_guess_len = stop_idx + 1 - start_idx
    buffer = int(buffer * plat_guess_len // 1)
    if buffer < 5 and plat_guess_len > 10:
        buffer = 5
    elif buffer < 10 and stop_idx > start_idx:
        result = linregress(
            x_values[start_idx : stop_idx + 1],
            y_values[start_idx : stop_idx + 1],
        )
        return (
            start_idx,
            np.array([np.nan, result.slope], dtype=np.float64),
            np.array([np.nan, result.intercept], dtype=np.float64),
            np.array([np.nan, result.rvalue], dtype=np.float64),
        )
    elif buffer < 10:
        return (
            start_idx,
            np.array([np.nan, np.nan], dtype=np.float64),
            np.array([np.nan, np.nan], dtype=np.float64),
            np.array([np.nan, np.nan], dtype=np.float64),
        )
    plat_guess_len -= int(2 * buffer)  # number of guesses to test (w/ buffer)
    slopes = np.zeros([plat_guess_len, 2], dtype=np.float64)  # store slopes
    intercepts = np.zeros([plat_guess_len, 2], dtype=np.float64)
    residuals = np.zeros(plat_guess_len, dtype=np.float64)
    r_sq_values = np.zeros([plat_guess_len, 2], dtype=np.float64)

    # initialize arrays for linear regression
    x_t = float(x_values[start_idx + buffer])
    matrix_a = np.zeros([stop_idx + 1 - start_idx, 3], dtype=np.float64)  # create array
    matrix_a[: buffer + 1, 0] = x_values[
        start_idx : start_idx + buffer + 1
    ]  # top of column for m1
    matrix_a[buffer + 1 :, 0] = x_t  # bottom of col for m1
    matrix_a[buffer + 1 :, 1] = (
        x_values[start_idx + buffer + 1 : stop_idx + 1] - x_t
    )  # bottom of col for m2
    matrix_a[:, 2] = 1  # column for b1
    y_vec = np.array(y_values[start_idx : stop_idx + 1], dtype=np.float64).reshape(
        (-1, 1)
    )

    # iteratively solve all least squares regressions
    for j in range(plat_guess_len):
        # update arrays - no changes made on first iteration
        x_t_idx = start_idx + buffer + j
        x_t = float(x_values[x_t_idx])  # update transition point
        matrix_a[buffer + j, 0] = x_values[x_t_idx]  # top of column for m1
        matrix_a[buffer + j + 1 :, 0] = x_t  # bottom of col for m1
        matrix_a[buffer + j, 1] = 0  # top of col for m2
        matrix_a[buffer + j + 1 :, 1] = (
            x_values[x_t_idx + 1 : stop_idx + 1] - x_t
        )  # bottom of col for m2

        # least squares regression
        try:
            params, res, rank, singular = np.linalg.lstsq(matrix_a, y_vec, rcond=None)
        except np.linalg.LinAlgError:
            continue
        params = params.flatten()  # m1, m2, b1
        if not isinstance(res, float):
            if len(res) > 1:
                raise IndexError(
                    "Residual has multiple indices. Are you using the correct version of numpy?"
                )
            res = float(res[0])

        # calc R^2
        y_expect1 = params[0] * x_values[start_idx : x_t_idx + 1] + params[2]
        y_expect2 = (
            params[1] * (x_values[x_t_idx + 1 : stop_idx + 1] - x_t) + y_expect1[-1]
        )
        y_mean1 = np.mean(y_values[start_idx : x_t_idx + 1])
        y_mean2 = np.mean(y_values[x_t_idx + 1 : stop_idx + 1])
        rss1 = np.sum(np.square(y_values[start_idx : x_t_idx + 1] - y_expect1))
        rss2 = np.sum(np.square(y_values[x_t_idx + 1 : stop_idx + 1] - y_expect2))
        tss1 = np.sum(np.square(y_values[start_idx : x_t_idx + 1] - y_mean1))
        tss2 = np.sum(np.square(y_values[x_t_idx + 1 : stop_idx + 1] - y_mean2))

        r_squares = (
            1.0
            - np.array(
                [
                    rss1 / tss1,
                    rss2 / tss2,
                ],
                dtype=np.float64,
            ).flatten()
        )

        # store results
        slopes[j, :] = params[:2]  # m1, m2
        intercepts[j, 0] = params[2]  # b1
        intercepts[j, 1] = y_expect1[-1] - params[1] * x_t
        residuals[j] = res  # residual, equal to np.sum(np.square(yAct - yExpect))
        r_sq_values[j, :] = r_squares[:]

    # post processing calculations, prepare to choose optimal curve
    # multiply |R_1| by |R_2|, except 0 if difference > tolerance
    r_sq_values[r_sq_values < 0] = 0
    r_sq_values[r_sq_values > 1] = 1
    r_values = np.sqrt(
        r_sq_values,
        out=np.full_like(r_sq_values, np.nan, dtype=np.float64),
        where=((r_sq_values <= 1) * (r_sq_values >= 0)),
    )
    r_values *= np.sign(slopes)
    r_sq_values_diff = np.abs(np.diff(r_sq_values, axis=1).flatten())
    fit = np.multiply(
        r_values[:, 0],
        r_values[:, 1],
        out=np.zeros(plat_guess_len),
        where=(r_sq_values_diff < tolerance),
    )
    fit = np.sqrt(np.abs(fit))  # ~geometric mean of R values
    fit[slopes[:, 0] < 0.0] = 0.0  # require initial slope to be upward

    # ===================================================
    # apply selected penalty method, choose optimal curve
    # ===================================================

    # penalize non-flat plateaus
    if plat_method == "cos":
        """Takes dot product between unit vector of second regression line and [1;0]
        which is equivalent to cos of the angle away from a flat line.

        If m is the slope of the second regression line:
        factor_i = (1/sqrt(1^2 + m^2)) [1, m] [1;0]
        factor_i = 1/sqrt(1 + m^2)
        """
        factor = np.reciprocal(np.sqrt(1.0 + np.square(slopes[:, 1]))).flatten()
        fit = np.multiply(factor, fit)
        opt_idx = len(fit) - np.nanargmax(fit[::-1]) - 1

    elif plat_method == "invslope":
        # uses the inverse of the slope as a penalty
        fit = np.divide(
            np.abs(slopes[:, 1]).flatten(),
            fit,
            out=np.full_like(fit, np.nan),
            where=(fit > 0),
        )
        mask = np.isfinite(fit)
        indices = np.arange(len(mask), dtype=np.int64) + start_idx + buffer
        try:
            fit = np.interp(x_values[indices][mask], x_values[indices][mask], fit[mask])
        except ValueError:
            return (
                start_idx,
                np.array([np.nan, np.nan], dtype=np.float64),
                np.array([np.nan, np.nan], dtype=np.float64),
                np.array([np.nan, np.nan], dtype=np.float64),
            )

        fit_smoothed = np.convolve(fit, np.array([1, 1, 1]) / 3, mode="valid")
        opt_idx = np.nanargmin(fit_smoothed) + 1

    elif plat_method == "none":
        # return curve with lowest residual
        opt_idx = len(residuals) - np.nanargmin(residuals[::-1]) - 1

    else:
        raise ValueError(f"Unrecognized plat_method '{plat_method}'")

    return (
        int(start_idx + buffer + opt_idx),
        slopes[opt_idx, :],
        intercepts[opt_idx, :],
        r_values[opt_idx, :],
    )


# ======================================================================== #
# other calculations are performed below                                   #
# ======================================================================== #


def nondimensionalize(
    time: np.ndarray, start_idx: int, end_idx: int
) -> tuple[np.ndarray, float]:
    """recomputes time as exercise fraction

    Shifts and rescales time array such that exercise starts at 0 and ends at 1

    Args:
        time: array of time values for each data point
        start_idx: index of start time
        end_idx: index of end time

    Returns (tuple):
        non-dimensionalized time array
        exercise duration (minutes)
    """
    start_time = float(time[int(start_idx)])
    stop_time = float(time[int(end_idx)])
    duration = stop_time - start_time

    time = (np.array(time, dtype=np.float64).flatten() - start_time) / duration

    return time


def add_flattening_fraction(
        study: dict, time: np.ndarray, transition_index: int, nondimensional: bool
) -> None:
    """Adds flattening fraction to the dict ('study')

    Parameters:
    ----------
    study: dict generated from reading meta data  

    time: time array  

    transition_index: transition index  

    nondimensional: true if time was nondimensionalized  
    """
    # transition time and flattening fraction
    if nondimensional:
        study["Flattening Fraction"] = float(time[transition_index])
    else:
        study["Transition Time (min)"] = float(time[transition_index])

    return None


def add_o2p_response_ratio(
        study: dict, slopes: np.ndarray, nondimensional: bool
) -> None:
    """Adds O2-Pulse Response Ratio to the dict ('study')

    Parameters:
    ----------
    study: dict generated from reading meta data  

    slopes: [slope1, slope2]  

    nondimensional: true if time was nondimensionalized
    """
    # transition time and flattening fraction
    if nondimensional:
        key = "O2-Pulse Response Ratio, time=nondimensional"
    else:
        key = "O2-Pulse Response Ratio, time=min"

    study[key] = float(slopes[0] / slopes[1]) if slopes[0] != 0 else float("nan")

    return None


def add_o2p_auc(
    study: dict,
    time: np.ndarray,
    o2p: np.ndarray,
    start_idx: int,
    end_idx: int,
    nondimensional: bool,
) -> None:
    """Adds flattening fraction to the dict ('study')

    Parameters:
    ----------
    study: dict generated from reading meta data  

    time: time array  

    o2p: O2-Pulse  

    start_idx: index of beginning of exercise  

    end_idx: index of end of exercise  

    nondimensional: true if time was nondimensionalized
    """
    start_idx = int(start_idx)
    end_idx = int(end_idx)

    if nondimensional:
        key = "O2-Pulse AUC, nondimensional time (mL/beat)"
    else:
        key = "O2-Pulse AUC (mL*min/beat)"

    # address any non-finite values here, if needed

    study[key] = float(
        trapezoid(o2p[start_idx : end_idx + 1], time[start_idx : end_idx + 1])
    )

    return None


def _get_start_end_indices(meta: META_TYPE, time: np.ndarray) -> tuple[int,int]:
    """Get the start and end indices, either supplied from meta or using the start/end times"""
    start_index = meta.get("Start Index", None)
    end_index = meta.get("End Index", None)

    if start_index is None:
        start_index = time_to_index(meta["Start Time"], time)
    if end_index is None:
        end_index = time_to_index(meta["End Time"], time)
    
    return (int(start_index), int(end_index))


def run_analysis(
        meta: META_TYPE, data: pd.DataFrame
) -> Tuple[RESULTS_TYPE, RESULTS_INFO_TYPE]:
    """ Runs the actual analysis once meta and data are provided """
    time = data["Time"].to_numpy(dtype=np.float64)

    start_index, end_index = _get_start_end_indices(meta, time)
    nd_time = nondimensionalize(
        time, start_idx=start_index, end_idx=end_index,
    )
    o2_pulse = data["O2-Pulse"].to_numpy(dtype=np.float64)

    # find optimized plateau - this is the key method described in the paper.
    # can use minutes or dimensional time
    study: RESULTS_TYPE = {**meta}
    study["Start Index"] = start_index
    study["End Index"] = end_index
    info: Dict[str, Any] = {"time_min": time, "time_nd": nd_time}

    for t, label in zip([time, nd_time], ["min", "nd"]):
        idx_transition, slopes, intercepts, r_values = optimize_plateau(
            t,
            o2_pulse,
            start_idx=int(study["Start Index"]),
            stop_idx=int(study["End Index"]),
            plat_method="invslope",
        )  # r_values are correlation coeffs (r), not coeffs of determination (R^2)
        info[f'slopes_{label}'] = slopes
        info[f'intercepts_{label}'] = intercepts
        info[f'r_values_{label}'] = r_values

        add_flattening_fraction(study, t, idx_transition, (label == "nd"))
        add_o2p_response_ratio(study, slopes, (label == "nd"))
        add_o2p_auc(
            study,
            t,
            o2_pulse,
            study["Start Index"],
            study["End Index"],
            (label == "nd"),
        )


    return (study, info)
