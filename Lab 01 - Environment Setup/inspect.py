"""
Lab 01: Environment Setup — Data Inspection Utility
Course: Introduction to Artificial Intelligence (AI-101L)
Instructor: Dr. Muhammad Awais

This module provides a robust, reusable dataset inspection function:
    inspect_dataset(path_or_url)
that loads a CSV and outputs a structured, professional first-look report
including shape, column types, missing values (counts and percentages),
descriptive statistics, and relative dispersion (coefficient of variation).
"""

import sys
import os

# Prevent local script filename from shadowing standard library 'inspect'
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir in sys.path:
    sys.path.remove(_current_dir)
import inspect as _stdlib_inspect
sys.modules['inspect'] = _stdlib_inspect
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import pandas as pd
import numpy as np


def inspect(path_or_url: str):
    """
    Load a CSV and print a compact, standard first-look report as required
    by the Home Assignment specification.

    Parameters
    ----------
    path_or_url : str
        A local file path or an http(s) URL pointing at a CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    print("=" * 80)
    print("SOURCE :", path_or_url)
    try:
        df = pd.read_csv(path_or_url)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return None

    print(f"SHAPE  : {df.shape[0]} rows x {df.shape[1]} columns")
    print("=" * 80)

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES ---")
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if miss.empty:
        print("None. (Dataset is complete)")
    else:
        report = pd.DataFrame({
            "missing": miss,
            "percent": (miss / len(df) * 100).round(2),
        })
        print(report.sort_values("missing", ascending=False))

    print("\n--- NUMERIC SUMMARY ---")
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        desc = numeric_df.describe().T
        desc["std/mean (CV)"] = (desc["std"] / desc["mean"].replace(0, np.nan)).abs().round(3)
        print(desc[["count", "mean", "std", "min", "50%", "max", "std/mean (CV)"]])
        top_cv = desc["std/mean (CV)"].dropna().sort_values(ascending=False)
        if not top_cv.empty:
            print(f"\nFeature with highest relative dispersion (CV = std/mean): {top_cv.index[0]} ({top_cv.iloc[0]})")
    else:
        print("No numerical features found.")

    print("\n--- CATEGORICAL SUMMARY ---")
    cat_df = df.select_dtypes(include=["object", "category", "bool"])
    if not cat_df.empty:
        print(cat_df.describe().T)
    else:
        print("No categorical features found.")

    return df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
        inspect(target)
    else:
        print("Running automated demonstration on two benchmark public datasets...\n")
        
        # Dataset 1: Titanic (Disaster survival prediction benchmark)
        titanic_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
        print(">>> DATASET 1: TITANIC PASSENGER DATA")
        inspect(titanic_url)
        
        print("\n\n" + "#" * 80 + "\n\n")
        
        # Dataset 2: Iris (Multivariate botanical classification benchmark)
        iris_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        print(">>> DATASET 2: IRIS FLOWER MEASUREMENTS")
        inspect(iris_url)
