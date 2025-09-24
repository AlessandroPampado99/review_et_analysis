# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 09:32:07 2025

@author: aless
"""

# - Minimal runner to load config, read Excel excluding last N columns, optionally flatten columns,
#   execute a declarative pipeline (registered steps), and save final CSV.

import os
from pathlib import Path
import pandas as pd
import yaml

from steps import STEP_REGISTRY  # registry and steps live together for simplicity


def ensure_parent_dir(path: str):
    """Ensure the parent directory of 'path' exists."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _count_excel_columns(excel_path: str, sheet: str, header: int = 0) -> int:
    """Read only header to quickly detect number of columns."""
    df_head = pd.read_excel(excel_path, sheet_name=sheet, header=header, nrows=0, engine="openpyxl")
    return df_head.shape[1]


def read_excel_excluding_last(
    excel_path: str,
    sheet: str = "Sheet1",
    header: int = 0,
    exclude_last_cols: int = 3,
    **kwargs
) -> pd.DataFrame:
    """
    Efficiently read an Excel sheet excluding the last N columns by
    first detecting column count, then passing an explicit usecols range.
    """
    ncols = _count_excel_columns(excel_path, sheet, header)
    use_until = max(0, ncols - max(0, exclude_last_cols))
    if use_until == 0:
        return pd.DataFrame()
    usecols = list(range(use_until))
    df = pd.read_excel(
        excel_path,
        sheet_name=sheet,
        header=header,
        engine="openpyxl",
        usecols=usecols,
        **kwargs
    )
    return df


def flatten_columns(
    df: pd.DataFrame,
    sep: str = " | ",
    lower: bool = False,
    strip: bool = True
) -> pd.DataFrame:
    """Flatten a MultiIndex column to a single level by joining levels with 'sep'."""
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for tup in df.columns:
            parts = [str(x) for x in tup if x is not None]
            name = sep.join(parts)
            if strip:
                name = name.strip()
            if lower:
                name = name.lower()
            new_cols.append(name)
        df = df.copy()
        df.columns = new_cols
    return df


def load_yaml_config(path: str) -> dict:
    """Load YAML as a plain dict (no pydantic for simplicity)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_dataframe(cfg_input: dict) -> pd.DataFrame:
    """Load DataFrame from Excel (excluding last N columns) + optional flatten and cache."""
    excel_path = cfg_input.get("excel_path")
    sheet = cfg_input.get("sheet", "Sheet1")
    header = int(cfg_input.get("header", 0))
    exclude_last_cols = int(cfg_input.get("exclude_last_cols", 3))
    cache_parquet = bool(cfg_input.get("cache_parquet", False))
    cache_path = cfg_input.get("cache_path", "./cache/data.parquet")

    flatten = bool(cfg_input.get("flatten_columns", True))
    flatten_sep = cfg_input.get("flatten_sep", " | ")
    flatten_lower = bool(cfg_input.get("flatten_lower", False))
    flatten_strip = bool(cfg_input.get("flatten_strip", True))

    if cache_parquet and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
    else:
        df = read_excel_excluding_last(
            excel_path=excel_path,
            sheet=sheet,
            header=header,
            exclude_last_cols=exclude_last_cols,
        )
        if flatten:
            df = flatten_columns(df, sep=flatten_sep, lower=flatten_lower, strip=flatten_strip)
        if cache_parquet:
            ensure_parent_dir(cache_path)
            df.to_parquet(cache_path)

    return df


def run_pipeline(config: dict):
    """
    Execute the pipeline described by the YAML config.
    Each step function must have signature: (df, ctx, **params) -> (df, ctx)
    """
    ctx = {"artifacts": []}
    df = _load_dataframe(config.get("input", {}))

    for item in config.get("pipeline", []):
        name = item.get("step")
        params = item.get("params", {}) or {}
        if name not in STEP_REGISTRY:
            raise ValueError(f"Unknown step: {name}")
        fn = STEP_REGISTRY[name]
        df, ctx = fn(df, ctx, **params)

    out_cfg = config.get("output", {})
    if out_cfg.get("save_csv", False):
        csv_path = out_cfg.get("csv_path", "./out/final_output.csv")
        ensure_parent_dir(csv_path)
        df.to_csv(csv_path, index=False)
        ctx["final_output"] = csv_path

    return df, ctx


def run_with_config(config_path: str):
    """Helper to load and run from a config path."""
    config = load_yaml_config(config_path)
    return run_pipeline(config)


if __name__ == "__main__":
    # Example quick run from Spyder: adjust the path below
    cfg_path = r"configs/config.yaml"
    df_final, context = run_with_config(cfg_path)
    print("Pipeline done.")
    print("Artifacts:", context.get("artifacts", []))
    print("Final output:", context.get("final_output"))
