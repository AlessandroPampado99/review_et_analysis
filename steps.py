# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 09:40:17 2025

@author: aless
"""

# - Single file containing the registry and a few useful steps.
# - Add new steps by writing a function with the decorator @step("name").

from typing import Callable, Dict, List, Optional
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import re


# --- utility ---
def _slugify(text: str) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^\w\-\.]", "", text)
    return text


# ---------- Registry ----------

STEP_REGISTRY: Dict[str, Callable] = {}

def step(name: str):
    """Decorator to register pipeline steps by name."""
    def deco(fn: Callable):
        if name in STEP_REGISTRY:
            raise ValueError(f"Step '{name}' already registered.")
        STEP_REGISTRY[name] = fn
        return fn
    return deco


def _ensure_parent_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ---------- Basic steps ----------

@step("drop_columns")
def drop_columns(df: pd.DataFrame, ctx: dict, names: Optional[List[str]] = None, **_):
    """Drop given columns if present."""
    names = names or []
    cols = [c for c in names if c in df.columns]
    if cols:
        df = df.drop(columns=cols, errors="ignore")
    return df, ctx


@step("coerce_numeric")
def coerce_numeric(df: pd.DataFrame, ctx: dict, columns=None, **_):
    """Convert given columns to numeric (coerce errors to NaN)."""
    columns = columns or []
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, ctx


@step("filter_rows")
def filter_rows(df: pd.DataFrame, ctx: dict, query: Optional[str] = None, **_):
    """Filter rows using pandas 'query' expression."""
    if query:
        df = df.query(query)
    return df, ctx


@step("rename_columns")
def rename_columns(df: pd.DataFrame, ctx: dict, mapping: dict = None, **_):
    """Rename columns using a provided mapping."""
    mapping = mapping or {}
    df = df.rename(columns=mapping)
    return df, ctx


@step("select_columns")
def select_columns(df: pd.DataFrame, ctx: dict, names: Optional[List[str]] = None, **_):
    """Keep only the given columns (if they exist)."""
    names = names or []
    keep = [c for c in names if c in df.columns]
    if keep:
        df = df[keep]
    return df, ctx


# ---------- Stats steps ----------

@step("group_stats")
def group_stats(
    df: pd.DataFrame,
    ctx: dict,
    by: Optional[List[str]] = None,
    metrics: Optional[dict] = None,
    save_as: Optional[str] = None,
    **_,
):
    """
    Generic grouped statistics.
    Example metrics: {"CAPEX": ["mean", "median"], "efficiency": ["mean"]}
    """
    by = by or []
    metrics = metrics or {}
    if not by or not metrics:
        return df, ctx

    out = df.groupby(by).agg(metrics)
    out.columns = ['_'.join(col).strip() if isinstance(col, tuple) else str(col) for col in out.columns.values]
    out = out.reset_index()

    if save_as:
        _ensure_parent_dir(save_as)
        out.to_csv(save_as, index=False)
        ctx.setdefault("artifacts", []).append(save_as)

    return out, ctx


@step("stats_by_technology")
def stats_by_technology(
    df: pd.DataFrame,
    ctx: dict,
    technology_col: str = "Technology name",
    columns: Optional[List[str]] = None,          # e.g. ["CAPEX","OPEX"]
    metrics: Optional[List[str]] = None,          # e.g. ["mean","median","min","max","count","std"]
    output_dir: Optional[str] = None,             # default: ctx["base_out_dir"]
    filename_prefix: str = "statistics",
    replace_df: bool = False,                     # if True, merge results back and replace df
    **_,
):
    """
    Compute grouped statistics by technology for multiple columns.
    Saves one CSV per column and stores each table in ctx["stats"][col].
    """
    if technology_col not in df.columns:
        raise KeyError(f"Missing technology column: '{technology_col}'")

    columns = columns or []
    if not columns:
        return df, ctx

    metrics = metrics or ["mean", "median", "min", "max", "count", "std"]

    # Decide output directory
    base_out_dir = ctx.get("base_out_dir", "./out")
    out_dir = output_dir or base_out_dir
    _ensure_parent_dir(os.path.join(out_dir, "dummy.txt"))

    # Compute stats per column
    merged = None
    for col in columns:
        if col not in df.columns:
            # skip missing columns gracefully
            continue

        # groupby & agg
        out = df.groupby(technology_col)[col].agg(metrics).reset_index()

        # Save CSV: statistics_<col>.csv
        slug = _slugify(col)
        fname = os.path.join(out_dir, f"{filename_prefix}_{slug}.csv")
        _ensure_parent_dir(fname)
        out.to_csv(fname, index=False)
        ctx.setdefault("artifacts", []).append(fname)

        # Store in context
        ctx.setdefault("stats", {})[col] = out

        # Prepare merged df (optional)
        if replace_df:
            # rename metrics columns with suffix to avoid collisions when merging multiple cols
            rename_map = {m: f"{col}_{m}" for m in metrics}
            out_ren = out.rename(columns=rename_map)
            merged = out_ren if merged is None else merged.merge(out_ren, on=technology_col, how="outer")

    # Replace df only if requested and at least one col processed
    if replace_df and merged is not None:
        return merged, ctx

    # Otherwise passthrough original df
    return df, ctx




# ---------- Plot steps ----------

@step("plot_capex_bar_by_technology")
def plot_capex_bar_by_technology(
    df: pd.DataFrame,
    ctx: dict,
    technology_col: str = "Technology name",
    value_col: str = "mean",
    top_n: Optional[int] = 20,
    out_path: str = "./out/capex_bar.png",
    **_,
):
    """
    Plot a bar chart using the aggregated CAPEX statistics (e.g., mean).
    """
    # usa tabella stats se disponibile
    stats_df = ctx.get("capex_stats", df)
    if technology_col not in stats_df.columns or value_col not in stats_df.columns:
        raise KeyError(f"Columns '{technology_col}' and/or '{value_col}' not found in capex_stats.")

    series = stats_df.set_index(technology_col)[value_col].sort_values(ascending=False)
    if top_n and top_n > 0:
        series = series.head(top_n)

    _ensure_parent_dir(out_path)
    plt.figure()
    series.plot(kind="bar")
    plt.xlabel(technology_col)
    plt.ylabel(value_col)
    plt.title(f"{value_col} by {technology_col}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    ctx.setdefault("artifacts", []).append(out_path)
    return df, ctx


@step("plot_capex_gaussian_by_technology")
def plot_capex_gaussian_by_technology(
    df: pd.DataFrame,
    ctx: dict,
    technology_col: str = "Technology name",
    mean_col: str = "mean",
    std_col: str = "std",
    tech_select: Optional[List[str]] = None,  # ["all"] or list
    out_dir: str = "./out/capex_gaussians",
    x_sigma_span: float = 4.0,
    **_,
):
    """
    Plot Gaussian curves for CAPEX using mean and std from aggregated stats.
    One PNG per technology.
    """
    stats_df = ctx.get("capex_stats", df)
    for col in [technology_col, mean_col, std_col]:
        if col not in stats_df.columns:
            raise KeyError(f"Column '{col}' not found in capex_stats.")

    # selezione tecnologie
    if tech_select is None or (len(tech_select) == 1 and tech_select[0].lower() == "all"):
        tech_list = sorted(stats_df[technology_col].dropna().unique().tolist())
    else:
        tech_list = tech_select

    _ensure_parent_dir(os.path.join(out_dir, "dummy.txt"))

    for tech in tech_list:
        row = stats_df[stats_df[technology_col] == tech]
        if row.empty:
            continue
        mu = float(row[mean_col].iloc[0])
        sigma = float(row[std_col].iloc[0])
        if sigma <= 0:
            continue

        xs = np.linspace(mu - x_sigma_span * sigma, mu + x_sigma_span * sigma, 400)
        pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)

        fname = os.path.join(out_dir, f"gaussian_{_slugify(tech)}.png")
        plt.figure()
        plt.plot(xs, pdf)
        plt.xlabel("CAPEX")
        plt.ylabel("Density")
        plt.title(f"{tech} — Normal(μ={mu:.2f}, σ={sigma:.2f})")
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()
        ctx.setdefault("artifacts", []).append(fname)

    return df, ctx

